"""训练入口：单文件 PyTorch PPO（连续动作、对角高斯策略）；--residual > 0 时为残差控制。

流程：并行采样 n_envs × n_steps 步 → GAE 计算优势和回报 → 打乱分小批，按裁剪目标更新 n_epochs 轮。
超参数取常用的 PPO 默认值：策略/价值网络各 2 层 64 单元 tanh、
正交初始化、优势按小批标准化、动作采样后截断到 [-1, 1] 再交给环境（对数概率用截断前的动作）、
Adam eps=1e-5、梯度裁剪 0.5。每次训练保存 config.json、model.pt、curve.json 和源码副本。
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .config import Config
from .env import DelayEnv
from .evaluation import TEST, TRAINING, VALIDATION, evaluate, load_config


@dataclass(frozen=True)
class PPOConfig:
    n_envs: int = 4
    n_steps: int = 256  # 每个环境每轮采样步数
    batch_size: int = 64
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    log_std_init: float = 0.0  # 残差控制用 -1：初始修正量较小


def layer(inputs, outputs, gain):
    linear = nn.Linear(inputs, outputs)
    nn.init.orthogonal_(linear.weight, gain)
    nn.init.zeros_(linear.bias)
    return linear


class ActorCritic(nn.Module):
    """策略网络输出动作均值，标准差为独立可学习参数；价值网络独立。"""

    def __init__(self, obs_size, action_size, log_std_init):
        super().__init__()
        hidden = np.sqrt(2)
        self.actor = nn.Sequential(layer(obs_size, 64, hidden), nn.Tanh(),
                                   layer(64, 64, hidden), nn.Tanh(), layer(64, action_size, 0.01))
        self.critic = nn.Sequential(layer(obs_size, 64, hidden), nn.Tanh(),
                                    layer(64, 64, hidden), nn.Tanh(), layer(64, 1, 1.0))
        self.log_std = nn.Parameter(torch.full((action_size,), float(log_std_init)))

    def distribution(self, obs):
        return Normal(self.actor(obs), self.log_std.exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def act(self, obs):
        """评估用确定性动作：策略均值，截断到动作范围。"""
        with torch.no_grad():
            mean = self.actor(torch.as_tensor(obs, dtype=torch.float32))
        return np.clip(mean.numpy(), -1.0, 1.0)


def collect(envs, obs, model, ppo):
    """每个环境采样 n_steps 步；回合结束立即重置（env 只有终止，没有截断）。"""
    shape = (ppo.n_steps, ppo.n_envs)
    buffer = {key: torch.zeros(shape + extra) for key, extra in (
        ("obs", obs.shape[1:]), ("actions", (2,)), ("logp", ()), ("rewards", ()),
        ("dones", ()), ("values", ()))}
    for t in range(ppo.n_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            dist = model.distribution(obs_t)
            action = dist.sample()
            buffer["logp"][t] = dist.log_prob(action).sum(-1)
            buffer["values"][t] = model.value(obs_t)
        buffer["obs"][t], buffer["actions"][t] = obs_t, action
        clipped = action.clamp(-1.0, 1.0).numpy()
        for i, env in enumerate(envs):
            next_obs, reward, terminated, truncated, _ = env.step(clipped[i])
            buffer["rewards"][t, i], buffer["dones"][t, i] = reward, float(terminated or truncated)
            obs[i] = env.reset()[0] if terminated or truncated else next_obs
    return buffer, obs


def advantages(buffer, last_value, ppo):
    """GAE：从后往前累积 TD 误差；回合结束处不向下一状态自举。"""
    adv = torch.zeros_like(buffer["rewards"])
    running = torch.zeros(ppo.n_envs)
    for t in reversed(range(ppo.n_steps)):
        next_value = last_value if t == ppo.n_steps - 1 else buffer["values"][t + 1]
        alive = 1.0 - buffer["dones"][t]
        delta = buffer["rewards"][t] + ppo.gamma * next_value * alive - buffer["values"][t]
        running = delta + ppo.gamma * ppo.gae_lambda * alive * running
        adv[t] = running
    return adv, adv + buffer["values"]


def update(model, optimizer, buffer, adv, returns, ppo, rng):
    flat = {key: value.reshape(-1, *value.shape[2:]) for key, value in buffer.items()}
    adv, returns = adv.reshape(-1), returns.reshape(-1)
    for _ in range(ppo.n_epochs):
        order = rng.permutation(len(adv))
        for start in range(0, len(order), ppo.batch_size):
            idx = torch.as_tensor(order[start:start + ppo.batch_size])
            dist = model.distribution(flat["obs"][idx])
            logp = dist.log_prob(flat["actions"][idx]).sum(-1)
            ratio = (logp - flat["logp"][idx]).exp()
            batch_adv = (adv[idx] - adv[idx].mean()) / (adv[idx].std() + 1e-8)
            policy_loss = -torch.min(ratio * batch_adv,
                                     ratio.clamp(1 - ppo.clip_range, 1 + ppo.clip_range) * batch_adv).mean()
            value_loss = (returns[idx] - model.value(flat["obs"][idx])).pow(2).mean()
            entropy = dist.entropy().sum(-1).mean()
            loss = policy_loss + ppo.vf_coef * value_loss - ppo.ent_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), ppo.max_grad_norm)
            optimizer.step()


def torch_policy(model):
    """评估用策略函数 policy(obs, env)。"""
    return lambda obs, env: model.act(obs)


def load_run(run_dir):
    """读取训练目录：环境配置、延迟和模型。"""
    run_dir = Path(run_dir)
    cfg, delay = load_config(run_dir)
    ppo = PPOConfig(**json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["ppo"])
    obs_size = DelayEnv(delay, cfg).observation_space.shape[0]
    model = ActorCritic(obs_size, 2, ppo.log_std_init)
    model.load_state_dict(torch.load(run_dir / "model.pt", weights_only=True))
    model.eval()
    return cfg, delay, model


def train(delay, cfg, steps, seed, ppo, output=None, eval_every=50_000):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    envs = [DelayEnv(delay, cfg, TRAINING) for _ in range(ppo.n_envs)]
    obs = np.stack([env.reset(seed=seed + i)[0] for i, env in enumerate(envs)])
    model = ActorCritic(obs.shape[1], 2, ppo.log_std_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo.learning_rate, eps=1e-5)
    curve, done_steps, next_eval = [], 0, eval_every
    while done_steps < steps:
        buffer, obs = collect(envs, obs, model, ppo)
        with torch.no_grad():
            last_value = model.value(torch.as_tensor(obs, dtype=torch.float32))
        adv, returns = advantages(buffer, last_value, ppo)
        update(model, optimizer, buffer, adv, returns, ppo, rng)
        done_steps += ppo.n_steps * ppo.n_envs
        if output is not None and done_steps >= next_eval:
            next_eval += eval_every
            result = evaluate(torch_policy(model), delay, cfg, VALIDATION)
            row = {k: v for k, v in result.items() if k != "episodes_detail"}
            row["timesteps"] = done_steps
            curve.append(row)
            print(json.dumps(row), flush=True)
            (output / "curve.json").write_text(json.dumps(curve, indent=2))
    return model, done_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=int, default=6)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--residual", type=float, default=Config.residual)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evaluate", type=Path, help="已有训练目录；读取其配置和模型")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    torch.set_num_threads(1)

    if args.evaluate:
        output = args.evaluate
        cfg, delay, model = load_run(output)
    else:
        if args.output is None:
            parser.error("training requires --output")
        output = args.output
        output.mkdir(parents=True, exist_ok=False)
        cfg, delay = Config(residual=args.residual), args.delay
        ppo = PPOConfig(log_std_init=-1.0 if cfg.residual > 0 else 0.0)
        metadata = dict(model_version="delay_torch_v2", environment=asdict(cfg), delay=delay,
                        training_seed=args.seed, training_scenarios=list(TRAINING),
                        steps_requested=args.steps, ppo=asdict(ppo))
        (output / "config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
        # 保存这次运行的全部源码，便于确认结果对应哪一版规则。
        (output / "source").mkdir()
        for source in Path(__file__).parent.glob("*.py"):
            (output / "source" / source.name).write_bytes(source.read_bytes())
        model, metadata["steps_actual"] = train(delay, cfg, args.steps, args.seed, ppo, output)
        torch.save(model.state_dict(), output / "model.pt")
        (output / "config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                            encoding="utf-8")

    model.eval()
    seeds = VALIDATION if args.split == "validation" else TEST
    result = evaluate(torch_policy(model), delay, cfg, seeds)
    (output / (args.split + ".json")).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "episodes_detail"}, indent=2))


if __name__ == "__main__":
    main()
