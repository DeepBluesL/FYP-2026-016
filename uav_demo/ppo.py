"""单进程 PPO-Clip，使用当前策略采样和同一份合法动作掩码更新。"""

from dataclasses import asdict, dataclass, replace

import torch
from torch.nn import functional as F

from .agent import ObservationAgent, restore_predictor
from .config import EnvConfig, TrainConfig
from .environment import ACTIONS
from .ppo_model import PPOActorCritic


@dataclass
class RolloutStep:
    features: torch.Tensor
    mask: torch.Tensor
    action: int
    value: float
    logp: float
    reward: float
    done: bool


def trajectory_targets(rollout, last_value, gamma, gae_lambda):
    """倒推 GAE 和折扣回报。done 标记 FYP 任务结束，包括用尽任务时间。"""
    advantages = [0.0] * len(rollout)
    returns = [0.0] * len(rollout)
    next_value = next_return = last_value
    next_advantage = 0.0
    for i in reversed(range(len(rollout))):
        step = rollout[i]
        discount = 0.0 if step.done else gamma
        delta = step.reward + discount * next_value - step.value
        advantages[i] = delta + discount * gae_lambda * next_advantage
        returns[i] = step.reward + discount * next_return
        next_value, next_advantage, next_return = step.value, advantages[i], returns[i]
    return advantages, returns


class PPOAgent(ObservationAgent):
    def __init__(self, observation_size, action_count, config, motion_predictor=None, *, env_config):
        default_iters = config.update_iters if config.update_iters is not None else 80
        config = replace(
            config, algorithm="ppo", activation=config.activation or "tanh",
            train_pi_iters=config.train_pi_iters if config.train_pi_iters is not None else default_iters,
            train_v_iters=config.train_v_iters if config.train_v_iters is not None else default_iters,
            update_iters=None,
        )
        if min(config.rollout_steps, config.train_pi_iters, config.train_v_iters) < 1:
            raise ValueError("PPO 的采样步数和更新次数至少为 1")
        if config.rule_prior:
            raise ValueError("rule_prior 仅用于 DQN")
        super().__init__(observation_size, action_count, config, motion_predictor, env_config=env_config)
        self.model = PPOActorCritic(
            self.network_input_size, action_count, config.hidden_sizes,
            config.activation, config.leaky_relu_slope
        ).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=config.policy_lr)
        self.value_optimizer = torch.optim.Adam(self.model.critic.parameters(), lr=config.value_lr)
        self.rollout = []
        self.environment_steps = 0
        self.gradient_steps = 0

    def parameter_counts(self):
        actor = sum(p.numel() for p in self.model.actor.parameters())
        critic = sum(p.numel() for p in self.model.critic.parameters())
        return {"actor": actor, "critic": critic, "optimized_total": actor + critic,
                "stored_total": actor + critic}

    def features(self, observation):
        return torch.as_tensor(self._features(observation), device=self.device)

    @torch.no_grad()
    def choose_action(self, observation, epsilon=0.0, mask=None):
        # 共用评估接口，选择合法动作中概率最大的一个
        policy = self.model.distribution(self.features(observation), torch.as_tensor(mask, device=self.device))
        return policy.probs.argmax().item()

    def collect_step(self, env, observation, mask):
        features = self.features(observation)
        mask = torch.as_tensor(mask, device=self.device)
        action, value, logp = self.model.step(features, mask)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        self.rollout.append(RolloutStep(features, mask, action, value, logp, reward, done))
        self.environment_steps += 1
        return next_obs, reward, done, info

    def update(self, next_observation):
        cfg = self.config
        batch = self.rollout
        with torch.no_grad():
            last_value = 0.0 if batch[-1].done else self.model.value(self.features(next_observation)).item()
        advantages, returns = trajectory_targets(batch, last_value, cfg.gamma, cfg.gae_lambda)
        features = torch.stack([step.features for step in batch])
        masks = torch.stack([step.mask for step in batch])
        actions = torch.tensor([step.action for step in batch], device=self.device)
        old_logp = torch.tensor([step.logp for step in batch], device=self.device)
        adv = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        targets = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if len(batch) > 1:
            adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)

        # 同一批固定回报目标，记录当前 Critic 在拟合前的表现。
        with torch.no_grad():
            values_before = self.model.value(features)
            value_loss_before = F.mse_loss(values_before, targets).item()
            target_variance = targets.var(unbiased=False)
            explained_variance = (
                (1 - (targets - values_before).var(unbiased=False) / target_variance).item()
                if target_variance.item() > 0 else float("nan")
            )

        policy_updates = 0
        for _ in range(cfg.train_pi_iters):
            logp = self.model.distribution(features, masks).log_prob(actions)
            approx_kl = (old_logp - logp).mean().item()
            if approx_kl > 1.5 * cfg.target_kl:
                break
            ratio = (logp - old_logp).exp()
            clipped = ratio.clamp(1 - cfg.clip_ratio, 1 + cfg.clip_ratio)
            policy_loss = -torch.minimum(ratio * adv, clipped * adv).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            policy_updates += 1
            self.gradient_steps += 1

        for _ in range(cfg.train_v_iters):
            value_loss = F.mse_loss(self.model.value(features), targets)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            self.gradient_steps += 1
        # after 指标使用更新后的网络，before 和 explained_variance 使用更新前的估值。
        with torch.no_grad():
            policy = self.model.distribution(features, masks)
            logp = policy.log_prob(actions)
            ratio = (logp - old_logp).exp()
            clipped = ratio.clamp(1 - cfg.clip_ratio, 1 + cfg.clip_ratio)
            value_loss_after = F.mse_loss(self.model.value(features), targets).item()
            metrics = {
                "batch_steps": len(batch),
                "policy_loss": -torch.minimum(ratio * adv, clipped * adv).mean().item(),
                "value_loss": value_loss_after,
                "value_loss_before": value_loss_before,
                "value_loss_after": value_loss_after,
                "explained_variance": explained_variance,
                "approx_kl": (old_logp - logp).mean().item(),
                "entropy": policy.entropy().mean().item(),
                "clip_fraction": ((ratio < 1 - cfg.clip_ratio) | (ratio > 1 + cfg.clip_ratio)).float().mean().item(),
                "policy_updates": policy_updates,
                "value_updates": cfg.train_v_iters,
            }
        batch.clear()
        return metrics

    def save(self, path, env_config):
        torch.save({
            "algorithm": "ppo", "format_version": 2, "parameter_counts": self.parameter_counts(), "actions": ACTIONS, "observation_size": self.observation_size,
            "env_config": asdict(env_config), "train_config": asdict(self.config),
            "run_metadata": self.run_metadata,
            "model": self.model.state_dict(), "environment_steps": self.environment_steps,
            "gradient_steps": self.gradient_steps,
            "motion_predictor": self.motion_predictor.export_state() if self.motion_predictor is not None else None,
        }, path)

    @classmethod
    def load(cls, path, device="cpu"):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("algorithm") != "ppo" or saved.get("actions") != ACTIONS:
            raise ValueError("请选择与当前动作定义匹配的 PPO checkpoint")
        env_config = EnvConfig(**saved["env_config"])
        config = TrainConfig(**{**saved["train_config"], "device": device, "prediction_device": device})
        if saved.get("format_version", 1) == 1:
            config.hidden_sizes = (64, 64)
            config.activation = "tanh"
        predictor = restore_predictor(saved["motion_predictor"], device)
        agent = cls(saved["observation_size"], len(ACTIONS), config, predictor, env_config=env_config)
        agent.model.load_state_dict(saved["model"])
        agent.model.eval()
        agent.environment_steps = saved["environment_steps"]
        agent.run_metadata = saved.get("run_metadata")
        agent.gradient_steps = saved["gradient_steps"]
        return agent, env_config
