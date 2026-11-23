"""DRQN 对照组：端到端的循环 Q 网络（参考 Hausknecht & Stone, 2015），不用运动预测器，也不用规则先验。

输入只有当前的命名观测特征（含延迟报告的车位、报告时长等），由 GRU 在一局之内自己积累历史；
经验回放按整局存储，每次更新抽若干整局、从零隐状态顺序展开（不截断，隐状态与行动时一致），
目标用 Double DQN，horizon 结束不再 bootstrap，与 DQNAgent 一致。
接口与 DQNAgent 相同：choose_action(observation, epsilon, mask)；每局 time=0 的观测会自动重置隐状态，
所以 evaluate.evaluate_episode 可以直接使用。
"""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import EnvConfig, TrainConfig
from .environment import ACTIONS
from .observation import encode_observation


class RecurrentQNetwork(nn.Module):
    def __init__(self, input_size: int, action_count: int, hidden_size: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LeakyReLU())
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(),
                                  nn.Linear(hidden_size, action_count))

    def forward(self, features, hidden=None):
        """features: [batch, time, input]；返回 Q [batch, time, action] 和最后的隐状态。"""
        output, hidden = self.gru(self.encoder(features), hidden)
        return self.head(output), hidden


class EpisodeReplay:
    """按整局存储；总步数超过容量时丢掉最早的整局。"""

    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        self.episodes = []
        self.steps = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.steps

    def add(self, episode: dict):
        self.episodes.append(episode)
        self.steps += len(episode["actions"])
        while self.steps > self.capacity and len(self.episodes) > 1:
            self.steps -= len(self.episodes.pop(0)["actions"])

    def sample(self, count: int) -> list[dict]:
        indices = self.rng.choice(len(self.episodes), size=min(count, len(self.episodes)), replace=False)
        return [self.episodes[int(i)] for i in indices]


class DRQNAgent:
    agent_type = "drqn"
    motion_predictor = None
    contact_model = None

    def __init__(self, observation_size: int, action_count: int, config: TrainConfig, *,
                 env_config: EnvConfig, hidden_size: int = 128, batch_episodes: int = 8):
        if config.rule_prior > 0 or config.contact_prior > 0 or config.use_motion_prediction:
            raise ValueError("DRQN 对照组是端到端的：不用运动预测器、规则先验或接触先验")
        self.config, self.env_config = config, env_config
        self.observation_size, self.action_count = observation_size, action_count
        self.network_input_size = observation_size
        self.hidden_size, self.batch_episodes = hidden_size, batch_episodes
        self.device = torch.device("cpu")
        torch.set_num_threads(config.torch_threads)
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.online = RecurrentQNetwork(observation_size, action_count, hidden_size)
        self.target = RecurrentQNetwork(observation_size, action_count, hidden_size)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = EpisodeReplay(config.replay_capacity, config.seed + 1)
        self.environment_steps = 0
        self.gradient_steps = 0
        self._hidden = None
        self._episode = None

    def _features(self, observation):
        return encode_observation(observation, self.env_config, self.config.observation_features)

    def choose_action(self, observation, epsilon: float = 0.0, mask=None) -> int:
        """隐状态每步都要更新（探索的步也一样），否则之后的 Q 值会建立在缺失的历史上。"""
        if observation["time"] == 0:
            self._hidden = None
        mask = np.ones(self.action_count, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        with torch.no_grad():
            inputs = torch.as_tensor(self._features(observation), dtype=torch.float32)[None, None]
            q, self._hidden = self.online(inputs, self._hidden)
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.choice(np.flatnonzero(mask)))
        return int(np.argmax(np.where(mask, q[0, 0].numpy(), -np.inf)))

    def remember(self, observation, action, reward, next_observation, done, next_mask):
        """逐步收集，一局结束（done）时整局放进回放池。"""
        if self._episode is None or observation["time"] == 0:
            self._episode = {"features": [self._features(observation)], "actions": [], "rewards": [],
                             "masks": []}
        episode = self._episode
        episode["features"].append(self._features(next_observation))
        episode["actions"].append(int(action))
        episode["rewards"].append(float(reward))
        episode["masks"].append(np.asarray(next_mask, dtype=bool))
        self.environment_steps += 1
        if done:
            self.replay.add({key: np.asarray(value) for key, value in episode.items()})
            self._episode = None

    def update(self) -> float | None:
        if not self.replay.episodes:
            return None
        episodes = self.replay.sample(self.batch_episodes)
        longest = max(len(e["actions"]) for e in episodes)
        batch = len(episodes)
        features = np.zeros((batch, longest + 1, self.observation_size), dtype=np.float32)
        actions = np.zeros((batch, longest), dtype=np.int64)
        rewards = np.zeros((batch, longest), dtype=np.float32)
        valid = np.zeros((batch, longest), dtype=bool)
        last = np.zeros((batch, longest), dtype=bool)  # 本局最后一步：不再 bootstrap
        masks = np.ones((batch, longest, self.action_count), dtype=bool)
        for index, episode in enumerate(episodes):
            length = len(episode["actions"])
            features[index, :length + 1] = episode["features"]
            actions[index, :length] = episode["actions"]
            rewards[index, :length] = episode["rewards"]
            masks[index, :length] = episode["masks"]
            valid[index, :length] = True
            last[index, length - 1] = True
        features, actions, rewards = (torch.as_tensor(v) for v in (features, actions, rewards))
        valid, last, masks = (torch.as_tensor(v) for v in (valid, last, masks))
        q, _ = self.online(features)
        predicted = q[:, :-1].gather(2, actions[..., None]).squeeze(2)
        with torch.no_grad():
            next_actions = q[:, 1:].masked_fill(~masks, -torch.inf).argmax(dim=2)
            target_q, _ = self.target(features)
            next_q = target_q[:, 1:].gather(2, next_actions[..., None]).squeeze(2)
            expected = rewards + self.config.gamma * torch.where(last, torch.zeros_like(next_q), next_q)
        loss = self.loss_fn(predicted[valid], expected[valid])
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_update_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path: str | Path, env_config: EnvConfig):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"agent_type": self.agent_type, "env_config": asdict(env_config),
                    "train_config": asdict(self.config), "observation_size": self.observation_size,
                    "action_count": self.action_count, "hidden_size": self.hidden_size,
                    "batch_episodes": self.batch_episodes,
                    "online_state_dict": self.online.state_dict(), "target_state_dict": self.target.state_dict(),
                    "environment_steps": self.environment_steps, "gradient_steps": self.gradient_steps}, path)

    @classmethod
    def load(cls, path: str | Path) -> tuple["DRQNAgent", EnvConfig]:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("agent_type") != cls.agent_type or saved.get("action_count") != len(ACTIONS):
            raise ValueError("不是 DRQN 模型，或动作定义与当前环境不一致")
        env_config = EnvConfig(**saved["env_config"])
        config = TrainConfig(**saved["train_config"])
        agent = cls(saved["observation_size"], saved["action_count"], config, env_config=env_config,
                    hidden_size=saved["hidden_size"], batch_episodes=saved["batch_episodes"])
        agent.online.load_state_dict(saved["online_state_dict"])
        agent.target.load_state_dict(saved["target_state_dict"])
        agent.online.eval()
        agent.environment_steps, agent.gradient_steps = saved["environment_steps"], saved["gradient_steps"]
        return agent, env_config


def load_agent(path):
    """按模型文件里的类型加载 DQN 或 DRQN。"""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("agent_type") == DRQNAgent.agent_type:
        return DRQNAgent.load(path)
    from .agent import DQNAgent
    return DQNAgent.load(path)
