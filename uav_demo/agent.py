"""Double DQN：当前状态与学习预测特征、经验回放、目标网络。"""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import EnvConfig, TrainConfig
from .environment import ACTIONS, ACTION_NAMES, GridWorldEnv
from .model import QNetwork


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用，请使用 --device cpu。")
    return torch.device(device)


class ReplayBuffer:
    """按需存储经验，避免第一次运行就分配整个回放池。"""

    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        self.items = []
        self.next_index = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def add(self, observation, action, reward, next_observation, done, next_mask):
        transition = (
            np.array(observation, dtype=np.float32, copy=True),
            int(action),
            float(reward),
            np.array(next_observation, dtype=np.float32, copy=True),
            bool(done),
            np.array(next_mask, dtype=bool, copy=True),
        )
        if len(self.items) < self.capacity:
            self.items.append(transition)
        else:
            self.items[self.next_index] = transition
        self.next_index = (self.next_index + 1) % self.capacity

    def sample(self, batch_size: int):
        indices = self.rng.choice(len(self.items), size=batch_size, replace=False)
        batch = [self.items[int(index)] for index in indices]
        return tuple(np.stack(values) for values in zip(*batch))


class DQNAgent:
    def __init__(self, observation_size: int, action_count: int, config: TrainConfig,
                 motion_predictor=None, env_config: EnvConfig | None = None):
        self.config = config
        self.observation_size = observation_size
        self.action_count = action_count
        self.motion_predictor = motion_predictor
        self.env_config = env_config
        self._prediction_cache = {}
        if motion_predictor is not None and (observation_size != 20 or env_config is None):
            raise ValueError("车辆预测特征需要 20 维当前状态和环境配置。")
        self.network_input_size = observation_size + (6 if motion_predictor is not None else 0)
        self.device = resolve_device(config.device)
        torch.set_num_threads(config.torch_threads)
        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        self.rng = np.random.default_rng(config.seed)
        network_options = {
            "hidden_sizes": config.hidden_sizes,
            "activation": config.activation,
            "leaky_relu_slope": config.leaky_relu_slope,
        }
        self.online = QNetwork(self.network_input_size, action_count, **network_options).to(self.device)
        self.target = QNetwork(self.network_input_size, action_count, **network_options).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(config.replay_capacity, config.seed + 1)
        self.environment_steps = 0
        self.gradient_steps = 0

    def predict_car(self, observation) -> np.ndarray | None:
        """仅根据当前观测预测 +4/+8/+12 步的均值位置，不读取环境未来轨迹。"""
        if self.motion_predictor is None:
            return None
        cfg = self.env_config
        scale = cfg.grid_size - 1
        car = tuple(int(round(float(v) * scale)) for v in observation[5:7])
        goal = tuple(int(round(float(v) * scale)) for v in observation[7:9])
        velocity = tuple(int(round(float(v))) for v in observation[17:19])
        time = int(round(float(observation[9]) * cfg.max_steps))
        key = (car, velocity, goal, time % cfg.car_move_every)
        if key not in self._prediction_cache:
            if len(self._prediction_cache) >= 50_000:
                self._prediction_cache.clear()
            self._prediction_cache[key] = self.motion_predictor.predict_positions(
                car, velocity, goal, cfg.grid_size,
                steps=(4, 8, 12), move_every=cfg.car_move_every, time=time,
            )
        return self._prediction_cache[key].copy()

    def _features(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        predicted = self.predict_car(observation)
        if predicted is None:
            return observation
        scale = self.env_config.grid_size - 1
        relative = predicted / scale - observation[:2]
        return np.concatenate((observation, relative.reshape(-1))).astype(np.float32)

    def choose_action(self, observation, epsilon: float = 0.0, mask=None) -> int:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon 必须在 [0, 1] 内。")
        if mask is None:
            mask = np.ones(self.action_count, dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.action_count,) or not mask.any():
            raise ValueError("动作 mask 尺寸不正确，或没有可选动作。")
        # epsilon=0 的评估不消耗训练使用的随机数。
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.choice(np.flatnonzero(mask)))
        with torch.no_grad():
            inputs = torch.as_tensor(self._features(observation), dtype=torch.float32, device=self.device)
            q_values = self.online(inputs).cpu().numpy()
        return int(np.argmax(np.where(mask, q_values, -np.inf)))

    def remember(self, observation, action, reward, next_observation, done, next_mask):
        self.replay.add(self._features(observation), action, reward,
                        self._features(next_observation), done, next_mask)
        self.environment_steps += 1

    def update(self) -> float | None:
        """每调用一次最多执行一次梯度更新，频率由主训练循环控制。"""
        if len(self.replay) < self.config.batch_size:
            return None
        observations, actions, rewards, next_observations, dones, next_masks = self.replay.sample(
            self.config.batch_size
        )
        observations = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        next_observations = torch.as_tensor(next_observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        next_masks = torch.as_tensor(next_masks, dtype=torch.bool, device=self.device)
        predicted = self.online(observations).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(next_observations).masked_fill(~next_masks, -torch.inf).argmax(dim=1)
            next_q = self.target(next_observations).gather(1, next_actions[:, None]).squeeze(1)
            # horizon 是任务的一部分，terminated 和 truncated 都不再 bootstrap。
            next_q = torch.where(dones, torch.zeros_like(next_q), next_q)
            expected = rewards + self.config.gamma * next_q
        loss = self.loss_fn(predicted, expected)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_update_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path: str | Path, env_config: EnvConfig):
        """保存推理用模型及配置；回放池未保存，不提供断点续训语义。"""
        expected_env = GridWorldEnv(env_config)
        if self.action_count != expected_env.action_count or self.observation_size != expected_env.observation_size:
            raise ValueError("模型输入或动作数量与保存的环境配置不一致，不能保存。")
        if self.motion_predictor is not None and self.env_config != env_config:
            raise ValueError("预测特征使用的环境配置与保存配置不一致，不能保存。")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 4,
                "observation_mode": "current_state",
                "network_input_size": self.network_input_size,
                "motion_predictor": self.motion_predictor.export_state() if self.motion_predictor is not None else None,
                "observation_size": self.observation_size,
                "action_count": self.action_count,
                "action_names": list(ACTION_NAMES),
                "actions": [list(action) for action in ACTIONS],
                "env_config": asdict(env_config),
                "train_config": asdict(self.config),
                "online_state_dict": self.online.state_dict(),
                "target_state_dict": self.target.state_dict(),
                "environment_steps": self.environment_steps,
                "gradient_steps": self.gradient_steps,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> tuple["DQNAgent", EnvConfig]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("模型文件必须包含 checkpoint 字典。")
        if checkpoint.get("format_version") in (1, 2, 3):
            raise ValueError(
                "旧版模型不兼容当前状态观测与学习预测输入；"
                "请使用当前代码重新训练，不能直接沿用旧权重。"
            )
        if checkpoint.get("format_version") != 4:
            raise ValueError("不支持的模型文件版本。")
        if checkpoint.get("observation_mode") != "current_state":
            raise ValueError("模型观测模式不兼容，当前只允许当前状态观测。")
        if (
            checkpoint.get("action_count") != len(ACTIONS)
            or checkpoint.get("action_names") != list(ACTION_NAMES)
            or checkpoint.get("actions") != [list(action) for action in ACTIONS]
        ):
            raise ValueError("模型的动作数量、顺序或动作定义与当前环境不兼容，请重新训练。")
        try:
            env_values = dict(checkpoint["env_config"])
            env_values.setdefault("progress_to_contact_boundary", False)
            env_values.setdefault("movement_penalty", 0.0)
            env_config = EnvConfig(**env_values)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("模型中的环境配置缺失或无效。") from error
        expected_env = GridWorldEnv(env_config)
        if checkpoint.get("observation_size") != expected_env.observation_size:
            raise ValueError("模型观测维度与保存的环境配置不一致，请重新训练。")
        try:
            train_values = dict(checkpoint["train_config"])
            # 必须显式保存结构，不能悄悄采用未来版本的新默认值。
            for field in ("hidden_sizes", "activation", "leaky_relu_slope"):
                if field not in train_values:
                    raise ValueError(f"缺少网络配置 {field}")
            train_values["device"] = device
            train_config = TrainConfig(**train_values)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("模型中的网络或训练配置缺失或无效。") from error
        predictor = None
        if checkpoint.get("motion_predictor") is not None:
            from .prediction import MotionPredictor
            predictor = MotionPredictor.from_state(checkpoint["motion_predictor"])
        agent = cls(checkpoint["observation_size"], checkpoint["action_count"], train_config,
                    motion_predictor=predictor, env_config=env_config)
        if checkpoint.get("network_input_size") != agent.network_input_size:
            raise ValueError("模型预测特征维度与网络输入不一致。")
        try:
            agent.online.load_state_dict(checkpoint["online_state_dict"])
            agent.target.load_state_dict(checkpoint["target_state_dict"])
        except (KeyError, RuntimeError, TypeError) as error:
            raise ValueError("模型权重与保存的网络结构不一致，请使用匹配的模型或重新训练。") from error
        agent.online.eval()
        agent.environment_steps = checkpoint["environment_steps"]
        agent.gradient_steps = checkpoint["gradient_steps"]
        return agent, env_config
