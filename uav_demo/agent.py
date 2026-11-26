"""Double DQN：当前状态与学习预测特征、经验回放、目标网络。"""

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import EnvConfig, TrainConfig
from .environment import (
    ACTIONS, ACTION_NAMES, GridWorldEnv,

)
from .observation import encode_observation, feature_slice
from .extrapolate_pursuit import ExtrapolatePursuit, predicted_car
from .linear_prediction import LinearPredictor
from .oracle_prediction import OraclePredictor
from .prediction import forecast_vehicle
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


def restore_predictor(saved_predictor, device):
    predictor = None

    if saved_predictor is not None:
        if saved_predictor.get("model_type") == "oracle":
            predictor = OraclePredictor()
        elif saved_predictor.get("model_type") == "linear":
            predictor = LinearPredictor()
        elif saved_predictor.get("model_type") == "goal_filter":
            from .goal_inference import GoalFilterPredictor
            predictor = GoalFilterPredictor.from_state(saved_predictor)
        elif saved_predictor.get("model_type") == "direction_lstm":
            from .direction_prediction import DirectionPredictor
            predictor = DirectionPredictor.from_state(saved_predictor, device=device)
        elif saved_predictor.get("model_type") == "tcn":
            from .tcn_prediction import TCNPredictor
            from .sequence_prediction import prediction_device
            predictor = TCNPredictor(
                hidden_size=saved_predictor["hidden_size"],
                head_size=saved_predictor["head_size"],
                num_blocks=saved_predictor["num_blocks"],
                kernel_size=saved_predictor["kernel_size"],
                max_steps=saved_predictor["max_steps"],
            )
            predictor.load_state_dict(saved_predictor["state_dict"])
            predictor.to(prediction_device(device))
            predictor.eval()
        elif saved_predictor.get("model_type") == "var":
            from .var_prediction import VARPredictor
            predictor = VARPredictor(lags=saved_predictor["lags"])
            predictor.coefficients = np.asarray(saved_predictor["coefficients"], dtype=np.float64)
        elif saved_predictor.get("model_type") in ("rnn", "lstm", "gru"):
            from .sequence_prediction import SequencePredictor
            predictor = SequencePredictor.from_state(saved_predictor, device=device)
        else:
            from .prediction import MotionPredictor
            predictor = MotionPredictor.from_state(saved_predictor)
    return predictor


class ObservationAgent:
    def __init__(self, observation_size: int, action_count: int, config: TrainConfig,
                 motion_predictor=None, *, env_config: EnvConfig):
        self.config = config
        self.run_metadata = None
        self.observation_size = observation_size
        self.action_count = action_count
        self.motion_predictor = motion_predictor
        self.env_config = env_config
        self._prediction_cache = {}
        feature_size = observation_size + (
            2 * len(config.prediction_steps)
            if motion_predictor is not None else 0
        )

        self.device = resolve_device(config.device)
        # rule_action_input=False：规则动作仍留在特征向量里（经验里也存着，先验要读），只是不进网络。
        self._network_columns = None
        if not config.rule_action_input:
            rule = feature_slice(config.observation_features, "rule_action")
            self._network_columns = torch.as_tensor(
                [i for i in range(feature_size) if not rule.start <= i < rule.stop], device=self.device)
        self.network_input_size = feature_size if self._network_columns is None else len(self._network_columns)
        torch.set_num_threads(config.torch_threads)
        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)

    def predict_car(self, observation, steps=None) -> np.ndarray | None:
        """steps 默认取 config.prediction_steps；规则用 steps=(1,) 取下一步车位。"""
        if self.motion_predictor is None:
            return None

        cfg = self.env_config
        state = observation
        steps = self.config.prediction_steps if steps is None else tuple(steps)
        if getattr(self.motion_predictor, "model_type", None) == "oracle":
            return self.motion_predictor.forecast(state, cfg.max_steps, steps)
        if not state["has_observation"]:
            return None
        age = state["observation_age"]
        observed_time = state["time"] - age

        key = (
            tuple((t, tuple(p)) for t, p in state["car_history"]),
            observed_time % cfg.car_move_every,
            tuple(age + h for h in steps),
        )

        if key not in self._prediction_cache:
            if len(self._prediction_cache) >= 50_000:
                self._prediction_cache.clear()

            self._prediction_cache[key] = forecast_vehicle(
                self.motion_predictor, state, cfg, steps
            )

        return self._prediction_cache[key].copy()

    def rule_target(self, observation):
        """规则追的下一步车位；无报告时为 None。"""
        if not observation["has_observation"]:
            return None
        source = self.config.rule_source
        if source == "oracle":
            if "true_car_trajectory" not in observation:
                raise ValueError("真实位置对照需要以 expose_truth=True 创建环境")
            point = observation["true_car_trajectory"][min(observation["time"] + 1, self.env_config.max_steps)]
            return tuple(int(v) for v in point)
        if source == "predictor":
            return tuple(float(v) for v in self.predict_car(observation, steps=(1,))[0])
        return predicted_car(observation, self.env_config)

    def rule_action(self, observation, mask=None) -> int:
        if mask is None:
            mask = np.ones(self.action_count, dtype=bool)
        return ExtrapolatePursuit(self.env_config).choose_action(observation, mask, self.rule_target(observation))

    def _features(self, observation):
        features = encode_observation(observation, self.env_config, self.config.observation_features)
        if self.config.rule_source != "linear" and "rule_action" in self.config.observation_features:
            # 编码时按线性外推填了规则动作，这里换成本 agent 的规则来源。
            rule = feature_slice(self.config.observation_features, "rule_action")
            features[rule] = 0.0
            features[rule.start + self.rule_action(observation)] = 1.0
        if self.motion_predictor is not None:
            predicted = self.predict_car(observation)
            if predicted is None:
                relative = np.zeros((len(self.config.prediction_steps), 2), dtype=np.float32)
            else:
                scale = self.env_config.grid_size - 1
                relative = predicted / scale - np.asarray(
                    [v / scale for v in observation["uav_pos"]], dtype=np.float32
                )
            features = np.concatenate((features, relative.reshape(-1)))
        return features.astype(np.float32)

class DQNAgent(ObservationAgent):
    def __init__(self, observation_size: int, action_count: int, config: TrainConfig,
                 motion_predictor=None, *, env_config: EnvConfig):
        config = replace(config, algorithm="dqn", activation=config.activation or "leaky_relu")
        super().__init__(observation_size, action_count, config, motion_predictor, env_config=env_config)
        # rule_prior > 0：选动作和下一步目标中加入外推规则先验（规则动作 0、其余 −rule_prior），
        # 被拟合的 Q 不含先验，偏离规则的代价在训练中持续存在（参考残差 Q 学习）。
        self._rule_slice = (feature_slice(config.observation_features, "rule_action")
                            if config.rule_prior > 0 else None)

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

    def parameter_counts(self):
        online = sum(p.numel() for p in self.online.parameters())
        target = sum(p.numel() for p in self.target.parameters())
        return {"online": online, "target": target, "optimized_total": online,
                "stored_total": online + target}

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
            q_values = self._q(self.online, inputs).cpu().numpy()
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
        # 被拟合的是不含先验的残差 Q；先验只进入动作选择和下一步目标。
        fitted = self._network(self.online, observations)
        predicted = fitted.gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_actions = self._q(self.online, next_observations).masked_fill(~next_masks, -torch.inf).argmax(dim=1)
            next_q = self._q(self.target, next_observations).gather(1, next_actions[:, None]).squeeze(1)
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

    def _network(self, network, features):
        """网络输出；rule_action_input=False 时先去掉特征里的规则动作列。"""
        if self._network_columns is not None:
            features = features.index_select(-1, self._network_columns)
        return network(features)

    def _q(self, network, features):
        """用于选动作和下一步目标的 Q 值；启用 rule_prior 时加上规则先验（规则 one-hot 位于特征向量中）。"""
        q = self._network(network, features)
        if self._rule_slice is not None:
            onehot = features[..., self._rule_slice]
            # log 概率形式：规则动作 0，其余 −rule_prior；用扣分而非加分，避免按步累积正奖励。
            q = q + self.config.rule_prior * (onehot - 1)
        return q

    def save(self, path: str | Path, env_config: EnvConfig):
        """保存推理用模型及配置；回放池未保存，不提供断点续训语义。"""
        expected_env = GridWorldEnv(env_config)
        if self.action_count != expected_env.action_count or self.observation_size != encode_observation(
                expected_env.reset(seed=0)[0], env_config, self.config.observation_features).size:
            raise ValueError("模型输入或动作数量与保存的环境配置不一致，不能保存。")
        if self.env_config is not None and self.env_config != env_config:
            raise ValueError("预测特征使用的环境配置与保存配置不一致，不能保存。")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 9,
                "parameter_counts": self.parameter_counts(),
                "observation_mode": "delayed_vehicle",
                "network_input_size": self.network_input_size,
                "motion_predictor": self.motion_predictor.export_state() if self.motion_predictor is not None else None,
                "observation_size": self.observation_size,
                "action_count": self.action_count,
                "action_names": list(ACTION_NAMES),
                "actions": [list(action) for action in ACTIONS],
                "env_config": asdict(env_config),
                "train_config": asdict(self.config),
                "run_metadata": self.run_metadata,
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

        legacy_v7 = (checkpoint.get("format_version") == 7
                     and "car_velocity" in checkpoint.get("train_config", {}).get("observation_features", ())
                     and checkpoint.get("motion_predictor") is None)
        if checkpoint.get("format_version") != 9 and not legacy_v7:
            raise ValueError("旧版模型与当前预测器结构及种子配置不兼容，请重新训练。")
        if checkpoint.get("observation_mode") != "delayed_vehicle":
            raise ValueError("模型观测模式与当前车辆观测接口不兼容。")

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
            env_values.pop("car_spawn_region", None)  # 旧模型保存过该字段，车辆起点已固定，不再使用
            env_config = EnvConfig(**env_values)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("模型中的环境配置缺失或无效。") from error
        expected_env = GridWorldEnv(env_config)

        try:
            train_values = dict(checkpoint["train_config"])
            # 必须显式保存结构，不能悄悄采用未来版本的新默认值。
            for field in ("hidden_sizes", "activation", "leaky_relu_slope"):
                if field not in train_values:
                    raise ValueError(f"缺少网络配置 {field}")
            # spec saved its one-step default; some historical exports omit default fields.
            spec_format = "rule_source" in train_values or "prediction_point" in train_values
            if spec_format:
                train_values.setdefault("prediction_steps", (1,))
                train_values.setdefault("prediction_model_type", "linear")
            train_values["device"] = device
            # 先验模式已固定为持续先验；只接受当时就用持续先验训练的模型。
            # rule_prior=0 时先验未启用，模式字段无意义，不拒绝。
            mode = train_values.pop("rule_prior_mode", "persistent")
            if mode != "persistent" and train_values.get("rule_prior", 0.0) > 0:
                raise ValueError("该模型使用已移除的 init 先验模式，请重新训练")
            removed = {"rule_prior_feature": "rule_action", "exploration": "uniform"}
            for field, default in removed.items():
                if train_values.pop(field, default) != default:
                    raise ValueError(f"该模型使用已移除的选项 {field}，请重新训练")
            train_values.pop("exploration_temperature", None)
            # 已移除的诊断选项（真实车位加噪声/平移）：取默认值时等于没用，可以忽略；用过的模型不再支持。
            for field, unused in (("rule_noise", 0.0), ("rule_noise_corr", 0.0), ("rule_offset", 0.0),
                                  ("prediction_oracle_noise", [])):
                if train_values.pop(field, unused) not in (unused, ()):
                    raise ValueError(f"该模型使用已移除的诊断选项 {field}，不再支持加载")
            # 2026-11 清理时移除的对照选项（接触先验、先验衰减、DRQN）：取默认值时等于没用；用过的模型不再支持。
            for field, unused in (("contact_prior", 0.0), ("rule_prior_decay_steps", 0), ("agent_type", "dqn")):
                if train_values.pop(field, unused) != unused:
                    raise ValueError(f"该模型使用已移除的选项 {field}，不再支持加载")
            for field in ("prediction_grid_turns", "contact_grid_turns"):
                train_values.pop(field, None)
            # 预测器留在训练时的设备上：方向 LSTM 在 CPU 和 GPU 上抽样的随机数不同，换设备会改变测试结果。
            # 只有训练时用 GPU、当前机器没有 CUDA 时，才退回到 device。
            if train_values.get("prediction_device", device) == "cuda" and not torch.cuda.is_available():
                train_values["prediction_device"] = device
            train_config = TrainConfig(**train_values)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("模型中的网络或训练配置缺失或无效。") from error
        expected_size = encode_observation(
            expected_env.reset(seed=0)[0], env_config, train_config.observation_features
        ).size
        if checkpoint.get("observation_size") != expected_size:
            raise ValueError("模型观测维度与保存的特征配置不一致，请重新训练。")
        if checkpoint.get("contact_model") is not None:
            raise ValueError("Unsupported historical contact prior")
        predictor = restore_predictor(checkpoint.get("motion_predictor"), train_config.prediction_device)
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
        agent.run_metadata = checkpoint.get("run_metadata")
        agent.gradient_steps = checkpoint["gradient_steps"]
        return agent, env_config
