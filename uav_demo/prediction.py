"""仅从当前可见车辆状态学习下一次移动方向，再递归预测未来位置均值。

离线监督标签来自调用方提供的训练种子，推理接口不接收真实未来轨迹。
均值预测是轻量近似，不代表一条必然发生、整格可执行的车辆路线。
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from .config import EnvConfig
from .vehicle import DIRECTIONS, generate_vehicle_trajectory


def make_features(position, velocity, goal, grid_size: int) -> np.ndarray:
    """8 维输入：当前位置、最近可见位移、终点、终点相对位置。

支持单个 ``(2,)`` 坐标和批量 ``(N, 2)`` 坐标。未观察到车辆移动时，
velocity 使用 (0, 0)；停顿帧沿用上次观察到的非零位移，不访问内部朝向。
"""
    if not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("grid_size 必须是至少为 2 的整数")
    position, velocity, goal = np.broadcast_arrays(
        np.asarray(position, dtype=np.float32),
        np.asarray(velocity, dtype=np.float32),
        np.asarray(goal, dtype=np.float32),
    )
    if position.shape[-1:] != (2,):
        raise ValueError("位置、位移和终点最后一维必须是 2")
    if not all(np.isfinite(value).all() for value in (position, velocity, goal)):
        raise ValueError("预测输入必须是有限数")
    scale = grid_size - 1
    return np.concatenate((position / scale, velocity, goal / scale,
                           (goal - position) / scale), axis=-1).astype(np.float32)


def _legal_directions(positions: np.ndarray, grid_size: int) -> np.ndarray:
    options = np.asarray(positions)[..., None, :] + DIRECTIONS
    return ((options >= 0) & (options <= grid_size - 1)).all(axis=-1)


def _examples(config: EnvConfig, seeds: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, labels, masks = [], [], []
    for seed in seeds:
        route, goal = generate_vehicle_trajectory(config, np.random.default_rng(seed))
        observed_velocity = np.zeros(2, dtype=np.float32)
        for time in range(1, len(route)):
            displacement = route[time] - route[time - 1]
            if not displacement.any():
                continue
            # route[t] 只用于训练标签；输入取发生该移动之前的可见状态。
            features.append(make_features(route[time - 1], observed_velocity, goal, config.grid_size))
            labels.append(int(np.flatnonzero((DIRECTIONS == displacement).all(axis=1))[0]))
            masks.append(_legal_directions(route[time - 1], config.grid_size))
            observed_velocity = displacement
    if not labels:
        raise ValueError("提供的种子没有产生可用于训练的车辆移动")
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), np.asarray(masks)


class MotionPredictor(nn.Module):
    """8 → 32 → 32 → 4 MLP，输出北/东/南/西四方向的 logits。"""

    feature_version = 1

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 4),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)

    def export_state(self) -> dict:
        """仅导出 CPU tensors 与基本类型，可用 torch.load(weights_only=True) 读取。"""
        return {
            "format_version": 1,
            "feature_version": self.feature_version,
            "directions": DIRECTIONS.tolist(),
            "state_dict": {key: value.detach().cpu().clone() for key, value in self.state_dict().items()},
        }

    @classmethod
    def from_state(cls, state: dict) -> "MotionPredictor":
        """恢复 CPU 模型，并核对输入及输出方向语义。"""
        if (not isinstance(state, dict) or state.get("format_version") != 1
                or state.get("feature_version") != cls.feature_version
                or state.get("directions") != DIRECTIONS.tolist()):
            raise ValueError("车辆预测器版本、输入特征或方向定义不兼容")
        with torch.random.fork_rng(devices=[]):
            predictor = cls()
        try:
            predictor.load_state_dict(state["state_dict"])
        except (KeyError, TypeError, RuntimeError) as error:
            raise ValueError("车辆预测器权重无效") from error
        predictor.eval()
        return predictor

    @classmethod
    def fit(
        cls, config: EnvConfig, seeds: Iterable[int], epochs: int = 20, *,
        seed: int = 0, batch_size: int = 256, learning_rate: float = 0.003,
    ) -> tuple["MotionPredictor", dict]:
        """按传入种子前 80% / 后 20% 分割拟合/内部验证，仅生成这些轨迹。

        调用方必须传入训练种子区间，不能传入 DQN 验证/测试种子。
        内部验证按整条路线划分，避免同一条路线的相邻帧跨集合泄漏。
        """
        seeds = list(seeds)
        if len(seeds) < 2 or any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s < 0 for s in seeds):
            raise ValueError("至少需要两个不同的非负整数训练种子")
        if len(set(seeds)) != len(seeds):
            raise ValueError("训练种子不能重复，否则内部验证会泄漏")
        if not isinstance(epochs, int) or epochs < 1 or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("epochs 和 batch_size 必须为正整数")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate 必须为有限正数")
        split = max(1, min(len(seeds) - 1, int(len(seeds) * 0.8)))
        fit_seeds, holdout_seeds = seeds[:split], seeds[split:]
        train_arrays = _examples(config, fit_seeds)
        holdout_arrays = _examples(config, holdout_seeds)
        train_x, train_y, train_mask = (torch.from_numpy(array) for array in train_arrays)
        holdout_x, holdout_y, holdout_mask = (torch.from_numpy(array) for array in holdout_arrays)
        # 不改变随后 DQN 初始化所使用的全局随机数序列。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            predictor = cls()
        optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        rng = np.random.default_rng(seed)
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            predictor.train()
            for _ in range(epochs):
                order = rng.permutation(len(train_y))
                for start in range(0, len(order), batch_size):
                    indices = order[start:start + batch_size]
                    logits = predictor(train_x[indices]).masked_fill(~train_mask[indices], -torch.inf)
                    loss = criterion(logits, train_y[indices])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            predictor.eval()
            with torch.no_grad():
                logits = predictor(holdout_x).masked_fill(~holdout_mask, -torch.inf)
                holdout_loss = float(criterion(logits, holdout_y))
                accuracy = float((logits.argmax(dim=1) == holdout_y).float().mean())
                # 直行基线只读取输入中的最近可见方向；初始零位移按北向处理。
                previous_direction = torch.as_tensor(DIRECTIONS, dtype=torch.float32)
                straight_logits = holdout_x[:, 2:4] @ previous_direction.T
                straight_logits = straight_logits.masked_fill(~holdout_mask, -torch.inf)
                straight_accuracy = float((straight_logits.argmax(dim=1) == holdout_y).float().mean())
        finally:
            torch.set_num_threads(previous_threads)
        metrics = {
            "fit_seeds": [int(s) for s in fit_seeds],
            "holdout_seeds": [int(s) for s in holdout_seeds],
            "fit_moves": len(train_y), "holdout_moves": len(holdout_y),
            "epochs": epochs, "holdout_cross_entropy": holdout_loss,
            "holdout_accuracy": accuracy, "straight_baseline_accuracy": straight_accuracy,
            "feature_version": cls.feature_version,
            "forecast_method": "recursive_probability_mean",
        }
        return predictor, metrics

    def predict_positions(
        self, position, velocity, goal, grid_size: int,
        steps: tuple[int, ...] = (4, 8, 12), move_every: int = 1, time: int = 0,
    ) -> np.ndarray:
        """返回各未来环境时距的预测位置均值，shape=(len(steps), 2)。

        每次可移动时对四方向 softmax 概率取期望位移，再递归传播位置和
        方向均值；这是近似均值传播，既不是 argmax 路线，也不是精确联合分布。
        只使用传入的当前状态和训练后的权重，绝不读取真实未来路线。
        """
        if not steps or any(isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 0 for step in steps):
            raise ValueError("steps 必须是非空的非负整数时距列表")
        if not isinstance(move_every, int) or move_every < 1 or not isinstance(time, int) or time < 0:
            raise ValueError("move_every 必须是正整数，time 必须是非负整数")
        position = np.asarray(position, dtype=np.float32).copy()
        velocity = np.asarray(velocity, dtype=np.float32).copy()
        goal = np.asarray(goal, dtype=np.float32)
        make_features(position, velocity, goal, grid_size)  # 共用形状与有限数检查
        if position.shape != (2,) or velocity.shape != (2,) or goal.shape != (2,):
            raise ValueError("预测只接受单辆车的二维状态")
        if not ((0 <= position).all() and (position < grid_size).all()
                and (0 <= goal).all() and (goal < grid_size).all()):
            raise ValueError("车辆位置和终点必须位于地图内")
        # 小矩阵使用 NumPy 推理，避免每个预测步创建 torch Tensor 的开销。
        # 每次调用读取当前权重，load_state_dict 后不会遗留旧预测缓存。
        weights = [(layer.weight.detach().cpu().numpy(), layer.bias.detach().cpu().numpy())
                   for layer in self.layers if isinstance(layer, nn.Linear)]
        forecast = [position.copy()]
        for offset in range(1, max(steps) + 1):
            if (time + offset) % move_every == 0 and not np.allclose(position, goal, atol=1e-6, rtol=0):
                values = make_features(position, velocity, goal, grid_size)
                for index, (weight, bias) in enumerate(weights):
                    values = values @ weight.T + bias
                    if index < len(weights) - 1:
                        values = np.maximum(values, 0)
                mask = _legal_directions(position, grid_size)
                if not mask.any():  # 极小地图中，分数坐标可能没有合法的整格方向。
                    forecast.append(position.copy())
                    continue
                logits = np.where(mask, values, -np.inf)
                probabilities = np.exp(logits - logits.max())
                probabilities /= probabilities.sum()
                velocity = (probabilities @ DIRECTIONS).astype(np.float32)
                position = np.clip(position + velocity, 0, grid_size - 1).astype(np.float32)
            forecast.append(position.copy())
        return np.asarray([forecast[step] for step in steps], dtype=np.float32)
