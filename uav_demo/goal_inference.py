"""目的地推断预测器：对车辆当前目的地做贝叶斯滤波，再按推断出的目的地往前抽样推演车位。

思路同「逆规划」（Baker, Saxe & Tenenbaum 2009）和目标驱动的轨迹预测（Ziebart et al. 2009）：假定车辆按「方向惯性 + 朝目的地偏向」行驶，
从观测到的每一步移动反推目的地的后验，再用后验预测未来。模型的形式与 vehicle.py 的随机路线相同：
- 每步在地图内的方向上按 log 转向权重（直行 / 右转 / 掉头 / 左转）+ 偏向 × 进度（靠近目的地 +1，远离 −1）取 softmax；
- 目的地在离出发点（或上一个目的地）曼哈顿距离不小于 min_goal_distance 的格子里均匀选，到达后换新目的地。
参数（转向权重、偏向、最小距离）只从预测器训练路线估计；训练时用路线上的真实目的地作标签，预测时只读报告历史。
只建模随机路线、车辆每步移动一格（car_route=random，car_move_every=1）。
"""

from __future__ import annotations

import numpy as np
import torch

from .point_selection import POINT_RULES, point_positions
from .vehicle import DIRECTIONS, generate_vehicle_trajectory


def _direction_index(move) -> int:
    return int(np.flatnonzero((DIRECTIONS == np.asarray(move)).all(axis=1))[0])


class GoalFilterPredictor:
    model_type = "goal_filter"

    def __init__(self, turn_log_weights, goal_bias: float, min_goal_distance: int, samples: int = 2000,
                 sample_seed: int = 0, point: str = "mode", radius: float = 2.0):
        if point not in POINT_RULES:
            raise ValueError(f"point 只能取 {POINT_RULES}")
        self.turn_log_weights = np.asarray(turn_log_weights, dtype=np.float64)  # 直行、右转、掉头、左转
        self.goal_bias = float(goal_bias)
        self.min_goal_distance = int(min_goal_distance)
        self.samples, self.sample_seed = samples, sample_seed
        self.point, self.radius = point, float(radius)
        self._last = None  # 最近一次的（报告历史, 目的地后验, 朝向），历史每次只多一条时增量更新

    @staticmethod
    def _check_route(config) -> None:
        if config.car_route != "random" or config.car_move_every != 1:
            raise ValueError("目的地推断只建模随机路线、每步移动一格（car_route=random，car_move_every=1）")

    # 模型
    def _new_goal_probability(self, cells, origin) -> np.ndarray:
        """从 origin 出发选新目的地的分布：曼哈顿距离不小于 min_goal_distance 的格子上均匀。"""
        weights = (np.abs(cells - origin).sum(axis=1) >= self.min_goal_distance).astype(np.float64)
        return weights / weights.sum()

    def _draw_new_goals(self, origins, cells, rng) -> np.ndarray:
        """为每个出发点各抽一个新目的地（与 _new_goal_probability 同分布）：在全图均匀抽格，离出发点太近的重抽。"""
        goals, pending = np.empty_like(origins), np.arange(len(origins))
        while pending.size:
            candidates = cells[rng.integers(len(cells), size=pending.size)]
            far = np.abs(candidates - origins[pending]).sum(axis=1) >= self.min_goal_distance
            goals[pending[far]] = candidates[far]
            pending = pending[~far]
        return goals

    def _move_probabilities(self, position, heading, goals, grid_size: int) -> np.ndarray:
        """各目的地下这一步走四个方向的概率 [G, 4]；heading 为 None（首次移动）时对四个初始朝向平均。"""
        legal = np.all((position + DIRECTIONS >= 0) & (position + DIRECTIONS < grid_size), axis=1)
        progress = np.where(((goals - position)[:, None, :] * DIRECTIONS).sum(axis=-1) > 0, 1.0, -1.0)
        total = 0.0
        for start in (range(4) if heading is None else (heading,)):
            logits = self.turn_log_weights[(np.arange(4) - start) % 4] + self.goal_bias * progress
            logits = np.where(legal, logits, -np.inf)
            probability = np.exp(logits - logits.max(axis=1, keepdims=True))
            total = total + probability / probability.sum(axis=1, keepdims=True)
        return total / (4 if heading is None else 1)

    def _update(self, belief, heading, previous, current, cells, grid_size: int):
        """一步滤波：车在 previous 时若已到目的地，先按换目的地的规则改写后验；再乘这一步移动的似然。

        返回（新后验, 新朝向, 这一步移动在旧后验下的预测概率）。
        """
        previous, current = np.asarray(previous), np.asarray(current)
        here = previous[0] * grid_size + previous[1]
        arrived = belief[here]
        belief = belief.copy()
        belief[here] = 0.0
        belief += arrived * self._new_goal_probability(cells, previous)
        direction = _direction_index(current - previous)
        likelihood = self._move_probabilities(previous, heading, cells, grid_size)[:, direction]
        predictive = float(belief @ likelihood)
        belief = belief * likelihood / predictive
        return belief, direction, predictive

    def belief(self, history, grid_size: int):
        """报告历史（须从 0 时刻起、相邻报告间车辆恰好移动一格）→（各格是当前目的地的后验 [grid_size²], 最后朝向或 None）。"""
        cells = np.indices((grid_size, grid_size)).reshape(2, -1).T
        key = (grid_size, tuple((int(t), (int(p[0]), int(p[1]))) for t, p in history))
        if self._last is not None and self._last[0] == key:
            return self._last[1], self._last[2]
        if self._last is not None and self._last[0][0] == grid_size and self._last[0][1] == key[1][:-1]:
            belief, heading = self._last[1], self._last[2]
            belief, heading, _ = self._update(belief, heading, key[1][-2][1], key[1][-1][1], cells, grid_size)
        else:
            belief, heading = self._new_goal_probability(cells, np.asarray(key[1][0][1])), None
            for (_, previous), (_, current) in zip(key[1], key[1][1:]):
                belief, heading, _ = self._update(belief, heading, previous, current, cells, grid_size)
        self._last = (key, belief, heading)
        return belief, heading

    def sample_positions(self, history, grid_size: int, steps) -> np.ndarray:
        """按目的地后验抽 samples 个目的地，按模型往前抽样推演，返回 steps 各步的位置 [len(steps), samples, 2]。

        每次用固定的 sample_seed 抽样，同样的历史总是给出同样的结果。
        """
        belief, heading = self.belief(history, grid_size)
        cells = np.indices((grid_size, grid_size)).reshape(2, -1).T
        rng = np.random.default_rng(self.sample_seed)
        goals = cells[rng.choice(len(cells), self.samples, p=belief)]
        position = np.repeat(np.asarray(history[-1][1], dtype=np.int64)[None], self.samples, axis=0)
        headings = np.full(self.samples, -1 if heading is None else heading)
        recorded = {0: position}
        for k in range(1, max(steps) + 1):
            arrived = np.flatnonzero((position == goals).all(axis=1))
            goals[arrived] = self._draw_new_goals(position[arrived], cells, rng)
            # DIRECTIONS 顺序为北、东、南、西
            x, y = position[:, 0], position[:, 1]
            legal = np.stack((y < grid_size - 1, x < grid_size - 1, y > 0, x > 0), axis=1)
            progress = np.where((goals - position) @ DIRECTIONS.T > 0, 1.0, -1.0)
            start = np.where(headings < 0, rng.integers(0, 4, self.samples), headings)
            logits = self.turn_log_weights[(np.arange(4) - start[:, None]) % 4] + self.goal_bias * progress
            # Gumbel-max：logits 加 Gumbel 噪声后取最大，等价于按 softmax 概率抽方向。
            headings = (np.where(legal, logits, -np.inf) + rng.gumbel(size=logits.shape)).argmax(axis=1)
            position = position + DIRECTIONS[headings]
            recorded[k] = position
        return np.stack([recorded[k] for k in steps])

    def predict_positions(self, history, grid_size: int, steps, move_every: int = 1) -> np.ndarray:
        """各步的点预测 [len(steps), 2]（抽样直方图上按 point 取格），steps 相对最后报告时刻。"""
        if move_every != 1:
            raise ValueError("目的地推断只支持车辆每步移动一格")
        counts = []
        for positions in self.sample_positions(history, grid_size, tuple(steps)):
            histogram = np.zeros((grid_size, grid_size))
            np.add.at(histogram, (positions[:, 0], positions[:, 1]), 1.0)
            counts.append(histogram)
        return point_positions(counts, self.point, self.radius)

    def log_likelihood(self, config, seeds) -> float:
        """路线上每次实际移动的平均预测对数概率：只用此前的报告推断目的地（与方向 LSTM 的验证指标同口径）。"""
        cells = np.indices((config.grid_size, config.grid_size)).reshape(2, -1).T
        total, count = 0.0, 0
        for seed in seeds:
            route, _ = generate_vehicle_trajectory(config, np.random.default_rng(seed))
            belief, heading = self._new_goal_probability(cells, route[0]), None
            for previous, current in zip(route[:-1], route[1:]):
                belief, heading, predictive = self._update(belief, heading, previous, current, cells, config.grid_size)
                total += np.log(predictive)
                count += 1
        return total / count

    @classmethod
    def fit(cls, config, groups, *, samples: int = 2000) -> tuple["GoalFilterPredictor", dict]:
        """在训练路线上估计参数：以真实目的地为标签，对转向权重和偏向做最大似然（首次移动朝向未知，不计入）；
        最小目的地距离取训练路线里观察到的最小值。验证路线只用来报告指标。"""
        cls._check_route(config)
        relative, progress, legal, chosen, distances = [], [], [], [], []
        for seed in groups["train"]:
            route, _, goals = generate_vehicle_trajectory(config, np.random.default_rng(seed), with_goals=True)
            distances.append(int(np.abs(goals[0] - route[0]).sum()))
            for time in range(1, len(route)):
                if not np.array_equal(goals[time], goals[time - 1]):
                    distances.append(int(np.abs(goals[time] - route[time - 1]).sum()))
                if time >= 2:
                    # goals[time] 是从 time − 1 走到 time 这一步所用的目的地（到达后先换目的地再选方向）。
                    position, goal = route[time - 1], goals[time]
                    heading = _direction_index(route[time - 1] - route[time - 2])
                    relative.append((np.arange(4) - heading) % 4)
                    progress.append(np.where(((goal - position) * DIRECTIONS).sum(axis=1) > 0, 1.0, -1.0))
                    legal.append(np.all((position + DIRECTIONS >= 0) & (position + DIRECTIONS < config.grid_size), axis=1))
                    chosen.append(_direction_index(route[time] - route[time - 1]))
        relative, progress = torch.as_tensor(np.stack(relative)), torch.as_tensor(np.stack(progress))
        legal, chosen = torch.as_tensor(np.stack(legal)), torch.as_tensor(np.asarray(chosen))
        weights = torch.zeros(3, dtype=torch.float64, requires_grad=True)  # 右转、掉头、左转相对直行
        bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([weights, bias], max_iter=200, line_search_fn="strong_wolfe")

        def loss():
            optimizer.zero_grad()
            log_weights = torch.cat((weights.new_zeros(1), weights))
            logits = (log_weights[relative] + bias * progress).masked_fill(~legal, -torch.inf)
            value = torch.nn.functional.cross_entropy(logits, chosen)
            value.backward()
            return value

        optimizer.step(loss)
        log_weights = np.concatenate(([0.0], weights.detach().numpy()))
        log_weights -= np.log(np.exp(log_weights).sum())  # 归一化成基础概率的 log，便于和生成器参数对照
        model = cls(log_weights, float(bias.detach()), min(distances), samples)
        metrics = {
            "model_type": cls.model_type, "turn_probabilities": np.exp(log_weights).tolist(),
            "goal_bias": model.goal_bias, "min_goal_distance": model.min_goal_distance,
            "train_moves": int(len(chosen)), "validation_log_likelihood": model.log_likelihood(config, groups["validation"]),
            "seed_groups": groups, "samples": samples,
        }
        return model, metrics

    def export_state(self) -> dict:
        return {"model_type": self.model_type, "turn_log_weights": self.turn_log_weights.tolist(),
                "goal_bias": self.goal_bias, "min_goal_distance": self.min_goal_distance, "samples": self.samples,
                "sample_seed": self.sample_seed, "point": self.point, "radius": self.radius}

    @classmethod
    def from_state(cls, saved: dict) -> "GoalFilterPredictor":
        return cls(saved["turn_log_weights"], saved["goal_bias"], saved["min_goal_distance"], saved["samples"],
                   saved["sample_seed"], saved.get("point", "mode"), saved.get("radius", 2.0))
