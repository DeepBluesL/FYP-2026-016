"""网格概率推演预测器：状态 = 格子 + 朝向，按转向概率逐步推演车位的概率图，点预测取最可能的格子（或 2 格内概率最大的格子）。

转向概率有三种来源（turns）：
- counted：从预测器训练路线里数出直行 / 右转 / 掉头 / 左转的频率；只数四个方向都在地图内的格子上的移动，
  这样边界上被迫转向的移动不会混进来；各类计数加 1 平滑，避免出现零概率；
- base：车辆生成器的基础转向概率，不含目的地偏向；
- goal：基础概率 + 目的地偏向，与车辆生成器的规则完全相同，但要读真实目的地序列，只能作上限诊断。
各来源都和 vehicle.py 一样：越界方向去掉后，在剩余方向上重新归一化。
只建模随机路线；固定路线的按时到站约束和到站停车都没有建模。
"""

from __future__ import annotations

import numpy as np

from .prediction import observed_motion
from .vehicle import DIRECTIONS, generate_vehicle_trajectory

TURN_SOURCES = ("counted", "base", "goal")
# 相对转向的顺序与 vehicle.py 一致：(方向序号 − 朝向) % 4 = 0 直行、1 右转、2 掉头、3 左转。
TURN_NAMES = ("straight", "right", "reverse", "left")


def last_heading(history, move_every: int = 1):
    """报告时刻最后一次观测到的行驶方向在 DIRECTIONS 中的序号；还没看到过移动时返回 None。"""
    motion = observed_motion(history, move_every)
    if not motion.any():
        return None
    return int(np.argmax(DIRECTIONS @ motion))


def _shift(offset: int, size: int) -> slice:
    """沿一个坐标轴平移 offset 格后落在地图内的那段下标。"""
    return slice(max(offset, 0), size + min(offset, 0))


def expected_positions(maps) -> np.ndarray:
    """概率图 [..., x, y] 的均值位置 [..., 2]。"""
    maps = np.asarray(maps)
    x, y = np.indices(maps.shape[-2:])
    return np.stack(((maps * x).sum(axis=(-2, -1)), (maps * y).sum(axis=(-2, -1))), axis=-1)


def most_likely_positions(maps) -> np.ndarray:
    """概率图 [n, x, y] 中每张图概率最大的格子 [n, 2]；并列时取下标最小的格子。"""
    maps = np.asarray(maps)
    flat = maps.reshape(len(maps), -1).argmax(axis=1)
    return np.stack(np.unravel_index(flat, maps.shape[-2:]), axis=-1).astype(np.float32)


POINT_RULES = ("mode", "disk")


def point_positions(maps, point: str = "mode", radius: float = 2.0) -> np.ndarray:
    """概率图 [n, x, y] → 点预测 [n, 2]。mode：概率最大的格子；disk：车落在该格半径 radius 内（欧氏距离）概率最大的格子。

    采集看的是车是否在半径内，disk 让这个事件的概率最大；mode 只让「恰好在这一格」的概率最大。
    disk 并列时取本格概率最大的（例如概率全在一格时取这一格本身），仍并列再取下标最小的格子；mode 并列取下标最小的格子。
    """
    if point == "mode":
        return most_likely_positions(maps)
    if point != "disk":
        raise ValueError(f"point 只能取 {POINT_RULES}")
    scores = []
    for probability in maps:
        probability = np.asarray(probability, dtype=np.float64)
        total = disk_sum(probability, radius)
        tied = np.isclose(total, total.max(), rtol=1e-12, atol=0.0)
        scores.append(np.where(tied, probability, -1.0))
    return most_likely_positions(scores)


def disk_sum(probability, radius: float) -> np.ndarray:
    """S[x, y] = 以 (x, y) 为圆心、欧氏半径 radius 内（与环境的采集判定相同）各格概率之和。"""
    size = probability.shape[0]
    reach = int(np.floor(radius))
    total = np.zeros_like(probability)
    for dx in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            # total[x, y] += probability[x + dx, y + dy]，越界部分不加。
            target = (slice(max(-dx, 0), size - max(dx, 0)), slice(max(-dy, 0), size - max(dy, 0)))
            source = (slice(max(dx, 0), size + min(dx, 0)), slice(max(dy, 0), size + min(dy, 0)))
            total[target] += probability[source]
    return total


def _goal_phases(goals, report_time: int) -> list:
    """从报告时刻起车辆将依次前往的真实目的地；最后补 None，表示走完已知目的地后不再知道下一个。"""
    phases = []
    for goal in np.asarray(goals)[report_time:]:
        goal = (int(goal[0]), int(goal[1]))
        if not phases or phases[-1] != goal:
            phases.append(goal)
    return phases + [None]


class GridPredictor:
    model_type = "grid"

    def __init__(self, turns: str, turn_probabilities, goal_bias: float = 0.0, point: str = "mode", radius: float = 2.0):
        if turns not in TURN_SOURCES:
            raise ValueError(f"turns 只能取 {TURN_SOURCES}")
        probabilities = np.asarray(turn_probabilities, dtype=np.float64)
        if probabilities.shape != (4,) or not np.all(probabilities > 0) or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("turn_probabilities 必须是 4 个和为 1 的正数（直行、右转、掉头、左转）")
        self.turns = turns
        self.turn_probabilities = probabilities
        self.goal_bias = float(goal_bias) if turns == "goal" else 0.0
        if point not in POINT_RULES:
            raise ValueError(f"point 只能取 {POINT_RULES}")
        self.point, self.radius = point, float(radius)  # 点预测的取法，见 point_positions
        self._transitions = {}

    @property
    def needs_truth(self) -> bool:
        """goal 来源要读真实目的地序列，环境须以 expose_truth=True 创建。"""
        return self.turns == "goal"

    @staticmethod
    def _check_route(config) -> None:
        if config.car_route != "random":
            raise ValueError("网格推演只建模随机路线（car_route=random）")

    @classmethod
    def base(cls, config) -> "GridPredictor":
        cls._check_route(config)
        return cls("base", _base_probabilities(config))

    @classmethod
    def with_goals(cls, config) -> "GridPredictor":
        cls._check_route(config)
        return cls("goal", _base_probabilities(config), config.car_goal_bias)

    @classmethod
    def fit(cls, config, groups) -> tuple["GridPredictor", dict]:
        """在 groups["train"] 的路线上数相对转向；用 groups["validation"] 比较数出来的概率和基础概率的拟合程度。"""
        cls._check_route(config)
        counts = np.ones(4)  # 加 1 平滑
        for seed in groups["train"]:
            route, _ = generate_vehicle_trajectory(config, np.random.default_rng(seed))
            for position, previous, current in _turns(route):
                if np.all((position + DIRECTIONS >= 0) & (position + DIRECTIONS < config.grid_size)):
                    counts[(current - previous) % 4] += 1
        model = cls("counted", counts / counts.sum())
        base = cls.base(config)
        metrics = {
            "model_type": "grid", "turns": "counted",
            "turn_probabilities": dict(zip(TURN_NAMES, model.turn_probabilities.tolist())),
            "counted_moves": int(counts.sum() - 4),
            "validation_mse": model.one_step_mse(config, groups["validation"]),
            "validation_log_likelihood": model.log_likelihood(config, groups["validation"]),
            "base_validation_log_likelihood": base.log_likelihood(config, groups["validation"]),
            "seed_groups": groups,
        }
        return model, metrics

    def transition(self, grid_size: int, goal=None) -> np.ndarray:
        """[朝向 h, 方向 j, x, y]：朝向 h、位于 (x, y) 的车下一步走方向 j 的概率；goal 为 None 时不含目的地偏向。"""
        key = (grid_size, goal if self.turns == "goal" else None)
        if key not in self._transitions:
            if len(self._transitions) >= 64:
                self._transitions.clear()
            cells = np.indices((grid_size, grid_size)).transpose(1, 2, 0)
            targets = cells[None] + DIRECTIONS[:, None, None]  # [方向, x, y, 2]
            legal = np.all((targets >= 0) & (targets < grid_size), axis=-1)
            relative = (np.arange(4)[None, :] - np.arange(4)[:, None]) % 4  # [h, j]
            weights = self.turn_probabilities[relative][:, :, None, None] * legal[None]
            if key[1] is not None:
                goal = np.asarray(key[1])
                progress = np.abs(cells - goal).sum(axis=-1)[None] - np.abs(targets - goal).sum(axis=-1)
                weights = weights * np.exp(self.goal_bias * progress)[None]
            self._transitions[key] = weights / weights.sum(axis=1, keepdims=True)
        return self._transitions[key]

    def distributions(self, history, grid_size: int, steps, move_every: int = 1, goals=None) -> np.ndarray:
        """steps 相对最后报告时刻；返回各步的车位概率图 [len(steps), x, y]。

        朝向取报告时最后一次观测到的行驶方向；首报时未知，四个朝向等概率（同 vehicle.py 抽初始朝向）。
        goal 来源需要 goals（环境的 true_car_goals）：车到达当前目的地后，下一次移动前换成序列里的下一个目的地；
        已知目的地用完后改按基础概率走。
        """
        steps = tuple(int(s) for s in steps)
        if self.needs_truth and goals is None:
            raise ValueError("goal 来源需要真实目的地序列，环境须以 expose_truth=True 创建")
        report_time, position = history[-1]
        heading = last_heading(history, move_every)
        phases = _goal_phases(goals, report_time) if self.needs_truth else [None]
        probability = np.zeros((len(phases), 4, grid_size, grid_size))
        if heading is None:
            probability[0, :, position[0], position[1]] = 0.25
        else:
            probability[0, heading, position[0], position[1]] = 1.0
        maps = {0: probability.sum(axis=(0, 1))}
        for offset in range(1, max(steps) + 1):
            if (report_time + offset) % move_every == 0:
                probability = self._move(probability, phases, grid_size)
            maps[offset] = probability.sum(axis=(0, 1))
        return np.stack([maps[s] for s in steps])

    def _move(self, probability, phases, grid_size: int):
        """所有概率质量移动一格；goal 来源下，正停在当前目的地上的质量先转到下一个目的地。"""
        probability = probability.copy()
        for phase, goal in enumerate(phases[:-1]):
            arrived = probability[phase, :, goal[0], goal[1]].copy()
            probability[phase, :, goal[0], goal[1]] = 0.0
            probability[phase + 1, :, goal[0], goal[1]] += arrived
        moved = np.zeros_like(probability)
        for phase, goal in enumerate(phases):
            if not probability[phase].any():
                continue
            # mass[j, x, y]：从 (x, y) 出发走方向 j 的概率质量，再整体平移到目标格。
            mass = np.einsum("hxy,hjxy->jxy", probability[phase], self.transition(grid_size, goal))
            for j, (dx, dy) in enumerate(DIRECTIONS):
                moved[phase, j, _shift(dx, grid_size), _shift(dy, grid_size)] = \
                    mass[j, _shift(-dx, grid_size), _shift(-dy, grid_size)]
        return moved

    def predict_positions(self, history, grid_size: int, steps, move_every: int = 1, goals=None) -> np.ndarray:
        """各步的点预测（取法见 point_positions），steps 相对最后报告时刻，与其他预测器同一接口。"""
        return point_positions(self.distributions(history, grid_size, steps, move_every, goals), self.point, self.radius)

    def log_likelihood(self, config, seeds) -> float:
        """路线上每次实际移动在本模型下的平均对数概率；首次移动的朝向未知，按四个朝向平均。"""
        total, count = 0.0, 0
        for seed in seeds:
            route, _, goals = generate_vehicle_trajectory(config, np.random.default_rng(seed), with_goals=True)
            for time, (position, previous, current) in _moves(route):
                goal = tuple(int(v) for v in goals[time]) if self.needs_truth else None
                transition = self.transition(config.grid_size, goal)[:, current, position[0], position[1]]
                total += np.log(transition.mean() if previous is None else transition[previous])
                count += 1
        return float(total / count)

    def one_step_mse(self, config, seeds) -> float:
        """一步预测（最可能的格子）的均方误差，坐标按 grid_size − 1 归一化，与 VAR、LSTM 的 validation_mse 同口径。"""
        squared, count = 0.0, 0
        scale = config.grid_size - 1
        for seed in seeds:
            route, _ = generate_vehicle_trajectory(config, np.random.default_rng(seed))
            history = [(0, tuple(int(v) for v in route[0]))]
            for time in range(1, len(route)):
                predicted = self.predict_positions(history, config.grid_size, (1,), config.car_move_every)[0]
                squared += float(np.square((predicted - route[time]) / scale).sum())
                count += 2
                history.append((time, tuple(int(v) for v in route[time])))
        return squared / count

    def export_state(self) -> dict:
        return {"model_type": self.model_type, "turns": self.turns,
                "turn_probabilities": self.turn_probabilities.tolist(), "goal_bias": self.goal_bias,
                "point": self.point, "radius": self.radius}

    @classmethod
    def from_state(cls, saved: dict) -> "GridPredictor":
        return cls(saved["turns"], saved["turn_probabilities"], saved.get("goal_bias", 0.0),
                   saved.get("point", "mode"), saved.get("radius", 2.0))


def _base_probabilities(config):
    return (config.car_straight_prob, config.car_turn_prob, config.car_reverse_prob, config.car_turn_prob)


def _moves(route):
    """路线上每次实际移动：(移动发生的时刻, (出发格, 上一次移动方向或 None, 本次方向))。"""
    displacements = np.diff(route, axis=0)
    previous = None
    for time, displacement in enumerate(displacements, start=1):
        if not displacement.any():
            continue
        current = int(np.flatnonzero((DIRECTIONS == displacement).all(axis=1))[0])
        yield time, (route[time - 1], previous, current)
        previous = current


def _turns(route):
    """路线上朝向已知的移动：(出发格, 上一次方向, 本次方向)。"""
    for _, (position, previous, current) in _moves(route):
        if previous is not None:
            yield position, previous, current
