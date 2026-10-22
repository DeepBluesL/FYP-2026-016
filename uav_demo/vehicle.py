"""生成独立于 UAV 的、带方向惯性的车辆完整轨迹。"""

from __future__ import annotations

import numpy as np

from .config import EnvConfig

# 顺时针排列：北、东、南、西，便于计算直行 / 左转 / 右转 / 掉头。
DIRECTIONS = np.asarray(((0, 1), (1, 0), (0, -1), (-1, 0)), dtype=np.int64)


def generate_vehicle_trajectory(
        config: EnvConfig, rng: np.random.Generator
) -> tuple[np.ndarray, tuple[int, int]]:
    """返回 ``trajectory[t] = (x, y)``，包含初始时刻与整个回合。

    平时按方向惯性及终点偏置随机游走；剩余时间不足以继续绕路时，
    只选择仍然能按时抵达终点的方向。到达后停留，非移动时刻也停留。
    这样每次 reset 都能得到一条完整、可复现且确实到终点的路线。
    """
    size = config.grid_size
    budget = config.max_steps // config.car_move_every
    if budget < 2 * (size - 1):
        raise ValueError("车辆移动预算必须至少覆盖地图的最大曼哈顿距离。")

    # x 向右、y 向上：起点固定右下角，50x50 时为 (49, 0)。
    position = np.asarray((size - 1, 0), dtype=np.int64)

    # 原随机起点逻辑暂时停用，保留以便恢复。
    # if config.car_spawn_region == "bottom_right":
    #     position = np.asarray((
    #         rng.integers(2 * size // 3, size),
    #         rng.integers(0, (size + 2) // 3),
    #     ), dtype=np.int64)
    # elif rng.random() < 0.5:
    #     low, high = size // 3, min(size, 2 * size // 3 + 1)
    #     position = rng.integers(low, high, size=2, dtype=np.int64)
    # else:
    #     position = rng.integers(0, size, size=2, dtype=np.int64)
    #     position[int(rng.integers(0, 2))] = rng.integers(2 * size // 3, size)

    # 车辆终点固定在左上角；坐标随地图尺寸变化。
    goal = (0, size - 1)
    goal_array = np.asarray(goal, dtype=np.int64)
    heading = int(rng.integers(0, 4))
    trajectory = np.empty((config.max_steps + 1, 2), dtype=np.int64)
    trajectory[0] = position
    turn_probabilities = np.asarray(
        (config.car_straight_prob, config.car_turn_prob,
         config.car_reverse_prob, config.car_turn_prob), dtype=np.float64
    )

    for time in range(1, config.max_steps + 1):
        # 每 car_move_every 个环境步移动一格，到达终点后停留。
        if time % config.car_move_every == 0 and not np.array_equal(position, goal_array):
            options = position + DIRECTIONS
            # 检查地图边界
            in_bounds = np.all((options >= 0) & (options < size), axis=1)
            remaining_moves = (config.max_steps - time) // config.car_move_every + 1
            next_distances = np.abs(options - goal_array).sum(axis=1)
            # 检查剩余移动机会是否满足要求
            feasible = in_bounds & (next_distances <= remaining_moves - 1)
            direction_indices = np.flatnonzero(feasible)
            if not len(direction_indices):
                raise RuntimeError("车辆无法在给定时间内抵达终点，请检查移动预算。")

            # 根据方向惯性和终点偏置计算概率
            relative_turns = (direction_indices - heading) % 4
            base_weights = turn_probabilities[relative_turns]
            current_distance = int(np.abs(position - goal_array).sum())
            progress = current_distance - next_distances[direction_indices]
            # 用 log 权重归一化，避免较大 goal_bias 造成 exp 溢出。
            positive = base_weights > 0
            if positive.any():
                logits = np.full(len(base_weights), -np.inf)
                logits[positive] = (
                        np.log(base_weights[positive]) + config.car_goal_bias * progress[positive]
                )
                weights = np.exp(logits - np.max(logits))
            else:
                # 例如配置为始终直行，却正好朝向边界：选择合法方向。
                weights = np.ones(len(direction_indices), dtype=np.float64)
            weights /= weights.sum()
            heading = int(rng.choice(direction_indices, p=weights))
            position = options[heading].copy()
        trajectory[time] = position

    return trajectory, goal
