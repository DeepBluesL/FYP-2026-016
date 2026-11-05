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
    if config.car_route == "random":
        return _random_route(config, rng)
    size = config.grid_size

    # x 向右、y 向上：起点固定右下角，50x50 时为 (49, 0)。
    position = np.asarray((size - 1, 0), dtype=np.int64)


    # 车辆终点固定在左上角；坐标随地图尺寸变化。
    goal = (0, size - 1)
    goal_array = np.asarray(goal, dtype=np.int64)
    heading = int(rng.integers(0, 4))
    trajectory = np.empty((config.max_steps + 1, 2), dtype=np.int64)
    trajectory[0] = position
    # EnvConfig 已保证各方向权重为正；固定的 log 权重只计算一次。
    log_turn_weights = np.log(np.asarray(
        (config.car_straight_prob, config.car_turn_prob,
         config.car_reverse_prob, config.car_turn_prob), dtype=np.float64
    ))

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

            # 根据方向惯性和终点偏置计算概率
            relative_turns = (direction_indices - heading) % 4
            current_distance = int(np.abs(position - goal_array).sum())
            progress = current_distance - next_distances[direction_indices]
            # 用 log 权重归一化，避免较大 goal_bias 造成 exp 溢出。
            logits = log_turn_weights[relative_turns] + config.car_goal_bias * progress
            weights = np.exp(logits - np.max(logits))
            weights /= weights.sum()
            heading = int(rng.choice(direction_indices, p=weights))
            position = options[heading].copy()
        trajectory[time] = position

    return trajectory, goal


def _random_route(config: EnvConfig, rng: np.random.Generator) -> tuple[np.ndarray, tuple[int, int]]:
    """随机起点和随机目的地；到达目的地后换新目的地继续行驶，不停车。返回首个目的地。"""
    size = config.grid_size
    cells = np.indices((size, size)).reshape(2, -1).T
    uav = np.asarray(config.uav_start)
    distance = np.linalg.norm(cells - uav, axis=1)
    starts = cells[(distance >= 8) & (distance <= 15)]
    position = starts[int(rng.integers(len(starts)))].copy()

    def new_goal(origin):
        options = cells[np.abs(cells - origin).sum(axis=1) >= 20]
        return options[int(rng.integers(len(options)))].copy()

    goal = new_goal(position)
    first_goal = (int(goal[0]), int(goal[1]))
    heading = int(rng.integers(0, 4))
    log_turn_weights = np.log(np.asarray(
        (config.car_straight_prob, config.car_turn_prob,
         config.car_reverse_prob, config.car_turn_prob), dtype=np.float64
    ))
    trajectory = np.empty((config.max_steps + 1, 2), dtype=np.int64)
    trajectory[0] = position
    for time in range(1, config.max_steps + 1):
        if time % config.car_move_every == 0:
            if np.array_equal(position, goal):
                goal = new_goal(position)
            options = position + DIRECTIONS
            direction_indices = np.flatnonzero(np.all((options >= 0) & (options < size), axis=1))
            relative_turns = (direction_indices - heading) % 4
            progress = (np.abs(position - goal).sum()
                        - np.abs(options[direction_indices] - goal).sum(axis=1))
            logits = log_turn_weights[relative_turns] + config.car_goal_bias * progress
            weights = np.exp(logits - np.max(logits))
            heading = int(rng.choice(direction_indices, p=weights / weights.sum()))
            position = options[heading].copy()
        trajectory[time] = position
    return trajectory, first_goal
