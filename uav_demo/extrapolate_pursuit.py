"""外推贪心：按最近报告的车辆位移线性外推到下一步，再选最接近外推点的动作。

只读取延迟观测（不访问真实车辆位置或未来路线），是延迟条件下比“追旧位置”更公平的基线。
接口与 PredictivePlanner 一致：reset() / choose_action(observation, mask)，可直接交给 evaluate_episode。
"""

from __future__ import annotations

import numpy as np

from .config import EnvConfig
from .environment import preview_move


def predicted_car(observation, config: EnvConfig) -> tuple[int, int]:
    """从报告时刻外推到下一步：车辆每 car_move_every 步移动一格，按最近位移方向计数。"""
    age = observation["observation_age"]
    observed_time = observation["time"] - age
    moves = sum((observed_time + k) % config.car_move_every == 0 for k in range(1, age + 2))
    x, y = (p + moves * v for p, v in zip(observation["car_pos"], observation["car_velocity"]))
    last = config.grid_size - 1
    return int(np.clip(x, 0, last)), int(np.clip(y, 0, last))


class ExtrapolatePursuit:
    """感知阶段追外推车位，首报前悬停；感知结束（成功或失败）后去 UAV 终点。"""

    def __init__(self, config: EnvConfig):
        self.config = config

    def reset(self) -> None:
        """无内部状态；保留接口以便与其他控制器统一评估。"""

    def choose_action(self, observation, mask) -> int:
        state = observation
        finished = state["sensing_complete"] or state["sensing_failed"]
        if not finished and not state["has_observation"]:
            return 0  # 首次报告前悬停，与 simple_pursuit 一致
        target = state["uav_goal"] if finished else predicted_car(state, self.config)

        best_action, best_distance = 0, float("inf")
        for action in np.flatnonzero(mask):
            position, _, invalid = preview_move(
                state["uav_pos"], state["uav_speed"], int(action), self.config.grid_size
            )
            if invalid:
                continue
            distance = sum((a - b) ** 2 for a, b in zip(position, target))
            if distance < best_distance:
                best_distance, best_action = distance, int(action)
        return best_action
