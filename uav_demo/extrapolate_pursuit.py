"""外推贪心：按最近报告的车辆位移线性外推到下一步，再选最接近外推点的动作。

只读取延迟观测（不访问真实车辆位置或未来路线），是延迟条件下比“追旧位置”更公平的基线。
接口与 PredictivePlanner 一致：reset() / choose_action(observation, mask)，可直接交给 evaluate_episode。
"""

from __future__ import annotations

import numpy as np

from .config import EnvConfig
from .environment import ACTIONS, preview_move


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
        distances = self.action_distances(observation)
        for action in range(len(distances)):
            if not mask[action] or preview_move(observation["uav_pos"], observation["uav_speed"],
                                                action, self.config.grid_size)[2]:
                distances[action] = np.inf
        return int(np.argmin(distances))  # 并列时取编号最小者

    def action_distances(self, observation) -> np.ndarray:
        """每个动作执行后到目标（外推车位或 UAV 终点）的欧氏距离；首报前悬停为 0、其余为 1。

        越界动作原地不动，按原位置计算；是否可选由调用方的掩码决定。
        """
        state = observation
        finished = state["sensing_complete"] or state["sensing_failed"]
        count = len(ACTIONS)
        if not finished and not state["has_observation"]:
            distances = np.ones(count)
            distances[0] = 0.0  # 首次报告前悬停，与 simple_pursuit 一致
            return distances
        target = state["uav_goal"] if finished else predicted_car(state, self.config)
        distances = np.empty(count)
        for action in range(count):
            position, _, _ = preview_move(state["uav_pos"], state["uav_speed"], action, self.config.grid_size)
            distances[action] = np.hypot(position[0] - target[0], position[1] - target[1])
        return distances
