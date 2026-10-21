"""无障碍 GridWorld：采集车辆数据后，从左下角飞到右上角。

使用 reset/step 五返回值约定，不依赖 Gymnasium。
Oracle 观测 = 12 个状态量 + 整条车辆轨迹 (t=0..max_steps) 展平。
状态量依次为 UAV x/y、名义速度、UAV 终点 x/y、车辆当前 x/y、
车辆终点 x/y、时间、采集进度、采集完成标志；坐标、时间、进度均归一化。
"""

from __future__ import annotations

import numbers

import numpy as np

from .config import EnvConfig
from .vehicle import generate_vehicle_trajectory

# 第三个量为网格步长（名义速度）：斜向实际距离为步长的 sqrt(2) 倍。
# 0 仅用于 hover，悬停时保留之前的名义速度。
ACTIONS = (
    (0, 0, 0),
    (0, 1, 1), (1, 0, 1), (0, -1, 1), (-1, 0, 1),
    (1, 1, 1), (1, -1, 1), (-1, 1, 1), (-1, -1, 1),
    (0, 1, 2), (1, 0, 2), (0, -1, 2), (-1, 0, 2),
    (1, 1, 2), (1, -1, 2), (-1, 1, 2), (-1, -1, 2)
)
# UAV 八向移动，每步选择网格步长 1 或 2。
ACTION_NAMES = ("hover",
                "north_1", "east_1", "south_1", "west_1",
                "north_east_1", "south_east_1", "north_west_1", "south_west_1",
                "north_2", "east_2", "south_2", "west_2",
                "north_east_2", "south_east_2", "north_west_2", "south_west_2")


def preview_move(
        position: tuple[int, int], speed: int, action: int, grid_size: int
) -> tuple[tuple[int, int], int, bool]:
    """无副作用地计算动作；越界则取消整段位移，但仍设置目标速度。"""
    if not isinstance(action, numbers.Integral) or isinstance(action, (bool, np.bool_)):
        raise ValueError(f"action 必须是 0..{len(ACTIONS) - 1} 的整数。")
    if not 0 <= action < len(ACTIONS):
        raise ValueError(f"action 必须是 0..{len(ACTIONS) - 1} 的整数。")
    dx, dy, target_speed = ACTIONS[action]
    new_speed = speed if action == 0 else target_speed
    candidate = (position[0] + dx * new_speed, position[1] + dy * new_speed)
    invalid = not all(0 <= coordinate < grid_size for coordinate in candidate)
    return (position if invalid else candidate), new_speed, invalid


class GridWorldEnv:
    """每步先同时更新双方位置，再判断采集及任务完成。"""

    action_count = len(ACTIONS)

    def __init__(self, config: EnvConfig | None = None):
        self.config = config if config is not None else EnvConfig()
        self.observation_size = 12 + 2 * (self.config.max_steps + 1)
        self.uav_pos = tuple(self.config.uav_start)
        self.uav_speed = self.config.initial_speed
        self.car_trajectory = np.empty((self.config.max_steps + 1, 2), dtype=np.int64)
        self.car_goal = (0, 0)
        self.time = 0
        self.sensing_progress = 0
        self.sensing_complete = False
        self.travel_distance = 0.0
        self._done = True  # step 前必须先 reset。

    @property
    def car_pos(self) -> tuple[int, int]:
        """当前车辆位置；未来位置只从 car_trajectory 读取。"""
        point = self.car_trajectory[self.time]
        return int(point[0]), int(point[1])

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """重置环境，seed 固定时生成相同车辆路线。"""
        self.car_trajectory, self.car_goal = generate_vehicle_trajectory(
            self.config, np.random.default_rng(seed)
        )
        self.uav_pos = tuple(self.config.uav_start)
        self.uav_speed = self.config.initial_speed
        self.time = 0
        self.sensing_progress = 0
        self.sensing_complete = False
        self.travel_distance = 0.0
        self._done = False
        return self._observation(), self._info(invalid_move=False)

    def valid_action_mask(self) -> np.ndarray:
        """仅屏蔽地图外动作；hover 始终合法。"""
        return np.asarray([
            not preview_move(self.uav_pos, self.uav_speed, action, self.config.grid_size)[2]
            for action in range(self.action_count)
        ], dtype=np.bool_)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._done:
            raise RuntimeError("请先 reset；结束的回合不能继续 step。")
        config = self.config
        old_position = self.uav_pos
        completed_before = self.sensing_complete
        # 在同一个任务阶段计算前后距离，避免完成采集时切换目标产生奖励突跳。
        old_target = config.uav_goal if completed_before else self.car_pos
        old_distance = self._distance(old_position, old_target)
        self.uav_pos, self.uav_speed, invalid = preview_move(
            old_position, self.uav_speed, action, config.grid_size
        )
        self.travel_distance += self._distance(old_position, self.uav_pos)
        self.time += 1
        new_target = config.uav_goal if completed_before else self.car_pos
        new_distance = self._distance(self.uav_pos, new_target)

        # reward计算
        reward = config.step_penalty + config.progress_reward * (old_distance - new_distance)
        if invalid:
            reward += config.invalid_move_penalty

        # 采集设置，默认半径为 2，使用欧氏距离
        contact = self._distance(self.uav_pos, self.car_pos) <= config.sensing_radius
        if not completed_before:
            if contact:
                self.sensing_progress += 1
                reward += config.contact_reward
            elif config.sensing_mode == "consecutive":
                self.sensing_progress = 0
            if self.sensing_progress >= config.sensing_steps:
                self.sensing_progress = config.sensing_steps
                self.sensing_complete = True
                reward += config.sensing_complete_reward

        terminated = bool(self.sensing_complete and self.uav_pos == tuple(config.uav_goal))
        truncated = bool(self.time >= config.max_steps and not terminated)
        if terminated:
            reward += config.success_reward
        elif truncated:
            reward += config.timeout_penalty
        self._done = terminated or truncated
        return self._observation(), float(reward), terminated, truncated, self._info(invalid)

    @staticmethod
    def _distance(first: tuple[int, int], second: tuple[int, int]) -> float:
        """采集、距离塑形与累计路程均使用欧氏距离；斜向路程含 sqrt(2)。"""
        return float(np.hypot(first[0] - second[0], first[1] - second[1]))

    def _observation(self) -> np.ndarray:
        scale = self.config.grid_size - 1
        car_x, car_y = self.car_pos
        state = np.asarray((
            self.uav_pos[0] / scale, self.uav_pos[1] / scale,
            self.uav_speed / 2,
            self.config.uav_goal[0] / scale, self.config.uav_goal[1] / scale,
            car_x / scale, car_y / scale,
            self.car_goal[0] / scale, self.car_goal[1] / scale,
            self.time / self.config.max_steps,
            self.sensing_progress / self.config.sensing_steps,
            float(self.sensing_complete),
        ), dtype=np.float32)
        trajectory = self.car_trajectory.astype(np.float32).reshape(-1) / scale
        return np.concatenate((state, trajectory))

    def _info(self, invalid_move: bool) -> dict:
        contact = self._distance(self.uav_pos, self.car_pos) <= self.config.sensing_radius
        return {
            "time": int(self.time),
            "success": bool(self.sensing_complete and self.uav_pos == tuple(self.config.uav_goal)),
            "sensing_complete": bool(self.sensing_complete),
            "sensing_progress": int(self.sensing_progress),
            "uav_pos": list(self.uav_pos),
            "uav_speed": int(self.uav_speed),
            "car_pos": list(self.car_pos),
            "car_goal": list(self.car_goal),
            "travel_distance": float(self.travel_distance),
            "contact": bool(contact),
            "invalid_move": bool(invalid_move),
        }
