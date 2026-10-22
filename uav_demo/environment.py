"""无障碍 GridWorld：采集车辆数据后，从左下角飞到右上角。

使用 reset/step 五返回值约定，不依赖 Gymnasium。
当前状态观测共 20 维，不暴露车辆未来轨迹。
状态量依次为 UAV x/y、名义速度、UAV 终点 x/y、车辆当前 x/y、
车辆终点 x/y、时间、采集进度、采集完成标志、采集失败标志，
以及相对车辆/终点坐标、最近观测的车辆位移和移动时钟相位。
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
        self.observation_size = 20
        self.uav_pos = tuple(self.config.uav_start)
        self.uav_speed = self.config.initial_speed
        self.car_trajectory = np.empty((self.config.max_steps + 1, 2), dtype=np.int64)
        self.car_goal = (0, 0)
        self.time = 0
        self.sensing_progress = 0
        self.sensing_complete = False
        self.sensing_failed = False
        self.car_velocity = (0, 0)  # 只能从已发生的位移观测方向，初始朝向未知。
        self.travel_distance = 0.0
        self._done = True  # step 前必须先 reset。

    @property
    def sensing_finished(self) -> bool:
        """成功或车辆到站导致失败，都结束采集阶段，后续目标变为 UAV 终点。"""
        return self.sensing_complete or self.sensing_failed

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
        self.sensing_failed = False
        self.car_velocity = (0, 0)
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
        finished_before = self.sensing_finished
        # 在同一个任务阶段计算前后距离，避免完成采集时切换目标产生奖励突跳。
        old_target = config.uav_goal if finished_before else self.car_pos
        old_distance = self._distance(old_position, old_target)
        old_car = self.car_pos
        self.uav_pos, self.uav_speed, invalid = preview_move(
            old_position, self.uav_speed, action, config.grid_size
        )
        moved_distance = self._distance(old_position, self.uav_pos)
        self.travel_distance += moved_distance
        self.time += 1
        displacement = tuple(b - a for a, b in zip(old_car, self.car_pos))
        if any(displacement):
            self.car_velocity = displacement
        new_target = config.uav_goal if finished_before else self.car_pos
        new_distance = self._distance(self.uav_pos, new_target)
        if not finished_before and config.progress_to_contact_boundary:
            old_distance = max(0.0, old_distance - config.sensing_radius)
            new_distance = max(0.0, new_distance - config.sensing_radius)

        # reward计算
        progress_scale = config.failed_return_progress_scale if self.sensing_failed else 1.0
        reward = (config.step_penalty + config.movement_penalty * moved_distance
                  + config.progress_reward * progress_scale * (old_distance - new_distance))
        if invalid:
            reward += config.invalid_move_penalty

        # 采集设置，默认半径为 2，使用欧氏距离
        contact = self._distance(self.uav_pos, self.car_pos) <= config.sensing_radius
        if not finished_before:
            if contact:
                self.sensing_progress += 1
                reward += config.contact_reward
            elif config.sensing_mode == "consecutive":
                self.sensing_progress = 0
            if self.sensing_progress >= config.sensing_steps:
                self.sensing_progress = config.sensing_steps
                self.sensing_complete = True
                reward += config.sensing_complete_reward
            # 到站这一步仍有最后一次采集机会；刚好完成则不处罚。
            if not self.sensing_complete and self.car_pos == self.car_goal:
                self.sensing_failed = True
                reward += config.sensing_failure_penalty

        terminated = bool(self.sensing_finished and self.uav_pos == tuple(config.uav_goal))
        truncated = bool(self.time >= config.max_steps and not terminated)
        if terminated and self.sensing_complete:
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
        return np.asarray((
            self.uav_pos[0] / scale, self.uav_pos[1] / scale,
            self.uav_speed / 2,
            self.config.uav_goal[0] / scale, self.config.uav_goal[1] / scale,
            car_x / scale, car_y / scale,
            self.car_goal[0] / scale, self.car_goal[1] / scale,
            self.time / self.config.max_steps,
            self.sensing_progress / self.config.sensing_steps,
            float(self.sensing_complete),
            float(self.sensing_failed),
            (car_x - self.uav_pos[0]) / scale,
            (car_y - self.uav_pos[1]) / scale,
            (self.config.uav_goal[0] - self.uav_pos[0]) / scale,
            (self.config.uav_goal[1] - self.uav_pos[1]) / scale,
            float(self.car_velocity[0]), float(self.car_velocity[1]),
            (self.time % self.config.car_move_every) / self.config.car_move_every,
        ), dtype=np.float32)

    def _info(self, invalid_move: bool) -> dict:
        contact = self._distance(self.uav_pos, self.car_pos) <= self.config.sensing_radius
        return {
            "time": int(self.time),
            "success": bool(self.sensing_complete and self.uav_pos == tuple(self.config.uav_goal)),
            "sensing_complete": bool(self.sensing_complete),
            "sensing_failed": bool(self.sensing_failed),
            "sensing_finished": bool(self.sensing_finished),
            "uav_reached_goal": self.uav_pos == tuple(self.config.uav_goal),
            "car_arrived": self.car_pos == self.car_goal,
            "phase": "to_goal" if self.sensing_finished else "sensing",
            "sensing_progress": int(self.sensing_progress),
            "uav_pos": list(self.uav_pos),
            "uav_speed": int(self.uav_speed),
            "car_pos": list(self.car_pos),
            "car_goal": list(self.car_goal),
            "car_velocity": list(self.car_velocity),
            "travel_distance": float(self.travel_distance),
            "contact": bool(contact),
            "invalid_move": bool(invalid_move),
        }
