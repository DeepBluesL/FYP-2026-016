"""学习运动模型驱动的预测拦截控制器；不是 DQN，也不访问真实未来。

先用车辆预测概率均值选择可提前抵达的整数会合点，保持该点等待；
接触后用最小必要位移维持感知，采集成功或失败后再导航至 UAV 终点。
"""

from __future__ import annotations

import numpy as np

from .config import EnvConfig
from .environment import ACTIONS, preview_move
from .planner import minimum_travel_steps


class PredictivePlanner:
    """以学得的车辆预测为模型的滚动规划，明确区分于端到端 RL 策略。

    ``target`` 为当前承诺的会合点，``target_time`` 为预计首次接触时刻。
    保持目标是为了真正提前到达并等待，避免每步重算变成追逐移动预测点。
    """

    def __init__(self, motion_predictor, config: EnvConfig, horizon: int = 30,
                 wait_margin: int = 3, expiry_slack: int = 3):
        if motion_predictor is None:
            raise ValueError("预测拦截需要已训练的车辆预测器")
        if horizon < 1 or wait_margin < 0 or expiry_slack < 0:
            raise ValueError("horizon 必须为正，等待余量和过期余量必须非负")
        self.motion_predictor = motion_predictor
        self.config = config
        self.horizon = horizon
        self.wait_margin = wait_margin
        self.expiry_slack = expiry_slack
        self.reset()

    def reset(self) -> None:
        self.target: tuple[int, int] | None = None
        self.target_time: int | None = None
        self._last_time: int | None = None

    def _rendezvous(self, uav, forecast, time):
        """在预测中寻找连续 K 步覆盖点，要求到达早于首次接触至少 3 步。"""
        radius = self.config.sensing_radius
        required = min(self.config.sensing_steps, len(forecast))
        windows = np.asarray([forecast[start:start + required].mean(axis=0)
                              for start in range(len(forecast) - required + 1)])
        offsets = np.asarray([(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1)])
        points = np.rint(windows[:, None, :]).astype(int) + offsets[None, :, :]
        points = np.unique(np.clip(points.reshape(-1, 2), 0, self.config.grid_size - 1), axis=0)
        distances = np.linalg.norm(points[:, None, :] - forecast[None, :, :], axis=2)
        contacts = distances <= radius
        best = None
        for index, point in enumerate(points):
            contact_times = np.flatnonzero(contacts[index])
            if not len(contact_times):
                continue
            first_contact = int(contact_times[0]) + 1
            travel = minimum_travel_steps(uav, point)
            if travel + self.wait_margin > first_contact:
                continue
            counts = np.convolve(contacts[index].astype(int), np.ones(required, dtype=int), mode="valid")
            feasible = np.flatnonzero(counts == required)
            if not len(feasible):
                continue
            start = int(feasible[0])
            score = (start + required, travel, float(distances[index, start:start + required].max()),
                     int(point[0]), int(point[1]))
            if best is None or score < best[0]:
                best = (score, tuple(int(value) for value in point), time + first_contact)
        if best is not None:
            return best[1], best[2]
        # 无法留出提前等待余量时，选择最早可赶到的预测位置，仍尝试感知。
        for offset, position in enumerate(forecast[:self.horizon], start=1):
            point = tuple(np.clip(np.rint(position), 0, self.config.grid_size - 1).astype(int))
            if minimum_travel_steps(uav, point) <= offset:
                return point, time + offset
        point = tuple(np.clip(np.rint(forecast[-1]), 0, self.config.grid_size - 1).astype(int))
        return point, time + len(forecast)

    @staticmethod
    def _toward(candidates, target):
        return min(candidates, key=lambda item: (
            minimum_travel_steps(item[1], target),
            sum(abs(a - b) for a, b in zip(item[1], target)),
            item[0],
        ))[0]

    def choose_action(self, observation, mask) -> int:
        """只接受 20 维当前观测及合法动作 mask；无环境或真实未来输入。"""
        observation = np.asarray(observation, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if observation.shape != (20,) or not np.isfinite(observation).all():
            raise ValueError("预测拦截需要有限的 20 维当前状态观测")
        if mask.shape != (len(ACTIONS),) or not mask.any():
            raise ValueError("动作 mask 必须含有合法动作")
        config = self.config
        scale = config.grid_size - 1
        coordinate = lambda values: tuple(int(round(float(value) * scale)) for value in values)
        uav, car, car_goal = coordinate(observation[:2]), coordinate(observation[5:7]), coordinate(observation[7:9])
        goal = coordinate(observation[3:5])
        time = int(round(float(observation[9]) * config.max_steps))
        if self._last_time is not None and (time == 0 or time < self._last_time):
            self.reset()
        self._last_time = time
        speed = int(round(float(observation[2]) * 2))
        candidates = []
        for action in np.flatnonzero(mask):
            position, _, invalid = preview_move(uav, speed, int(action), config.grid_size)
            if not invalid:
                candidates.append((int(action), position))
        if not candidates:
            raise ValueError("mask 没有留下地图内的合法动作")
        if observation[11] > 0.5 or observation[12] > 0.5:
            self.target, self.target_time = goal, None
            return self._toward(candidates, goal)

        velocity = tuple(float(value) for value in observation[17:19])
        forecast = self.motion_predictor.predict_positions(
            car, velocity, car_goal, config.grid_size,
            steps=tuple(range(1, self.horizon + config.sensing_steps)),
            move_every=config.car_move_every, time=time,
        )
        radius = config.sensing_radius
        # 已接触或一两格即可接触时，悬停优先；需要移动才用最短位移跟踪。
        current_distance = np.linalg.norm(np.asarray(uav) - car)
        if current_distance <= radius or (self.target is None and current_distance <= radius + 2):
            contacts = [(action, position) for action, position in candidates
                        if np.linalg.norm(np.asarray(position) - forecast[0]) <= radius]
            if contacts:
                return min(contacts, key=lambda item: (
                    np.linalg.norm(np.asarray(item[1]) - uav),
                    np.linalg.norm(np.asarray(item[1]) - forecast[0]), item[0],
                ))[0]

        if self.target is not None:
            expired = time > self.target_time + config.sensing_steps + self.expiry_slack
            missed = np.linalg.norm(forecast - np.asarray(self.target), axis=1).min() > radius + 2
            if expired or missed:
                self.target, self.target_time = None, None
        if self.target is None:
            self.target, self.target_time = self._rendezvous(uav, forecast, time)
        return self._toward(candidates, self.target)
