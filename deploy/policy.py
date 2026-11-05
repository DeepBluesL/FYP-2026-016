"""仿真平台调用接口：加载训练好的 DQN（外推规则先验 b=0.1 + 简化奖励），每步输入观测、输出动作编号。

用法：
    policy = Policy(delay=6)                 # 或 Policy("deploy/models/d6.pt")
    builder = ObservationBuilder(policy.config)
    每一步：builder.report(t, car_pos)       # 第 t 步时刻车的位置报告到达（可能是更早时刻的位置）
            obs = builder.build(time, uav_pos, uav_speed, sensing_progress, sensing_complete)
            action = policy.act(obs)        # 0..16，含义见 ACTION_NAMES / ACTIONS
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_demo.agent import DQNAgent  # noqa: E402
from uav_demo.environment import ACTION_NAMES, ACTIONS, preview_move  # noqa: E402
from uav_demo.planner import minimum_travel_steps  # noqa: E402

MODELS = Path(__file__).resolve().parent / "models"
REQUIRED_KEYS = ("uav_pos", "uav_speed", "uav_goal", "car_pos", "car_velocity", "time", "sensing_progress",
                 "sensing_complete", "sensing_failed", "observation_age", "has_observation")

__all__ = ["Policy", "ObservationBuilder", "ACTIONS", "ACTION_NAMES"]


class Policy:
    """每步一次 act(observation)；无内部状态，不需要在回合之间 reset。"""

    def __init__(self, model_path: str | Path | None = None, delay: int | None = None):
        if model_path is None:
            if delay is None:
                raise ValueError("model_path 和 delay 至少给一个")
            model_path = MODELS / f"d{delay}.pt"
        self.agent, self.config = DQNAgent.load(model_path)
        if delay is not None and delay != self.config.observation_delay:
            raise ValueError(f"模型按延迟 {self.config.observation_delay} 训练，与 delay={delay} 不一致")

    @property
    def delay(self) -> int:
        return self.config.observation_delay

    def valid_mask(self, observation: dict) -> np.ndarray:
        """与训练环境相同：屏蔽出界动作；采集结束后（auto_return）只保留最短路径返程动作。"""
        observation = _normalized(observation)
        position, speed = observation["uav_pos"], observation["uav_speed"]
        grid = self.config.grid_size
        if self.config.auto_return and (observation["sensing_complete"] or observation["sensing_failed"]):
            goal = observation["uav_goal"]
            options = []
            for action in range(len(ACTIONS)):
                target, _, invalid = preview_move(position, speed, action, grid)
                if not invalid:
                    distance = sum(abs(a - b) for a, b in zip(target, goal))
                    options.append((minimum_travel_steps(target, goal), distance, action))
            mask = np.zeros(len(ACTIONS), dtype=bool)
            mask[min(options)[2]] = True
            return mask
        return np.asarray([not preview_move(position, speed, action, grid)[2]
                           for action in range(len(ACTIONS))], dtype=bool)

    def act(self, observation: dict) -> int:
        observation = _normalized(observation)
        return self.agent.choose_action(observation, epsilon=0.0, mask=self.valid_mask(observation))

    def next_state(self, observation: dict, action: int) -> tuple[tuple[int, int], int]:
        """执行动作后的网格位置和速度档（出界则原地不动），供仿真端换算成飞行指令。"""
        observation = _normalized(observation)
        position, speed, _ = preview_move(observation["uav_pos"], observation["uav_speed"], action,
                                          self.config.grid_size)
        return tuple(position), speed


class ObservationBuilder:
    """把仿真端收到的延迟车辆报告整理成模型要的观测；每回合新建一个。

    report(t, pos) 记录“第 t 步时刻车的位置”这条报告（仿真端在它到达时调用，t 是位置对应的时刻）。
    车速 car_velocity 取最近一次非零位移（相邻两条报告之差），与训练环境一致。
    """

    def __init__(self, config):
        self.config = config
        self.reports: dict[int, tuple[int, int]] = {}

    def report(self, report_time: int, car_pos) -> None:
        self.reports[int(report_time)] = tuple(int(v) for v in car_pos)

    def build(self, time: int, uav_pos, uav_speed: int, sensing_progress: int,
              sensing_complete: bool, sensing_failed: bool = False) -> dict:
        latest = max((t for t in self.reports if t <= time), default=None)
        car_pos = car_velocity = None
        if latest is not None:
            car_pos, car_velocity = self.reports[latest], (0, 0)
            for t in range(latest, 0, -1):
                if t in self.reports and t - 1 in self.reports:
                    step = tuple(b - a for a, b in zip(self.reports[t - 1], self.reports[t]))
                    if any(step):
                        car_velocity = step
                        break
        return {
            "uav_pos": tuple(int(v) for v in uav_pos), "uav_speed": int(uav_speed),
            "uav_goal": tuple(self.config.uav_goal),
            "car_pos": car_pos, "car_velocity": car_velocity, "time": int(time),
            "sensing_progress": int(sensing_progress), "sensing_complete": bool(sensing_complete),
            "sensing_failed": bool(sensing_failed),
            "observation_age": time - latest if latest is not None else 0,
            "has_observation": latest is not None,
        }


def _normalized(observation: dict) -> dict:
    missing = [key for key in REQUIRED_KEYS if key not in observation]
    if missing:
        raise KeyError(f"观测缺少字段 {missing}")
    result = dict(observation)
    for key in ("uav_pos", "uav_goal", "car_pos", "car_velocity"):
        if result[key] is not None:
            result[key] = tuple(int(v) for v in result[key])
    return result
