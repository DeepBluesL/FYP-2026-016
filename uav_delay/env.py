"""Gymnasium 环境：固定步数位置延迟下的跟踪采集任务。

第 t 步只收到第 t-d 步的车辆位置；t < d 时还没有任何报告，不知道车辆位置。
从起点出发跟踪车辆尽量多采集，至少采够 5 步，
并在 100 步内到达终点；到达即结束，失败则本局采集作废。终点已知、直线返程所需步数可以精确算出，
所以返程不交给策略：采够最少量后，时间余量降到 return_margin 时自动直线飞向终点。
全速直飞也赶不到终点时立即判失败。residual > 0 时为残差控制：
实际动作 = 线性外推规则的动作 + residual × 策略输出的修正量。
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import Config
from .contact import contact_time
from .rules import extrapolate_action
from .scenario import scenario


class DelayEnv(gym.Env):
    """动作两维：平面移动；观测 13 维：
    UAV 位置2、相对终点2、剩余步数1、最少采集进度1、
    收到的车辆相对位置2（÷50 m，超出 50 m 时为单位方向）、车辆上一步位移2（÷5）、
    时间余量1（剩余步数 − 直飞到终点所需步数，÷100）、是否已返程1、是否已收到报告1；
    残差控制时再加规则动作2（由上述观测算出，不是新信息），共 15 维。
    首次报告（第 d 步）之前，车辆相对位置和位移填 0，报告标记为 0。
    不提供车辆真实位置、方向、目的地或未来路线。
    """

    metadata = {"render_modes": []}

    def __init__(self, delay=4, cfg=None, seed_range=None):
        if not isinstance(delay, (int, np.integer)) or delay < 0:
            raise ValueError("delay must be a non-negative integer number of steps")
        self.delay, self.cfg, self.seed_range = int(delay), cfg or Config(), seed_range
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        size = 15 if self.cfg.residual > 0 else 13
        self.observation_space = spaces.Box(-1.0, 1.0, (size,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "scenario" in options:
            self.scene = options["scenario"]
        else:
            limits = self.seed_range or (0, 1_000_000)
            scene_seed = (
                seed if seed is not None and self.seed_range is None
                else int(self.np_random.integers(*limits))
            )
            self.scene = scenario(scene_seed, self.cfg)
        self.route = np.asarray(self.scene["route"], dtype=float)
        self.pos = np.array(self.scene["start"], dtype=float)
        self.goal = np.array(self.scene["goal"], dtype=float)
        self.step_count = 0
        self.collected = self.travel = self.total_reward = 0.0
        self.done = self.returning = False
        return self.observation(), self.info(False)

    # ---------- 延迟观测 ----------

    def has_report(self):
        """第 d 步收到第 0 步的报告；在此之前没有任何车辆信息。"""
        return self.step_count >= self.delay

    def observed_step(self):
        return self.step_count - self.delay if self.has_report() else None

    def car_at(self, step):
        return self.route[min(max(step, 0), self.cfg.duration)]

    def received_car(self):
        """收到的车辆位置及其上一步位移，都是 d 步前的数据；尚无报告时返回 None。
        第 0 步的报告没有更早的位置，位移为 0。"""
        seen = self.observed_step()
        if seen is None:
            return None
        return self.car_at(seen), self.car_at(seen) - self.car_at(seen - 1)

    def progress(self):
        # 完成判定与策略看到的进度使用同一精度，避免进度已为1却无法到站。
        return np.float32(min(self.collected / self.cfg.required, 1.0))

    def slack(self):
        """剩余步数 − 全速直飞进入终点范围所需的最少步数；小于 0 即不可能按时到达。"""
        cfg = self.cfg
        distance = max(0.0, float(np.linalg.norm(self.goal - self.pos)) - cfg.goal_radius)
        return cfg.duration - self.step_count - int(np.ceil(distance / cfg.speed - 1e-9))

    def observation(self):
        cfg = self.cfg
        report = self.received_car()
        if report is None:  # 尚无报告：车辆信息填 0，由报告标记区分
            relative, shift = np.zeros(2), np.zeros(2)
        else:
            car, shift = report
            relative = (car - self.pos) / cfg.car_view
            relative = relative / max(1.0, float(np.linalg.norm(relative)))  # 近处保留精度，远处只给方向
        extra = list(extrapolate_action(self)) if cfg.residual > 0 else []
        return np.asarray([
            *(self.pos / cfg.span),
            *((self.goal - self.pos) / cfg.span),
            (cfg.duration - self.step_count) / cfg.duration,
            self.progress(),
            *relative,
            *(shift / cfg.spacing),  # 车辆每步位移 5 m，归一化到 ±1
            np.clip(self.slack() / cfg.duration, -1.0, 1.0),
            float(self.returning),
            float(report is not None),
            *extra,
        ], dtype=np.float32)

    # ---------- 一步动力学与奖励 ----------

    def car_distance(self):
        """奖励用的真实距离，不进入观测。"""
        return float(np.linalg.norm(self.car_at(self.step_count) - self.pos))

    def step(self, action):
        if self.done:
            raise RuntimeError("reset required after episode end")
        cfg = self.cfg
        guided = bool(self.progress() < 1.0)  # 只在采够最少量之前引导靠近车辆
        old_distance = self.car_distance()
        old_pos = self.pos.copy()
        if not self.returning and self.progress() >= 1.0 and self.slack() <= cfg.return_margin:
            self.returning = True
        if self.returning:  # 自动返程：直线全速飞向终点，忽略策略动作
            to_goal = self.goal - old_pos
            move = to_goal * min(1.0, cfg.speed / max(np.linalg.norm(to_goal), 1e-9))
        else:
            move = np.clip(np.asarray(action, dtype=float)[:2], -1, 1)
            if cfg.residual > 0:
                move = extrapolate_action(self) + cfg.residual * move
            move = move * cfg.speed / max(1.0, np.linalg.norm(move))
        self.pos = np.clip(old_pos + move, 0, cfg.span)
        move = self.pos - old_pos

        # 步内按直线运动，计入途中经过采集范围的接触比例。
        car = self.route[self.step_count]
        car_move = self.route[self.step_count + 1] - car
        gained = contact_time(old_pos - car, move - car_move, cfg.radius, 1.0)
        self.step_count += 1
        self.collected += gained
        self.travel += float(np.linalg.norm(move))

        success = bool(self.progress() >= 1.0
                       and np.linalg.norm(self.pos - self.goal) <= cfg.goal_radius)
        failure = not success and (self.step_count >= cfg.duration or self.slack() < 0)
        self.done = bool(success or failure)
        closer = old_distance - self.car_distance() if guided else 0.0
        parts = dict(
            sensing=cfg.sensing * gained,
            approach=cfg.approach * closer / (cfg.span * np.sqrt(2)),
            # 失败（超时或已来不及）：扣回整局采集分再减 2，使失败回合的采集量不再有价值。
            outcome=2.0 if success else (-2.0 - cfg.sensing * self.collected if failure else 0.0),
        )
        reward = float(sum(parts.values()))
        self.total_reward += reward
        info = self.info(success)
        info.update(gained=gained, reward_parts=parts)
        return self.observation(), reward, self.done, False, info

    def info(self, success):
        seen = self.observed_step()
        return dict(
            steps=self.step_count, collected=self.collected, success=bool(success),
            score=self.collected if success else 0.0,
            sensing_complete=bool(self.progress() >= 1.0),
            travel=self.travel, total_reward=self.total_reward,
            observed_step=seen, observation_age=None if seen is None else self.step_count - seen,
            returning=self.returning,
            scenario_seed=self.scene["seed"],
        )
