"""任务环境：车辆轨迹、连续运动、接触时长、能量消耗和 B/C 观测。

阅读顺序：Config → scenario → EnergyEnv.reset → observation → step。
训练奖励的额外处理放在 learn.py，本文件计算原始任务奖励。
"""

from dataclasses import dataclass
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from uav_demo.vehicle import DIRECTIONS


@dataclass(frozen=True)
class Config:
    span: float = 245.0  # 正方形地图边长，m
    spacing: float = 5.0  # 车辆每秒行驶距离，m
    speed: float = 8.0  # UAV 速度上限，m/s
    radius: float = 4.0  # 感知半径，m
    duration: float = 100.0  # 任务截止时间，s
    car_duration: int = 70  # 车辆行驶时间，s；之后停止采集
    required: float = 5.0  # 成功所需累计接触时长，s
    energy: float = 240.0  # 初始能量，使用模型内部单位
    sensing_power: float = 0.5  # 开启感知的额外功率
    query_cost: float = 2.0  # 每次定位查询的能量开销
    # 直行、右转、掉头、左转；在边界处重新归一化。
    turns: tuple = (0.85, 0.07, 0.01, 0.07)

    def __post_init__(self):
        if (
            min(
                self.span,
                self.spacing,
                self.speed,
                self.radius,
                self.duration,
                self.required,
                self.energy,
            )
            <= 0
        ):
            raise ValueError("Physical scales and budgets must be positive")
        if self.sensing_power < 0 or self.query_cost < 0 or self.car_duration < 1:
            raise ValueError("Invalid power, cost, or car duration")
        if (
            len(self.turns) != 4
            or min(self.turns) < 0
            or not np.isclose(sum(self.turns), 1)
        ):
            raise ValueError("Four turn probabilities must sum to one")


def car_step(position, heading, rng, cfg):
    options = position + cfg.spacing * DIRECTIONS
    valid = np.all((options >= 0) & (options <= cfg.span), axis=1)
    weights = np.asarray(cfg.turns)[(np.arange(4) - heading) % 4] * valid
    heading = int(rng.choice(4, p=weights / weights.sum()))
    return options[heading].copy(), heading


def scenario(seed, cfg):
    """Joint initial geometry only; no conditioning on hidden future outcomes."""
    rng = np.random.default_rng(seed)
    size = round(cfg.span / cfg.spacing) + 1
    for _ in range(10000):
        start, goal, car = rng.integers(size, size=(3, 2)) * cfg.spacing
        if (
            225 <= np.linalg.norm(goal - start) <= 300
            and 40 <= np.linalg.norm(car - start) <= 75
        ):
            break
    else:
        raise RuntimeError("Initial geometry sampling budget exhausted")
    heading = int(rng.integers(4))
    route, headings = [car.copy()], [heading]
    for _ in range(cfg.car_duration):
        car, heading = car_step(car, heading, rng, cfg)
        route.append(car.copy())
        headings.append(heading)
    return dict(
        seed=int(seed),
        start=start.tolist(),
        goal=goal.tolist(),
        route=np.asarray(route).tolist(),
        headings=headings,
    )


def contact_time(relative_position, relative_velocity, radius, dt):
    """解二次不等式 ||相对位置 + 相对速度*t||² <= 半径²，返回接触秒数。"""
    a = float(relative_velocity @ relative_velocity)
    c = float(relative_position @ relative_position - radius * radius)
    if a < 1e-14:
        return dt if c <= 1e-10 else 0.0
    b = 2 * float(relative_position @ relative_velocity)
    discriminant = b * b - 4 * a * c
    if discriminant <= 0:
        return 0.0  # a single tangent instant has zero duration
    root = np.sqrt(discriminant)
    return max(0.0, min(dt, (-b + root) / (2 * a)) - max(0.0, (-b - root) / (2 * a)))


def return_plan(distance, time_left, cfg):
    """Minimum propulsion energy for direct return under this abstract power law."""
    if distance < 1e-9:
        return 0.0, 0.0
    if time_left <= 0 or distance > cfg.speed * time_left + 1e-8:
        return float("inf"), cfg.speed
    speed = min(cfg.speed, max(5.0, distance / time_left))
    return distance / speed * (1 + (speed / 5) ** 2), speed


class EnergyEnv(gym.Env):
    """动作五维：[移动 x, 移动 y, 感知开关, 查询开关, 到站开关]。

    观测共 14 维，位置、时间、能量和采集量均已归一化：
    0:2 UAV 位置；2:4 相对终点；4 剩余时间；5 电量；6 累计接触时长；
    7 是否采满 K；8:10 最近车辆相对位置；10:12 最近车辆方向；
    12 遥测年龄；13 本次是否获得新遥测。

    B 每步获得新遥测。C 的查询在动作执行后更新遥测，不能影响本步移动。
    """

    metadata = {"render_modes": []}

    def __init__(self, mode="B", cfg=None, seed_range=None):
        if mode not in ("B", "C"):
            raise ValueError("mode must be B or C")
        self.mode, self.cfg, self.seed_range = mode, cfg or Config(), seed_range
        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(14,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "scenario" in options:
            scene = options["scenario"]
        else:
            scene_seed = (
                int(self.np_random.integers(*self.seed_range))
                if self.seed_range
                else (0 if seed is None else seed)
            )
            scene = scenario(scene_seed, self.cfg)
        self.scene = scene
        self.route = np.asarray(scene["route"], dtype=float)
        if self.route.shape != (self.cfg.car_duration + 1, 2):
            raise ValueError("Scenario route length does not match car duration")
        self.pos, self.goal = np.array(scene["start"], float), np.array(
            scene["goal"], float
        )
        self.time, self.energy, self.collected = 0.0, self.cfg.energy, 0.0
        self.travel, self.queries, self.invalid = 0.0, 0, 0
        self.energy_flight = self.energy_sensing = self.energy_queries = (
            self.total_reward
        ) = 0.0
        self.done = False
        self.last_car, self.last_heading = self.route[0].copy(), self._heading(0)
        self.stamp, self.fresh = 0.0, True
        return self.observation(), self.info()

    def _car(self, time):
        i = min(int(time), self.cfg.car_duration)
        if i >= self.cfg.car_duration:
            return self.route[-1].copy()
        return self.route[i] + (time - i) * (self.route[i + 1] - self.route[i])

    def _heading(self, time):
        if time >= self.cfg.car_duration:
            return np.zeros(2)
        return DIRECTIONS[self.scene["headings"][int(time)]].astype(float)

    def observation(self):
        cfg = self.cfg
        return np.asarray(
            [
                *(self.pos / cfg.span),
                *((self.goal - self.pos) / cfg.span),
                (cfg.duration - self.time) / cfg.duration,
                self.energy / cfg.energy,
                self.collected / cfg.car_duration,
                float(self.collected >= cfg.required),
                *((self.last_car - self.pos) / cfg.span),
                *self.last_heading,
                (self.time - self.stamp) / cfg.duration,
                float(self.fresh),
            ],
            dtype=np.float32,
        )

    def step(self, action):
        # 1. 检查动作，把归一化的二维移动转换为实际速度。
        if self.done:
            raise RuntimeError("reset required after termination")
        action = np.asarray(action, dtype=float)
        if action.shape != (5,) or not np.isfinite(action).all():
            raise ValueError("Expected five finite action components")
        cfg = self.cfg
        action = np.clip(action, -1, 1)
        velocity = action[:2] * cfg.speed / max(1.0, np.linalg.norm(action[:2]))
        sense, query = action[2] > 0, action[3] > 0
        previous_position, previous_collected = self.pos.copy(), self.collected
        previous_goal_distance = float(np.linalg.norm(self.goal - self.pos))
        # 2. 满足采集门槛且距终点不超过一步时，可以执行末段到站。
        dock = (
            action[4] > 0
            and self.collected >= cfg.required
            and previous_goal_distance <= cfg.speed + 1e-8
        )
        requested_dt = previous_goal_distance / cfg.speed if dock else 1.0
        if dock:
            velocity = (self.goal - self.pos) / max(requested_dt, 1e-12)
        invalid = bool(
            np.any(self.pos + velocity * requested_dt < -1e-8)
            or np.any(self.pos + velocity * requested_dt > cfg.span + 1e-8)
        )
        if invalid:
            velocity = np.zeros(2)
            self.invalid += 1
        # 3. 先支付定位查询费用，再确定剩余能量能支持多长时间。
        cost_query = cfg.query_cost if query else 0.0
        paid = min(self.energy, cost_query)
        self.energy -= paid
        self.energy_queries += paid
        self.queries += int(query)
        power = 1 + float(velocity @ velocity) / 25 + cfg.sensing_power * sense
        actual_dt = max(
            0.0, min(requested_dt, cfg.duration - self.time, self.energy / power)
        )
        # 4. 根据本步的相对运动，精确计算感知开启时的接触秒数。
        car_position = self._car(self.time)
        car_velocity = (
            self._car(self.time + min(1.0, cfg.duration - self.time)) - car_position
        )
        sensing_dt = min(actual_dt, max(0.0, cfg.car_duration - self.time))
        gained = (
            contact_time(
                self.pos - car_position, velocity - car_velocity, cfg.radius, sensing_dt
            )
            if sense
            else 0.0
        )
        # 5. 同步更新位置、时间、采集量和三类能耗账目。
        self.pos = np.clip(self.pos + velocity * actual_dt, 0, cfg.span)
        if dock and actual_dt >= requested_dt - 1e-9:
            self.pos = self.goal.copy()
        self.collected += gained
        self.energy_flight += actual_dt * (1 + float(velocity @ velocity) / 25)
        self.energy_sensing += actual_dt * cfg.sensing_power * sense
        self.energy = max(0.0, self.energy - actual_dt * power)
        self.travel += float(np.linalg.norm(self.pos - previous_position))
        self.time += actual_dt
        # 6. 动作结束后才更新可见的车辆遥测。
        self.fresh = self.mode == "B" or query
        if self.fresh:
            self.last_car, self.last_heading, self.stamp = (
                self._car(self.time),
                self._heading(self.time),
                self.time,
            )
        success = (
            self.collected >= cfg.required
            and np.linalg.norm(self.pos - self.goal) < 1e-8
        )
        failure = not success and (
            self.time >= cfg.duration - 1e-9 or self.energy <= 1e-9
        )
        self.done = bool(success or failure)
        reward = (
            gained
            + 20 * (previous_collected < cfg.required <= self.collected)
            + 20 * success
            - 200 * failure
            - invalid
            + 0.01 * (previous_goal_distance - np.linalg.norm(self.goal - self.pos))
        )
        self.total_reward += float(reward)
        info = self.info()
        info.update(
            gained=gained,
            reward=float(reward),
            sensing=bool(sense),
            query=bool(query),
            dock=bool(dock),
        )
        return self.observation(), float(reward), self.done, False, info

    def info(self):
        success = (
            self.collected >= self.cfg.required
            and np.linalg.norm(self.pos - self.goal) < 1e-8
        )
        failure_reason = ""
        if self.done and not success:
            failure_reason = "energy" if self.energy <= 1e-9 else "time"
        return dict(
            time=self.time,
            uav=self.pos.tolist(),
            car=self._car(self.time).tolist(),
            energy=self.energy,
            collected=self.collected,
            success=bool(success),
            failure=failure_reason,
            queries=self.queries,
            travel=self.travel,
            invalid=self.invalid,
            energy_flight=self.energy_flight,
            energy_sensing=self.energy_sensing,
            energy_queries=self.energy_queries,
            total_reward=self.total_reward,
            scenario_seed=self.scene["seed"],
        )


class FixedSceneEnv(EnergyEnv):
    """Repeat the same start, goal and complete car route on every reset.

    Physics, reward, observations and docking are inherited without changes.
    The policy receives only observations; reset randomness cannot change route.
    B's unavailable query action is disabled, as in the multi-scene experiment.
    """

    def __init__(self, mode, cfg, scene):
        super().__init__(mode, cfg)
        self.fixed_scene = scene

    def reset(self, *, seed=None, options=None):
        return super().reset(seed=seed, options={"scenario": self.fixed_scene})

    def step(self, action):
        action = np.asarray(action).copy()
        if self.mode == "B":
            action[3] = -1.0
        return super().step(action)
