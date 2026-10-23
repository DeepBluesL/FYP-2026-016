"""手写基线：只读取公开观测；采样预测不读取车辆真实未来路线。"""

from dataclasses import dataclass
import numpy as np

from .env import Config, DIRECTIONS, car_step, contact_time, return_plan


def decode(obs, cfg):
    """把归一化观测还原为米、秒和能量，索引定义见 EnergyEnv。"""
    return dict(
        pos=np.asarray(obs[:2], float) * cfg.span,
        goal=(np.asarray(obs[:2]) + obs[2:4]) * cfg.span,
        left=float(obs[4]) * cfg.duration,
        energy=float(obs[5]) * cfg.energy,
        collected=float(obs[6]) * cfg.car_duration,
        complete=bool(obs[7] > 0.5),
        car=(np.asarray(obs[:2]) + obs[8:10]) * cfg.span,
        heading=np.asarray(obs[10:12], float),
        age=float(obs[12]) * cfg.duration,
    )


@dataclass(frozen=True)
class Rule:
    lead: bool = True
    goal_side: float = 0.0
    reserve: float = 5.0
    period: int = 2
    adaptive: bool = True
    speed: float = 8.0


class Tracker:
    """预测拦截车辆，并根据剩余时间和电量预算决定何时返航。"""

    def __init__(self, cfg=None, mode="B", rule=None):
        self.cfg, self.mode, self.rule = cfg or Config(), mode, rule or Rule()
        self.returning = False

    def reset(self):
        self.returning = False

    def __call__(self, obs):
        cfg, rule = self.cfg, self.rule
        state = decode(obs, cfg)
        pos, goal, left = state["pos"], state["goal"], state["left"]
        # Straight-line dead reckoning from timestamped telemetry, clipped at the map.
        estimate = np.clip(
            state["car"] + cfg.spacing * state["heading"] * state["age"], 0, cfg.span
        )
        target = estimate.copy()
        if rule.lead:
            # Earliest predicted intercept; model uses no real future trajectory.
            for horizon in range(1, 26):
                target = np.clip(
                    estimate + cfg.spacing * state["heading"] * horizon, 0, cfg.span
                )
                if np.linalg.norm(target - pos) <= min(rule.speed, cfg.speed) * horizon:
                    break
        toward_goal = goal - target
        target += (
            rule.goal_side
            * cfg.radius
            * toward_goal
            / max(np.linalg.norm(toward_goal), 1e-8)
        )
        target = np.clip(target, 0, cfg.span)
        move = target - pos
        move *= min(1.0, min(rule.speed, cfg.speed) / max(np.linalg.norm(move), 1e-8))
        predicted_velocity = cfg.spacing * state["heading"]
        if cfg.duration - left >= cfg.car_duration:
            predicted_velocity = np.zeros(2)
        expected_contact = contact_time(
            pos - estimate, move - predicted_velocity, cfg.radius, 1.0
        )
        sense = expected_contact > 0.02 and cfg.duration - left < cfg.car_duration
        period = rule.period
        if rule.adaptive:
            period = max(
                period, min(12, int(np.linalg.norm(estimate - pos) / cfg.speed) - 1)
            )
        query = self.mode == "C" and state["age"] >= period - 1 - 1e-5
        flight = (
            1
            + float(move @ move) / 25
            + cfg.sensing_power * sense
            + cfg.query_cost * query
        )
        needed, _ = return_plan(np.linalg.norm(goal - pos - move), left - 1, cfg)
        if state["complete"] and (
            state["energy"] - flight < needed + rule.reserve
            or cfg.duration - left >= cfg.car_duration
        ):
            self.returning = True
        if self.returning:
            distance = np.linalg.norm(goal - pos)
            _, speed = return_plan(distance, left, cfg)
            move = (goal - pos) * min(1.0, speed / max(distance, 1e-8))
            return np.array(
                [
                    *(move / cfg.speed),
                    -1.0,
                    -1.0,
                    1.0 if distance <= cfg.speed + 1e-5 else -1.0,
                ],
                np.float32,
            )
        return np.array(
            [*(move / cfg.speed), 1.0 if sense else -1.0, 1.0 if query else -1.0, -1.0],
            np.float32,
        )


class SampledLookahead(Tracker):
    """Four-slot shooting over four target rules and two speed choices.

    Expected contact is averaged over independently sampled public Markov-model
    paths; terminal time/energy feasibility is checked. This is a limited MPC
    baseline, not an exact optimizer. C query timing follows its Tracker rule.
    """

    def __init__(self, cfg=None, mode="B", rule=None, horizon=4, particles=8, seed=0):
        super().__init__(cfg, mode, rule)
        self.horizon, self.particles, self.seed = horizon, particles, seed

    def __call__(self, obs):
        base = super().__call__(obs)
        if self.returning:
            return base
        cfg, state = self.cfg, decode(obs, self.cfg)
        rng = np.random.default_rng(
            self.seed + round((cfg.duration - state["left"]) * 37)
        )
        car_paths = []
        for _ in range(self.particles):
            # Telemetry is float32 but reported car states are grid intersections.
            car = np.rint(state["car"] / cfg.spacing) * cfg.spacing
            heading = int(np.argmax(DIRECTIONS @ state["heading"]))
            for _ in range(round(state["age"])):
                car, heading = car_step(car, heading, rng, cfg)
            path = [car.copy()]
            for j in range(self.horizon):
                if cfg.duration - state["left"] + j < cfg.car_duration:
                    car, heading = car_step(car, heading, rng, cfg)
                path.append(car.copy())
            car_paths.append(path)
        car_paths = np.asarray(car_paths)
        chosen, best = base, -float("inf")
        for lead in (False, True):
            for side in (0.0, 0.7):
                for speed in (5.0, cfg.speed):
                    candidate = Tracker(
                        cfg,
                        self.mode,
                        Rule(
                            lead,
                            side,
                            self.rule.reserve,
                            self.rule.period,
                            self.rule.adaptive,
                            speed,
                        ),
                    )(obs)
                    pos, energy, gain = state["pos"].copy(), state["energy"], 0.0
                    for j in range(self.horizon):
                        if j == 0:
                            move = candidate[:2] * cfg.speed
                        else:
                            target = car_paths[:, j + 1].mean(axis=0)
                            move = target - pos
                            move *= min(1.0, speed / max(np.linalg.norm(move), 1e-8))
                        contacts = np.array(
                            [
                                contact_time(
                                    pos - p[j],
                                    move - (p[j + 1] - p[j]),
                                    cfg.radius,
                                    1.0,
                                )
                                for p in car_paths
                            ]
                        )
                        on = (
                            bool(candidate[2] > 0) if j == 0 else contacts.mean() > 0.02
                        )
                        on = on and cfg.duration - state["left"] + j < cfg.car_duration
                        gain += contacts.mean() * on
                        energy -= 1 + float(move @ move) / 25 + cfg.sensing_power * on
                        if j == 0:
                            energy -= cfg.query_cost * (candidate[3] > 0)
                        pos += move
                    need, _ = return_plan(
                        np.linalg.norm(state["goal"] - pos),
                        state["left"] - self.horizon,
                        cfg,
                    )
                    if (
                        not np.all((pos >= 0) & (pos <= cfg.span))
                        or energy < need + self.rule.reserve
                    ):
                        score = -1000.0 - max(0.0, need + self.rule.reserve - energy)
                    else:
                        score = gain + 0.005 * (energy - need)
                    if score > best:
                        chosen, best = candidate, score
        return chosen
