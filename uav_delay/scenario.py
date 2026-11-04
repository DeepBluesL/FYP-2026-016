"""场景生成：UAV 起点与终点、车辆起点，以及整局车辆路线。路线开局生成，策略只能经延迟观测看到。"""

import numpy as np

# 顺时针：北、东、南、西，便于按相对朝向计算直行 / 右转 / 掉头 / 左转。
DIRECTIONS = np.asarray(((0, 1), (1, 0), (0, -1), (-1, 0)), dtype=np.int64)


def scenario(seed, cfg):
    """起点、终点相距 225–300 m，车辆起点离 UAV 起点 40–75 m；再生成车辆路线。"""
    rng = np.random.default_rng(seed)
    cells = round(cfg.span / cfg.spacing) + 1
    for _ in range(10000):
        start, goal, car = rng.integers(cells, size=(3, 2)) * cfg.spacing
        if 225 <= np.linalg.norm(goal - start) <= 300 and 40 <= np.linalg.norm(car - start) <= 75:
            break
    else:
        raise RuntimeError("Initial geometry sampling budget exhausted")
    heading = int(rng.integers(4))
    route = destination_route(seed, car.astype(float), heading, cfg)
    return dict(seed=int(seed), start=start.tolist(), goal=goal.tolist(), route=route.tolist())


def destination_route(seed, car, heading, cfg):
    """车辆按原转向概率 × exp(bias × 本步缩短的曼哈顿距离 / 5) 选方向；到达后换新目的地。"""
    rng = np.random.default_rng(seed + 99)
    cells = round(cfg.span / cfg.spacing) + 1

    def new_goal(origin):
        while True:
            goal = rng.integers(cells, size=2) * cfg.spacing
            if np.abs(goal - origin).sum() >= 100:
                return goal

    goal, route = new_goal(car), [car.copy()]
    for _ in range(cfg.duration):
        if np.array_equal(car, goal):
            goal = new_goal(car)
        options = car + cfg.spacing * DIRECTIONS
        valid = np.all((options >= 0) & (options <= cfg.span), axis=1)
        closer = (np.abs(car - goal).sum() - np.abs(options - goal).sum(axis=1)) / cfg.spacing
        weights = np.asarray(cfg.turns)[(np.arange(4) - heading) % 4]
        weights = weights * np.exp(cfg.car_goal_bias * closer) * valid
        heading = int(rng.choice(4, p=weights / weights.sum()))
        car = options[heading].copy()
        route.append(car.copy())
    return np.asarray(route)
