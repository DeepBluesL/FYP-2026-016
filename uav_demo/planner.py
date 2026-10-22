"""可直接运行的未来轨迹拦截启发式；不是最优 Oracle 求解器。"""

from .environment import GridWorldEnv, preview_move


def minimum_travel_steps(start: tuple[int, int], goal: tuple[int, int]) -> int:
    """八方向、网格步长 1/2 时，无障碍下到指定整数格子的最少步数。

    大坐标差为偶数时，若只用最快步数，每步都必须在该轴移动 2；
    另一轴此时只能累计偶数位移，因此小坐标差为奇数时要多一步。
    最短路线可留在起终点包围的矩形内，地图边界不会额外增加步数。
    """
    low, high = sorted(abs(a - b) for a, b in zip(start, goal))
    return (high + 1) // 2 + int(high % 2 == 0 and low % 2 == 1)


class OraclePlanner:
    """预测最早可拦截位置，接近后跟车采集，完成后飞往终点。"""

    def choose_action(self, env: GridWorldEnv) -> int:
        candidates = []
        for action in range(env.action_count):
            position, speed, invalid = preview_move(
                env.uav_pos, env.uav_speed, action, env.config.grid_size
            )
            if not invalid:
                candidates.append((action, position, speed))

        if env.sensing_finished:
            target = env.config.uav_goal
        else:
            next_car = env.car_trajectory[min(env.time + 1, env.config.max_steps)]
            # 已能在下一步接触车辆时优先保持接触，避免因重规划打断连续采集。
            contacts = [
                item for item in candidates
                if sum((a - b) ** 2 for a, b in zip(item[1], next_car))
                <= env.config.sensing_radius ** 2
            ]
            if contacts:
                return min(
                    contacts,
                    key=lambda item: (
                        sum((a - b) ** 2 for a, b in zip(item[1], next_car)),
                        item[0] != 0,
                        item[0],
                    ),
                )[0]

            # 直接读取整条已知未来轨迹，选择最早能够赶到的车辆位置。
            target = env.car_trajectory[-1]
            for future_time in range(env.time + 1, env.config.max_steps + 1):
                future_position = env.car_trajectory[future_time]
                if minimum_travel_steps(env.uav_pos, future_position) <= future_time - env.time:
                    target = future_position
                    break

        return min(
            candidates,
            key=lambda item: (
                minimum_travel_steps(item[1], target),
                sum(abs(a - b) for a, b in zip(item[1], target)),
                item[0],
            ),
        )[0]
