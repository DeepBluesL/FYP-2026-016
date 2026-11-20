"""Project AirSim（Unreal Engine）接入示例：回放导出的测试路线，用我们的策略控制无人机。

主循环（车辆轨迹、延迟报告、采集判定、成功判定）都在本脚本里，和训练环境的规则逐条一致；
仿真平台只负责“读无人机位置”和“按速度飞 1 s”。两个后端：

- grid：理想网格，无人机每步正好到达指令格子（等同训练环境）。不需要 Project AirSim，
  用来确认主循环无误：结果应和 routes 文件里的参考结果逐条相同。
- projectairsim：真实接入。标 TODO 的地方要按 Project AirSim 官方 API 文档核对后填写。

用法：
    python deploy/projectairsim_example.py --backend grid --delay 6
    python deploy/projectairsim_example.py --backend projectairsim --delay 6 --episodes 5 --scene scene_basic_drone.jsonc
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy import ObservationBuilder, Policy  # noqa: E402

CELL_M = 5.0  # 1 格 = 5 m
STEP_S = 1.0  # 1 步 = 1 s
ROUTES = Path(__file__).resolve().parent / "routes" / "test_routes.json"


class GridBackend:
    """理想网格：无人机每一步都正好到达指令格子。"""

    async def reset(self, start_cell):
        self.cell = tuple(start_cell)

    async def uav_cell(self):
        return self.cell

    async def fly(self, current_cell, next_cell):
        self.cell = tuple(next_cell)

    async def show_car(self, car_cell):
        pass

    async def close(self):
        pass


class ProjectAirSimBackend:
    """Project AirSim：用 move_by_velocity_async 在 1 s 内飞向下一个格子，再读实际位置换算回格子。

    坐标：Project AirSim 用 NED（北、东、下）；我们的格子 x 向东、y 向北，原点是无人机起点。
    """

    def __init__(self, scene: str, drone_name: str, altitude_m: float, grid_size: int):
        self.scene, self.drone_name, self.altitude_m, self.grid_size = scene, drone_name, altitude_m, grid_size

    async def connect(self):
        from projectairsim import Drone, ProjectAirSimClient, World
        self.client = ProjectAirSimClient()
        self.client.connect()
        self.world = World(self.client, self.scene, delay_after_load_sec=2)
        self.drone = Drone(self.client, self.world, self.drone_name)
        self.drone.enable_api_control()
        self.drone.arm()
        await (await self.drone.takeoff_async())
        # TODO: 爬升到固定高度 self.altitude_m（例如用向上的速度飞一段，或官方的定高 API）。

    def _north_east(self) -> tuple[float, float]:
        # TODO: 按官方 API 文档核对读取真实位置的函数和返回格式。
        position = self.drone.get_ground_truth_kinematics()["pose"]["position"]
        return float(position["x"]), float(position["y"])

    async def reset(self, start_cell):
        # TODO: 每局开始前把无人机放回起点（重新加载场景，或飞回起点）。下面把当前位置当作起点格子。
        north, east = self._north_east()
        self.origin = (north - start_cell[1] * CELL_M, east - start_cell[0] * CELL_M)

    async def uav_cell(self):
        north, east = self._north_east()
        last = self.grid_size - 1
        x = min(max(round((east - self.origin[1]) / CELL_M), 0), last)
        y = min(max(round((north - self.origin[0]) / CELL_M), 0), last)
        return (x, y)

    async def fly(self, current_cell, next_cell):
        dx, dy = next_cell[0] - current_cell[0], next_cell[1] - current_cell[1]
        task = await self.drone.move_by_velocity_async(
            v_north=dy * CELL_M / STEP_S, v_east=dx * CELL_M / STEP_S, v_down=0.0, duration=STEP_S)
        await task

    async def show_car(self, car_cell):
        # 可选：把场景里的车模型移到 (car_cell × 5 m)，只用于显示，不影响判定。
        pass

    async def close(self):
        await (await self.drone.land_async())
        self.drone.disarm()
        self.drone.disable_api_control()
        self.client.disconnect()


async def run_episode(policy: Policy, backend, car: list) -> dict:
    """一局：规则与训练环境相同（报告晚 d 步、采集需连续 5 步在 10 m 内、采完后自动返程、150 步上限）。"""
    config = policy.config
    goal = tuple(config.uav_goal)
    builder = ObservationBuilder(config)
    speed, progress, complete, sensed_at = config.initial_speed, 0, False, None
    await backend.reset(config.uav_start)
    cell = await backend.uav_cell()
    for t in range(config.max_steps):
        if t - policy.delay >= 0:  # 第 t 秒收到车在第 t − d 秒的位置
            builder.report(t - policy.delay, car[t - policy.delay])
        observation = builder.build(t, cell, speed, progress, complete)
        target, speed = policy.next_state(observation, policy.act(observation))
        await backend.show_car(car[t])
        await backend.fly(cell, target)
        cell, now = await backend.uav_cell(), t + 1
        if not complete:  # 每秒判定一次：与车（真实位置）的距离 ≤ 10 m 记一次接触，中断清零
            near = math.dist(cell, car[now]) <= config.sensing_radius
            progress = progress + 1 if near else 0
            if progress >= config.sensing_steps:
                progress, complete, sensed_at = config.sensing_steps, True, now
        if complete and cell == goal:
            return {"success": True, "steps": now, "sensed_at": sensed_at}
    return {"success": False, "steps": config.max_steps, "sensed_at": sensed_at}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("grid", "projectairsim"), default="grid")
    parser.add_argument("--delay", type=int, choices=(4, 6, 8), default=6)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--routes", type=Path, default=ROUTES)
    parser.add_argument("--scene", default="scene_basic_drone.jsonc")
    parser.add_argument("--drone", default="Drone1")
    parser.add_argument("--altitude", type=float, default=10.0, help="飞行高度（m），只用于 projectairsim")
    args = parser.parse_args()

    data = json.loads(args.routes.read_text(encoding="utf-8"))
    policy = Policy(delay=args.delay)
    routes = data["routes"][:args.episodes]
    reference = data["reference"][str(args.delay)]["cv_prediction_plus_rl"]
    if args.backend == "grid":
        backend = GridBackend()
    else:
        backend = ProjectAirSimBackend(args.scene, args.drone, args.altitude, policy.config.grid_size)
        await backend.connect()

    results, mismatches = [], 0
    try:
        for route, ref in zip(routes, reference):
            car = [tuple(p) for p in route["car"]]
            result = await run_episode(policy, backend, car)
            results.append(result)
            same = result["success"] == ref["success"] and result["steps"] == ref["steps"]
            mismatches += not same
            print(f"seed {route['seed']}: {'goal' if result['success'] else 'fail'} at {result['steps']} "
                  f"(reference {'goal' if ref['success'] else 'fail'} at {ref['steps']})")
    finally:
        await backend.close()

    n, max_steps = len(results), policy.config.max_steps
    def summary(rows):
        return (sum(r["success"] for r in rows) / n,
                sum(r["steps"] if r["success"] else max_steps for r in rows) / n)
    ours, ref = summary(results), summary(reference[:n])
    print(f"\nd={args.delay}, {n} routes, backend={args.backend}")
    print(f"this run : success {ours[0]:.1%}, mean completion {ours[1]:.1f} steps")
    print(f"reference: success {ref[0]:.1%}, mean completion {ref[1]:.1f} steps")
    print(f"routes with a different result: {mismatches}")
    if args.backend == "grid" and mismatches:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
