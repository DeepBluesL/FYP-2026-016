"""检验 Oracle 使用未来信息，且能完成完整任务。"""

from collections import deque
import unittest

import numpy as np

from uav_demo.config import EnvConfig
from uav_demo.environment import ACTIONS, ACTION_NAMES, GridWorldEnv
from uav_demo.planner import OraclePlanner, minimum_travel_steps


class OraclePlannerTests(unittest.TestCase):
    def test_failed_sensing_routes_to_uav_goal(self):
        env = GridWorldEnv(EnvConfig(grid_size=5, max_steps=20, uav_goal=(4, 4), sensing_radius=0))
        env.reset(seed=42)
        env.car_trajectory[:] = (0, 4)
        env.car_goal = (0, 4)
        env.step(0)
        planner = OraclePlanner()
        self.assertTrue(env.sensing_failed)
        for _ in range(2):
            action = planner.choose_action(env)
            self.assertEqual(ACTION_NAMES[action], "north_east_2")
            _, _, terminated, truncated, info = env.step(action)
        self.assertTrue(terminated)
        self.assertFalse(truncated or info["success"])
        self.assertTrue(info["uav_reached_goal"])

    def test_minimum_travel_steps_handles_diagonals_and_parity(self):
        for destination, expected in (
            ((0, 0), 0), ((2, 2), 1), ((2, 1), 2),
            ((4, 3), 3), ((49, 49), 25),
        ):
            with self.subTest(destination=destination):
                self.assertEqual(minimum_travel_steps((0, 0), destination), expected)

    def test_minimum_travel_steps_matches_bfs_with_map_boundaries(self):
        # 用实际动作图的 BFS 作独立参照，覆盖不同方向、奇偶和边界位置。
        size = 7
        points = [(x, y) for x in range(size) for y in range(size)]
        for start in points:
            distances = {start: 0}
            queue = deque([start])
            while queue:
                position = queue.popleft()
                for dx, dy, speed in ACTIONS:
                    candidate = (position[0] + dx * speed, position[1] + dy * speed)
                    if all(0 <= value < size for value in candidate) and candidate not in distances:
                        distances[candidate] = distances[position] + 1
                        queue.append(candidate)
            self.assertEqual(len(distances), size * size)
            for goal in points:
                with self.subTest(start=start, goal=goal):
                    self.assertEqual(minimum_travel_steps(start, goal), distances[goal])

    def test_full_mission_on_independent_routes(self):
        for mode in ("consecutive", "cumulative"):
            for seed in range(2_000_000, 2_000_010):
                with self.subTest(mode=mode, seed=seed):
                    env = GridWorldEnv(EnvConfig(sensing_mode=mode))
                    env.reset(seed=seed)
                    planner = OraclePlanner()
                    for _ in range(env.config.max_steps):
                        action = planner.choose_action(env)
                        self.assertTrue(env.valid_action_mask()[action])
                        _, _, terminated, truncated, info = env.step(action)
                        if terminated or truncated:
                            break
                    self.assertTrue(terminated)
                    self.assertFalse(truncated)
                    self.assertTrue(info["sensing_complete"])
                    self.assertEqual(env.uav_pos, (49, 49))

    def test_future_route_changes_interception_action(self):
        config = EnvConfig(grid_size=9, max_steps=40, uav_goal=(8, 8), sensing_radius=0)
        west_route = GridWorldEnv(config)
        south_route = GridWorldEnv(config)
        for env in (west_route, south_route):
            env.reset(seed=1)
            # 与当前车辆相距 3 格，一步内无法接触，必须使用未来信息。
            env.uav_pos = (5, 8)
        moves = np.minimum(np.arange(config.max_steps + 1) // 2, 8)
        # 两个当前车辆位置完全一样，但未来一个朝左、一个朝下。
        west_route.car_trajectory[:, 0] = 8 - moves
        west_route.car_trajectory[:, 1] = 8
        south_route.car_trajectory[:, 0] = 8
        south_route.car_trajectory[:, 1] = 8 - moves
        self.assertEqual(west_route.car_pos, south_route.car_pos)
        np.testing.assert_array_equal(west_route.car_trajectory[1], south_route.car_trajectory[1])
        planner = OraclePlanner()
        self.assertEqual(ACTION_NAMES[planner.choose_action(west_route)], "east_2")
        self.assertEqual(ACTION_NAMES[planner.choose_action(south_route)], "south_east_1")


if __name__ == "__main__":
    unittest.main()
