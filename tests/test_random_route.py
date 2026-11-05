"""随机车辆路线：可复现、每次移动一格、起点在 UAV 附近、到达目的地后继续行驶；默认固定路线不变。"""

import unittest

import numpy as np

from uav_demo.config import EnvConfig
from uav_demo.environment import GridWorldEnv
from uav_demo.vehicle import generate_vehicle_trajectory


class RandomRouteTests(unittest.TestCase):
    def test_default_route_is_still_fixed_corner_to_corner(self):
        route, goal = generate_vehicle_trajectory(EnvConfig(), np.random.default_rng(1))
        self.assertEqual(tuple(route[0]), (49, 0))
        self.assertEqual(goal, (0, 49))

    def test_random_route_is_reproducible_moves_one_cell_and_starts_near_uav(self):
        config = EnvConfig(car_route="random")
        starts = set()
        for seed in range(20):
            a, goal_a = generate_vehicle_trajectory(config, np.random.default_rng(seed))
            b, goal_b = generate_vehicle_trajectory(config, np.random.default_rng(seed))
            np.testing.assert_array_equal(a, b)
            self.assertEqual(goal_a, goal_b)
            np.testing.assert_array_equal(np.abs(np.diff(a, axis=0)).sum(axis=1), 1)  # 不停车
            self.assertTrue(8 <= np.linalg.norm(a[0] - np.asarray(config.uav_start)) <= 15)
            self.assertTrue(np.all((a >= 0) & (a < config.grid_size)))
            self.assertGreaterEqual(np.abs(np.asarray(goal_a) - a[0]).sum(), 20)
            starts.add(tuple(a[0]))
        self.assertGreater(len(starts), 5)  # 起点确实随机

    def test_passing_first_destination_does_not_fail_sensing_on_random_route(self):
        env = GridWorldEnv(EnvConfig(car_route="random"))
        env.reset(seed=3)
        while not env._done:
            env.step(0)  # 原地悬停，采集不到；随机路线只会超时失败
        self.assertFalse(env.sensing_failed)
        self.assertEqual(env.time, env.config.max_steps)


if __name__ == "__main__":
    unittest.main()
