"""完整路线数据的分组隔离、按需生成和可复现性。"""

import unittest
from unittest.mock import patch

import numpy as np

from uav_demo.config import EnvConfig
from uav_demo.prediction_data import iter_prediction_routes, split_route_seeds
from uav_demo.vehicle import generate_vehicle_trajectory


class PredictionDataTests(unittest.TestCase):
    def test_seed_groups_are_disjoint_complete_and_reproducible(self):
        groups = split_route_seeds()
        self.assertEqual([len(group) for group in groups.values()], [800, 100, 100])
        seeds = sum(groups.values(), [])
        self.assertEqual(sorted(seeds), list(range(1000)))
        self.assertEqual(groups, split_route_seeds())
        self.assertNotEqual(groups, split_route_seeds(split_seed=43))

    def test_each_seed_returns_one_complete_route_with_all_timestamps(self):
        config = EnvConfig(grid_size=5, uav_goal=(4, 4), max_steps=20,
                           observation_delay=3, car_move_every=2)
        samples = list(iter_prediction_routes(config, [7, 8]))
        self.assertEqual(len(samples), 2)
        for seed, sample in zip([7, 8], samples):
            route, _ = generate_vehicle_trajectory(config, np.random.default_rng(seed))
            self.assertEqual(sample["route_seed"], seed)
            self.assertEqual(sample["features"].shape, (21, 3))
            self.assertEqual(sample["features"].dtype, np.float32)
            np.testing.assert_allclose(sample["features"][:, :2], route / 4)
            np.testing.assert_allclose(sample["features"][:, 2], np.arange(21) / 20)
            np.testing.assert_array_equal(sample["report_times"], np.arange(21))

    def test_routes_are_generated_on_demand_and_reproducible(self):
        config = EnvConfig(grid_size=5, uav_goal=(4, 4), max_steps=20)
        with patch("uav_demo.prediction_data.generate_vehicle_trajectory",
                   wraps=generate_vehicle_trajectory) as generate:
            stream = iter_prediction_routes(config, [7, 8])
            generate.assert_not_called()
            first = next(stream)
            self.assertEqual(generate.call_count, 1)
            second = next(stream)
            self.assertEqual(generate.call_count, 2)
            with self.assertRaises(StopIteration):
                next(stream)
        repeated = next(iter_prediction_routes(config, [7]))
        np.testing.assert_array_equal(first["features"], repeated["features"])
        second["features"][:] = -1
        np.testing.assert_array_equal(first["features"], repeated["features"])


if __name__ == "__main__":
    unittest.main()
