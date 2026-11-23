"""目的地推断预测器：参数估计、后验更新、向前抽样与车辆生成器比对、点预测取法、接入训练与保存加载。"""

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from uav_demo.prediction import report_history
from train import train
from uav_demo.agent import DQNAgent
from uav_demo.config import EnvConfig, TrainConfig
from uav_demo.goal_inference import GoalFilterPredictor
from uav_demo.point_selection import disk_sum
from uav_demo.prediction_data import split_route_seeds
from uav_demo.vehicle import DIRECTIONS, generate_vehicle_trajectory

RANDOM = EnvConfig(car_route="random")
TRUE_LOG_WEIGHTS = np.log([0.65, 0.15, 0.05, 0.15])


def true_model(**options):
    return GoalFilterPredictor(TRUE_LOG_WEIGHTS, RANDOM.car_goal_bias, 20, **options)


def exact_known_goal(position, heading: int, goal, steps: int) -> np.ndarray:
    """目的地已知且途中不换时，按车辆生成器的规则精确推演 steps 步后的车位分布 [50, 50]。"""
    goal = np.asarray(goal)
    states = {(tuple(position), heading): 1.0}
    for _ in range(steps):
        moved = {}
        for (cell, facing), probability in states.items():
            options = np.asarray(cell) + DIRECTIONS
            legal = np.flatnonzero(np.all((options >= 0) & (options < 50), axis=1))
            progress = np.abs(np.asarray(cell) - goal).sum() - np.abs(options[legal] - goal).sum(axis=1)
            logits = TRUE_LOG_WEIGHTS[(legal - facing) % 4] + RANDOM.car_goal_bias * progress
            weights = np.exp(logits - logits.max())
            for direction, weight in zip(legal, weights / weights.sum()):
                key = (tuple(options[direction]), int(direction))
                moved[key] = moved.get(key, 0.0) + probability * weight
        states = moved
    grid = np.zeros((50, 50))
    for (cell, _), probability in states.items():
        grid[cell] += probability
    return grid


class FitTests(unittest.TestCase):
    def test_fit_recovers_generator_parameters(self):
        with redirect_stdout(StringIO()):
            model, metrics = GoalFilterPredictor.fit(RANDOM, split_route_seeds(200, start_seed=70_000))
        np.testing.assert_allclose(metrics["turn_probabilities"], (0.65, 0.15, 0.05, 0.15), atol=0.03)
        self.assertAlmostEqual(model.goal_bias, 1.0, delta=0.15)
        self.assertEqual(model.min_goal_distance, 20)
        self.assertGreater(metrics["validation_log_likelihood"], np.log(0.25))

    def test_only_random_routes_with_one_move_per_step(self):
        with self.assertRaisesRegex(ValueError, "car_route=random"):
            GoalFilterPredictor.fit(EnvConfig(), split_route_seeds(10))
        with self.assertRaisesRegex(ValueError, "每步移动一格"):
            true_model().predict_positions([(0, (10, 10))], 50, (1,), move_every=2)


class BeliefTests(unittest.TestCase):
    def test_prior_is_uniform_over_far_cells(self):
        belief, heading = true_model().belief([(0, (10, 5))], 50)
        cells = np.indices((50, 50)).reshape(2, -1).T
        far = np.abs(cells - (10, 5)).sum(axis=1) >= 20
        self.assertIsNone(heading)
        self.assertAlmostEqual(belief.sum(), 1.0)
        np.testing.assert_allclose(belief[far], 1 / far.sum())
        self.assertTrue((belief[~far] == 0).all())

    def test_incremental_update_matches_full_recomputation(self):
        route, _ = generate_vehicle_trajectory(RANDOM, np.random.default_rng(3))
        history = report_history(route, 40, 0)
        model = true_model()
        model.belief(history[:-1], 50)
        incremental, heading = model.belief(history, 50)
        full, full_heading = true_model().belief(history, 50)
        np.testing.assert_allclose(incremental, full)
        self.assertEqual(heading, full_heading)

    def test_posterior_moves_toward_true_goal(self):
        cells = np.indices((50, 50)).reshape(2, -1).T
        prior_mass, posterior_mass = [], []
        for seed in range(30):
            route, _, goals = generate_vehicle_trajectory(RANDOM, np.random.default_rng(seed), with_goals=True)
            near = np.abs(cells - goals[30]).sum(axis=1) <= 8
            prior_mass.append(true_model().belief(report_history(route, 0, 0), 50)[0][near].sum())
            posterior_mass.append(true_model().belief(report_history(route, 29, 0), 50)[0][near].sum())
        self.assertGreater(np.mean(posterior_mass), 2 * np.mean(prior_mass))


class SamplingTests(unittest.TestCase):
    def test_forward_samples_match_exact_propagation_for_a_known_goal(self):
        model = true_model(samples=20_000)
        goal = np.zeros(2500)
        goal[40 * 50 + 30] = 1.0
        model.belief = lambda history, grid_size: (goal, 1)  # 目的地已知为 (40, 30)、朝东
        history = [(0, (19, 20)), (1, (20, 20))]
        samples = model.sample_positions(history, 50, (4,))[0]
        exact = exact_known_goal((20, 20), 1, (40, 30), 4)
        sampled = np.zeros((50, 50))
        np.add.at(sampled, (samples[:, 0], samples[:, 1]), 1 / len(samples))
        self.assertLess(0.5 * np.abs(sampled - exact).sum(), 0.03)

    def test_prediction_is_repeatable_and_disk_point_uses_sample_histogram(self):
        route, _ = generate_vehicle_trajectory(RANDOM, np.random.default_rng(8))
        history = report_history(route, 30, 0)
        model = true_model(point="disk")
        first = model.predict_positions(history, 50, (0, 9))
        np.testing.assert_array_equal(first, model.predict_positions(history, 50, (0, 9)))
        np.testing.assert_array_equal(first[0], route[30])
        samples = model.sample_positions(history, 50, (9,))[0]
        counts = np.zeros((50, 50))
        np.add.at(counts, (samples[:, 0], samples[:, 1]), 1)
        np.testing.assert_array_equal(first[1], np.unravel_index(disk_sum(counts, 2.0).argmax(), counts.shape))
        restored = GoalFilterPredictor.from_state(model.export_state())
        np.testing.assert_array_equal(restored.predict_positions(history, 50, (0, 9)), first)

    def test_new_goals_respect_minimum_distance(self):
        model = true_model()
        cells = np.indices((50, 50)).reshape(2, -1).T
        origins = np.repeat([[25, 25], [0, 0]], 500, axis=0)
        goals = model._draw_new_goals(origins, cells, np.random.default_rng(0))
        self.assertTrue((np.abs(goals - origins).sum(axis=1) >= 20).all())


class TrainingTests(unittest.TestCase):
    def test_training_entrypoint_fits_and_checkpoint_keeps_goal_filter(self):
        env_config = EnvConfig(car_route="random", max_steps=98, observation_delay=2)
        config = TrainConfig(episodes=1, eval_episodes=1, prediction_routes=10, prediction_seed=50_000,
                             prediction_model_type="goal_filter", prediction_point="disk", prediction_steps=(1,),
                             hidden_sizes=(8,))
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(StringIO()):
            agent = train(env_config, config, folder)
            metrics = json.loads((Path(folder) / "prediction_metrics.json").read_text())
            loaded, _ = DQNAgent.load(Path(folder) / "model.pt")
        self.assertEqual(metrics["model_type"], "goal_filter")
        self.assertIsInstance(loaded.motion_predictor, GoalFilterPredictor)
        self.assertEqual((agent.motion_predictor.point, loaded.motion_predictor.point), ("disk", "disk"))
        np.testing.assert_allclose(loaded.motion_predictor.turn_log_weights, agent.motion_predictor.turn_log_weights)


if __name__ == "__main__":
    unittest.main()
