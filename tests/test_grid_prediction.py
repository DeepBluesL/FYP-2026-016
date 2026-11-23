"""网格概率推演预测器：手算例子、概率图性质、与车辆生成器的大量随机路线比对、数转向概率、接入训练与保存加载。"""

import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from compare_predictors import report_history
from train import train
from uav_demo.agent import DQNAgent
from uav_demo.config import EnvConfig, TrainConfig
from uav_demo.environment import GridWorldEnv
from uav_demo.grid_prediction import (GridPredictor, disk_sum, expected_positions, last_heading,
                                      most_likely_positions, point_positions)
from uav_demo.observation import encode_observation
from uav_demo.prediction import forecast_vehicle
from uav_demo.prediction_data import split_route_seeds
from uav_demo.vehicle import DIRECTIONS, generate_vehicle_trajectory

RANDOM = EnvConfig(car_route="random")


def calibration(config, model, starts, steps, routes):
    """对每条路线，从 starts 中各时刻的真实车位与朝向出发推演 steps 步，与实际走到的位置比较。

    返回每个起点的（推演均值位移, 实际均值位移, 汇总后位移分布的总变差距离）。推演分布若正确，
    多条路线上推演分布的平均就等于实际位移的分布。
    """
    span = 2 * steps + 1
    results = []
    predicted_mean, observed_mean = np.zeros((len(starts), 2)), np.zeros((len(starts), 2))
    predicted_hist, observed_hist = np.zeros((len(starts), span, span)), np.zeros((len(starts), span, span))
    for seed in range(routes):
        route, _, goals = generate_vehicle_trajectory(config, np.random.default_rng(seed), with_goals=True)
        for index, start in enumerate(starts):
            history = report_history(route, start, 0)
            probability = model.distributions(history, config.grid_size, (steps,), goals=goals)[0]
            x, y = (int(v) for v in route[start])
            window = np.zeros((config.grid_size + 2 * steps, config.grid_size + 2 * steps))
            window[steps:steps + config.grid_size, steps:steps + config.grid_size] = probability
            predicted_hist[index] += window[x:x + span, y:y + span]
            predicted_mean[index] += expected_positions(probability) - (x, y)
            dx, dy = route[start + steps] - (x, y)
            observed_hist[index, dx + steps, dy + steps] += 1
            observed_mean[index] += (dx, dy)
    for index in range(len(starts)):
        distance = 0.5 * np.abs(predicted_hist[index] - observed_hist[index]).sum() / routes
        results.append((predicted_mean[index] / routes, observed_mean[index] / routes, distance))
    return results


class HeadingTests(unittest.TestCase):
    def test_last_observed_move_and_unknown_at_first_report(self):
        route = np.asarray(((5, 5), (6, 5), (6, 6)))
        self.assertIsNone(last_heading(report_history(route, 0, 0)))
        self.assertEqual(last_heading(report_history(route, 1, 0)), 1)  # 东
        self.assertEqual(last_heading(report_history(route, 2, 0)), 0)  # 北

    def test_pauses_between_moves_keep_last_heading(self):
        history = ((0, (5, 5)), (1, (5, 5)), (2, (6, 5)), (3, (6, 5)))
        self.assertEqual(last_heading(history, move_every=2), 1)


class HandExampleTests(unittest.TestCase):
    def test_one_step_inside_map_uses_base_probabilities(self):
        history = ((0, (10, 9)), (1, (10, 10)))  # 朝北：直行 0.65、左右各 0.15、掉头 0.05
        maps = GridPredictor.base(RANDOM).distributions(history, 50, (1,))
        np.testing.assert_allclose(expected_positions(maps)[0], (10.0, 10.6))
        np.testing.assert_array_equal(most_likely_positions(maps)[0], (10, 11))

    def test_one_step_at_corner_renormalizes_legal_directions(self):
        history = ((0, (0, 48)), (1, (0, 49)))  # 左上角朝北：只能右转向东或掉头向南
        probability = GridPredictor.base(RANDOM).distributions(history, 50, (1,))[0]
        self.assertAlmostEqual(probability[1, 49], 0.15 / 0.20)
        self.assertAlmostEqual(probability[0, 48], 0.05 / 0.20)

    def test_goal_bias_weights_moves_toward_goal(self):
        # 朝北、目的地在东北：北、东各靠近 1 格，乘 e；南、西各远离 1 格，除以 e。
        history = ((0, (10, 9)), (1, (10, 10)))
        goals = np.tile((20, 30), (10, 1))
        probability = GridPredictor.with_goals(RANDOM).distributions(history, 50, (1,), goals=goals)[0]
        weights = {(10, 11): 0.65 * math.e, (11, 10): 0.15 * math.e, (9, 10): 0.15 / math.e, (10, 9): 0.05 / math.e}
        total = sum(weights.values())
        for cell, weight in weights.items():
            self.assertAlmostEqual(probability[cell], weight / total)


class DistributionTests(unittest.TestCase):
    def test_maps_are_probabilities_and_step_zero_is_report_cell(self):
        history = ((0, (3, 3)), (1, (4, 3)), (2, (5, 3)))
        maps = GridPredictor.base(RANDOM).distributions(history, 50, (0, 1, 5, 20))
        np.testing.assert_allclose(maps.sum(axis=(1, 2)), 1.0)
        self.assertEqual(maps[0, 5, 3], 1.0)

    def test_car_only_moves_on_its_clock(self):
        config = EnvConfig(car_route="random", car_move_every=2, max_steps=200)
        history = ((0, (5, 5)), (1, (5, 5)), (2, (6, 5)), (3, (6, 5)))
        maps = GridPredictor.base(config).distributions(history, 50, (1, 2, 3), move_every=2)
        np.testing.assert_array_equal(maps[0], maps[1])  # 时刻 4 移动，时刻 5 不动
        self.assertFalse(np.array_equal(maps[1], maps[2]))

    def test_point_prediction_is_most_likely_cell(self):
        model = GridPredictor.base(RANDOM)
        history = ((0, (2, 40)), (1, (2, 41)), (2, (2, 42)))
        steps = (1, 4, 9)
        np.testing.assert_array_equal(model.predict_positions(history, 50, steps),
                                      most_likely_positions(model.distributions(history, 50, steps)))

    def test_goal_source_requires_true_goals(self):
        with self.assertRaisesRegex(ValueError, "真实目的地"):
            GridPredictor.with_goals(RANDOM).predict_positions(((0, (5, 5)),), 50, (1,))

    def test_fixed_route_is_not_modeled(self):
        with self.assertRaisesRegex(ValueError, "随机路线"):
            GridPredictor.base(EnvConfig(car_route="fixed"))


class SimulationTests(unittest.TestCase):
    """与车辆生成器的随机路线比对：起点 0 时朝向未知（首报），起点 3 时朝向已知。"""

    def test_base_probabilities_match_generator_without_goal_bias(self):
        config = EnvConfig(car_route="random", car_goal_bias=0.0, max_steps=98)
        for predicted, observed, distance in calibration(config, GridPredictor.base(config), (0, 3), 4, 1000):
            np.testing.assert_allclose(predicted, observed, atol=0.2)
            self.assertLess(distance, 0.1)

    def test_true_goals_match_generator_with_goal_bias(self):
        config = EnvConfig(car_route="random", max_steps=98)
        for predicted, observed, distance in calibration(config, GridPredictor.with_goals(config), (0, 3), 4, 1000):
            np.testing.assert_allclose(predicted, observed, atol=0.2)
            self.assertLess(distance, 0.1)

    def test_counted_probabilities_recover_base_values_without_goal_bias(self):
        config = EnvConfig(car_route="random", car_goal_bias=0.0, max_steps=98)
        model, metrics = GridPredictor.fit(config, split_route_seeds(100, start_seed=0))
        np.testing.assert_allclose(model.turn_probabilities, (0.65, 0.15, 0.05, 0.15), atol=0.015)
        self.assertGreater(metrics["counted_moves"], 5000)

    def test_counted_probabilities_with_goal_bias_go_straight_more(self):
        model, metrics = GridPredictor.fit(EnvConfig(car_route="random", max_steps=98),
                                           split_route_seeds(50, start_seed=0))
        self.assertGreater(model.turn_probabilities[0], 0.7)
        self.assertGreater(metrics["validation_log_likelihood"], metrics["base_validation_log_likelihood"])


class GoalSequenceTests(unittest.TestCase):
    def test_recording_goals_leaves_route_unchanged(self):
        for route_type in ("random", "fixed"):
            config = EnvConfig(car_route=route_type)
            route, goal = generate_vehicle_trajectory(config, np.random.default_rng(7))
            again, same_goal, goals = generate_vehicle_trajectory(config, np.random.default_rng(7), with_goals=True)
            np.testing.assert_array_equal(route, again)
            self.assertEqual(tuple(goals[0]), tuple(goal))
            self.assertEqual(goal, same_goal)

    def test_goal_changes_exactly_after_arrival(self):
        for seed in range(20):
            route, _, goals = generate_vehicle_trajectory(RANDOM, np.random.default_rng(seed), with_goals=True)
            for time in range(1, len(route)):
                arrived = np.array_equal(route[time - 1], goals[time - 1])
                self.assertEqual(arrived, not np.array_equal(goals[time], goals[time - 1]))

    def test_environment_exposes_goals_only_for_diagnostics(self):
        hidden, _ = GridWorldEnv(RANDOM).reset(seed=1)
        self.assertNotIn("true_car_goals", hidden)
        env = GridWorldEnv(RANDOM, expose_truth=True)
        shown, _ = env.reset(seed=1)
        np.testing.assert_array_equal(shown["true_car_goals"], env.car_goals)


class IntegrationTests(unittest.TestCase):
    def test_only_goal_source_needs_truth(self):
        for turns, needs in (("counted", False), ("base", False), ("goal", True)):
            config = TrainConfig(prediction_model_type="grid", prediction_grid_turns=turns)
            self.assertEqual(config.needs_true_trajectory, needs)
        with self.assertRaisesRegex(ValueError, "prediction_grid_turns"):
            TrainConfig(prediction_grid_turns="learned")

    def test_forecast_passes_true_goals_from_observation(self):
        config = EnvConfig(car_route="random", observation_delay=3)
        model = GridPredictor.with_goals(config)
        env = GridWorldEnv(config, expose_truth=True)
        observation, _ = env.reset(seed=4)
        for _ in range(10):
            observation, *_ = env.step(0)
        expected = model.predict_positions(observation["car_history"], 50, (4,), goals=env.car_goals)
        np.testing.assert_array_equal(forecast_vehicle(model, observation, config, (1,)), expected)
        del observation["true_car_goals"]
        with self.assertRaisesRegex(ValueError, "expose_truth"):
            forecast_vehicle(model, observation, config, (1,))

    def test_checkpoint_round_trip_keeps_grid_predictor(self):
        env_config = EnvConfig(car_route="random", observation_delay=4)
        env = GridWorldEnv(env_config)
        config = TrainConfig(prediction_model_type="grid")
        size = encode_observation(env.reset(seed=0)[0], env_config, config.observation_features).size
        agent = DQNAgent(size, env.action_count, config, env_config=env_config,
                         motion_predictor=GridPredictor("counted", (0.74, 0.11, 0.03, 0.12)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            agent.save(path, env_config)
            loaded, _ = DQNAgent.load(path)
        self.assertIsInstance(loaded.motion_predictor, GridPredictor)
        np.testing.assert_allclose(loaded.motion_predictor.turn_probabilities, (0.74, 0.11, 0.03, 0.12))
        observation, _ = env.reset(seed=0)
        for _ in range(10):
            observation, *_ = env.step(0)
        np.testing.assert_array_equal(agent.predict_car(observation), loaded.predict_car(observation))

    def test_disk_point_maximizes_probability_within_radius(self):
        probability = np.zeros((50, 50))
        probability[10, 10] = 0.3  # 单格最可能
        for cell in ((30, 30), (31, 30), (30, 31), (29, 30), (30, 29)):
            probability[cell] = 0.14  # 一簇加起来更多
        np.testing.assert_array_equal(point_positions([probability], "mode"), [[10, 10]])
        # (29, 30)、(30, 29)、(30, 30) 等都能把整簇圈进 2 格内，并列取下标最小的 (29, 30)。
        np.testing.assert_array_equal(point_positions([probability], "disk", 2.0), [[29, 30]])
        self.assertAlmostEqual(disk_sum(probability, 2.0)[29, 30], 0.7)
        with self.assertRaisesRegex(ValueError, "point"):
            point_positions([probability], "mean")

    def test_disk_point_prediction_and_checkpoint(self):
        model = GridPredictor("counted", (0.74, 0.11, 0.03, 0.12), point="disk")
        history = [(0, (20, 20)), (1, (21, 20)), (2, (22, 20))]
        maps = model.distributions(history, 50, (9,))
        np.testing.assert_array_equal(model.predict_positions(history, 50, (9,)), point_positions(maps, "disk", 2.0))
        restored = GridPredictor.from_state(model.export_state())
        self.assertEqual((restored.point, restored.radius), ("disk", 2.0))
        self.assertEqual(GridPredictor.from_state({"turns": "base", "turn_probabilities": (0.65, 0.15, 0.05, 0.15)}).point,
                         "mode")
        with self.assertRaisesRegex(ValueError, "prediction_point"):
            TrainConfig(prediction_model_type="linear", prediction_point="disk")
        config = TrainConfig(episodes=1, eval_episodes=1, prediction_routes=10, prediction_seed=50_000,
                             prediction_model_type="grid", prediction_point="disk", hidden_sizes=(8,))
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(StringIO()):
            agent = train(EnvConfig(car_route="random", max_steps=98), config, folder)
            loaded, _ = DQNAgent.load(Path(folder) / "model.pt")
        self.assertEqual(agent.motion_predictor.point, "disk")
        self.assertEqual(loaded.motion_predictor.point, "disk")

    def test_training_entrypoint_counts_turns_and_saves_metrics(self):
        env_config = EnvConfig(car_route="random", max_steps=98)
        config = TrainConfig(episodes=1, eval_episodes=1, prediction_routes=10, prediction_seed=50_000,
                             prediction_model_type="grid", hidden_sizes=(8,))
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(StringIO()):
            agent = train(env_config, config, folder)
            metrics = json.loads((Path(folder) / "prediction_metrics.json").read_text())
        self.assertEqual(metrics["turns"], "counted")
        self.assertIsInstance(agent.motion_predictor, GridPredictor)
        self.assertEqual(sorted(sum(metrics["seed_groups"].values(), [])), list(range(50_000, 50_010)))


if __name__ == "__main__":
    unittest.main()
