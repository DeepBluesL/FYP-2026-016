"""方向概率 LSTM 预测器：输入编码、抽样分布与精确枚举比对、地图边界、最可能格、拟合、接入训练与保存加载。"""

import itertools
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import torch

from train import train
from uav_demo.agent import DQNAgent
from uav_demo.config import EnvConfig, TrainConfig
from uav_demo.direction_prediction import (DirectionPredictor, encode_positions, legal_directions,
                                           masked_log_softmax, route_examples)
from uav_demo.environment import GridWorldEnv
from uav_demo.grid_prediction import disk_sum
from uav_demo.observation import encode_observation
from uav_demo.prediction_data import split_route_seeds
from uav_demo.vehicle import DIRECTIONS, generate_vehicle_trajectory

RANDOM = EnvConfig(car_route="random")


def model_with_bias(bias, samples=2000):
    """固定随机种子的小模型；头部最后一层的偏置决定四个方向的大致偏好。"""
    torch.manual_seed(3)
    model = DirectionPredictor(hidden_size=16, head_size=8, samples=samples)
    with torch.no_grad():
        model.head[2].bias.copy_(torch.as_tensor(bias, dtype=torch.float32))
    return model


@torch.no_grad()
def exact_distribution(model, history, steps):
    """枚举 4^steps 条路径，用完整的 nn.LSTM 逐条算概率，得到 steps 步后的位置分布。"""
    positions = [tuple(p) for _, p in history]
    result = {}
    for path in itertools.product(range(4), repeat=steps):
        sequence, log_p = list(positions), 0.0
        for direction in path:
            logits = model(torch.as_tensor(encode_positions(sequence, 50)[None]))[0][0, -1]
            step = float(masked_log_softmax(logits, torch.as_tensor(legal_directions(sequence[-1], 50)))[direction])
            if step == -np.inf:
                break
            log_p += step
            sequence.append(tuple(int(v) for v in np.asarray(sequence[-1]) + DIRECTIONS[direction]))
        else:
            result[sequence[-1]] = result.get(sequence[-1], 0.0) + np.exp(log_p)
    return result


class EncodingTests(unittest.TestCase):
    def test_inputs_are_scaled_position_and_previous_move(self):
        inputs = encode_positions([(0, 0), (0, 1), (1, 1)], 50)
        np.testing.assert_allclose(inputs, [[0, 0, 0, 0], [0, 1 / 49, 0, 1], [1 / 49, 1 / 49, 1, 0]], atol=1e-7)

    def test_targets_are_actual_moves_and_always_legal(self):
        inputs, targets, legal = route_examples(RANDOM, [7, 8])
        for index, seed in enumerate((7, 8)):
            route, _ = generate_vehicle_trajectory(RANDOM, np.random.default_rng(seed))
            np.testing.assert_array_equal(DIRECTIONS[targets[index].numpy()], np.diff(route, axis=0))
            self.assertTrue(legal[index].gather(1, targets[index][:, None]).all())
            np.testing.assert_allclose(inputs[index, :, :2].numpy() * 49, route[:-1], atol=1e-4)


class SamplingTests(unittest.TestCase):
    def test_sampled_distribution_matches_exact_enumeration(self):
        model = model_with_bias((1.0, 0.0, -0.5, 0.3), samples=20_000)
        for history in ([(0, (20, 20)), (1, (20, 21)), (2, (21, 21))], [(0, (1, 48)), (1, (1, 49))]):
            exact = exact_distribution(model, history, 3)
            samples = model.sample_positions(history, 50, (3,))[0]
            cells, counts = np.unique(samples, axis=0, return_counts=True)
            sampled = {tuple(int(v) for v in cell): count / len(samples) for cell, count in zip(cells, counts)}
            self.assertLessEqual(set(sampled), set(exact))
            distance = 0.5 * sum(abs(exact[cell] - sampled.get(cell, 0.0)) for cell in exact)
            self.assertLess(distance, 0.03)

    def test_samples_stay_on_map_and_are_reachable(self):
        model = model_with_bias((2.0, 2.0, -2.0, -2.0))
        history = [(0, (48, 47)), (1, (48, 48)), (2, (49, 48))]
        for k, points in zip((1, 4, 9), model.sample_positions(history, 50, (1, 4, 9))):
            self.assertTrue(((points >= 0) & (points <= 49)).all())
            manhattan = np.abs(points - (49, 48)).sum(axis=1)
            self.assertTrue(((manhattan <= k) & ((k - manhattan) % 2 == 0)).all())

    def test_prediction_is_repeatable_and_step_zero_is_report(self):
        model = model_with_bias((0.0, 0.0, 0.0, 0.0))
        history = [(0, (10, 10)), (1, (11, 10))]
        first = model.predict_positions(history, 50, (0, 1, 9))
        np.testing.assert_array_equal(first, model.predict_positions(history, 50, (0, 1, 9)))
        np.testing.assert_array_equal(first[0], (11, 10))

    def test_point_prediction_is_most_likely_cell(self):
        model = model_with_bias((0.0, 8.0, 0.0, 0.0))  # 几乎总是向东
        np.testing.assert_array_equal(model.predict_positions([(0, (10, 20))], 50, (9,)), [[19, 20]])
        samples = model.sample_positions([(0, (10, 20))], 50, (9,))[0]
        cells, counts = np.unique(samples, axis=0, return_counts=True)
        np.testing.assert_array_equal(cells[counts.argmax()], (19, 20))

    @unittest.skipUnless(torch.cuda.is_available(), "需要 GPU")
    def test_cuda_graph_replay_matches_direct_rollout(self):
        model = model_with_bias((1.0, 0.0, -0.5, 0.3)).to("cuda")
        histories = ([(0, (20, 20)), (1, (20, 21))], [(0, (1, 48)), (1, (1, 49))], [(0, (30, 5))])
        for history in histories + histories:  # 交替调用，确认固定输入张量每次都被改写
            with torch.no_grad():
                inputs = torch.as_tensor(encode_positions([p for _, p in history], 50)[None], device="cuda")
                logits, (h, c) = model(inputs)
                report = history[-1][1][0] * 50 + history[-1][1][1]
                direct = model._rollout(50, (1, 4, 9), torch.tensor(report, device="cuda"),
                                        logits[0, -1], h[0, 0], c[0, 0]).cpu().numpy()
            np.testing.assert_array_equal(model._sample_cells(history, 50, (1, 4, 9)), direct)
        self.assertEqual(len(model._graphs), 1)

    def test_disk_point_maximizes_sampled_share_within_radius(self):
        model = model_with_bias((1.0, 0.0, -0.5, 0.3))
        history = [(0, (20, 20)), (1, (20, 21)), (2, (21, 21))]
        samples = model.sample_positions(history, 50, (9,))[0]
        counts = np.zeros((50, 50))
        np.add.at(counts, (samples[:, 0], samples[:, 1]), 1)
        expected = np.unravel_index(disk_sum(counts, 2.0).argmax(), counts.shape)
        model.point = "disk"
        np.testing.assert_array_equal(model.predict_positions(history, 50, (9,)), [expected])
        restored = DirectionPredictor.from_state(model.export_state(), device="cpu")
        self.assertEqual((restored.point, restored.radius), ("disk", 2.0))
        with self.assertRaisesRegex(ValueError, "point"):
            DirectionPredictor(point="mean")

    def test_only_random_routes_with_one_move_per_step(self):
        model = model_with_bias((0.0, 0.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "每步移动一格"):
            model.predict_positions([(0, (10, 10))], 50, (1,), move_every=2)
        groups = split_route_seeds(10, start_seed=0)
        for config in (EnvConfig(), EnvConfig(car_route="random", car_move_every=2, max_steps=200)):
            with self.assertRaisesRegex(ValueError, "car_route=random"):
                DirectionPredictor.fit(config, groups, epochs=1, device="cpu")


class TrainingTests(unittest.TestCase):
    def test_fit_beats_uniform_directions_and_keeps_best_epoch(self):
        groups = split_route_seeds(40, start_seed=60_000)
        with redirect_stdout(StringIO()):
            model, metrics = DirectionPredictor.fit(RANDOM, groups, epochs=4, batch_size=4, learning_rate=0.01,
                                                    device="cpu", hidden_size=16, head_size=8)
        self.assertGreater(metrics["best_validation_log_likelihood"], np.log(0.25))
        self.assertEqual(metrics["best_validation_log_likelihood"], max(metrics["validation_log_likelihood_by_epoch"]))
        self.assertAlmostEqual(model.log_likelihood(RANDOM, groups["validation"]),
                               metrics["best_validation_log_likelihood"], places=5)

    def test_checkpoint_round_trip_keeps_direction_predictor(self):
        env_config = EnvConfig(car_route="random", observation_delay=4)
        env = GridWorldEnv(env_config)
        config = TrainConfig(prediction_model_type="direction_lstm", prediction_steps=(1,), prediction_device="cpu")
        size = encode_observation(env.reset(seed=0)[0], env_config, config.observation_features).size
        agent = DQNAgent(size, env.action_count, config, env_config=env_config,
                         motion_predictor=model_with_bias((0.5, 0.0, 0.0, 0.0), samples=500))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            agent.save(path, env_config)
            loaded, _ = DQNAgent.load(path)
        self.assertIsInstance(loaded.motion_predictor, DirectionPredictor)
        self.assertEqual(loaded.motion_predictor.samples, 500)
        observation, _ = env.reset(seed=0)
        for _ in range(10):
            observation, *_ = env.step(0)
        np.testing.assert_array_equal(agent.predict_car(observation), loaded.predict_car(observation))

    def test_training_entrypoint_fits_direction_lstm(self):
        env_config = EnvConfig(car_route="random", max_steps=98, observation_delay=2)
        config = TrainConfig(episodes=1, eval_episodes=1, prediction_routes=10, prediction_seed=50_000,
                             prediction_model_type="direction_lstm", prediction_epochs=1, prediction_steps=(1,),
                             prediction_hidden_size=8, prediction_head_size=4, prediction_device="cpu",
                             hidden_sizes=(8,))
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(StringIO()):
            agent = train(env_config, config, folder)
            metrics = json.loads((Path(folder) / "prediction_metrics.json").read_text())
        self.assertEqual(metrics["model_type"], "direction_lstm")
        self.assertIsInstance(agent.motion_predictor, DirectionPredictor)
        self.assertEqual(sorted(sum(metrics["seed_groups"].values(), [])), list(range(50_000, 50_010)))


if __name__ == "__main__":
    unittest.main()
