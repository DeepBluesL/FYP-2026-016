import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from uav_demo.agent import DQNAgent
from uav_demo.config import EnvConfig, TrainConfig
from uav_demo.environment import GridWorldEnv
from uav_demo.observation import encode_observation
from uav_demo.sequence_prediction import SequencePredictor, fit_sequence_predictor, run_epoch, prediction_device


class SequencePredictionTests(unittest.TestCase):
    def test_all_models_are_causal_and_survive_agent_roundtrip(self):
        torch.set_num_threads(1)
        env = GridWorldEnv(EnvConfig(grid_size=5, uav_goal=(4, 4), max_steps=12))
        observation, _ = env.reset(seed=7)
        for kind in ("rnn", "lstm", "gru"):
            with self.subTest(kind=kind):
                model = SequencePredictor(kind, hidden_size=8, max_steps=12)
                x = torch.rand(1, 13, 3)
                first, _ = model(x)
                changed = x.clone()
                changed[:, 6:] = 0
                second, _ = model(changed)
                self.assertEqual(first.shape, (1, 13, 2))
                torch.testing.assert_close(first[:, :6], second[:, :6])
                before = {k: v.clone() for k, v in model.state_dict().items()}
                self.assertTrue(np.isfinite(run_epoch(model, env.config, [7])))
                for k, v in model.state_dict().items():
                    torch.testing.assert_close(v, before[k])
                forecast = model.predict_positions(((0, (2, 2)),), 5, (3, 0, 1))
                self.assertEqual(forecast.shape, (3, 2))
                np.testing.assert_array_equal(forecast[1], (2, 2))
                agent = DQNAgent(encode_observation(observation, env.config).size,
                                 env.action_count, TrainConfig(hidden_sizes=(8,), prediction_device="cpu"),
                                 motion_predictor=model, env_config=env.config)
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "agent.pt"
                    agent.save(path, env.config)
                    restored, _ = DQNAgent.load(path)
                self.assertEqual(restored.motion_predictor.model_type, kind)
                np.testing.assert_allclose(agent._features(observation), restored._features(observation))

    def test_fit_runs_with_position_labels(self):
        torch.set_num_threads(1)
        config = EnvConfig(grid_size=5, uav_goal=(4, 4), max_steps=12)
        model, metrics = fit_sequence_predictor(
            config, {"train": [1, 2], "validation": [3], "test": [4]}, epochs=1, device="cpu", hidden_size=8, head_size=4,
        )
        self.assertFalse(model.training)
        self.assertTrue(np.isfinite(metrics["best_validation_mse"]))

    def test_batches_keep_whole_routes_and_weight_partial_batch_correctly(self):
        torch.set_num_threads(1)
        config = EnvConfig(grid_size=5, uav_goal=(4, 4), max_steps=12)
        model = SequencePredictor(hidden_size=8, head_size=4)
        seen = []
        handle = model.register_forward_pre_hook(lambda module, args: seen.append(tuple(args[0].shape)))
        batched = run_epoch(model, config, [1, 2, 3, 4, 5], batch_size=2)
        handle.remove()
        self.assertEqual(seen, [(2, 13, 3), (2, 13, 3), (1, 13, 3)])
        single = run_epoch(model, config, [1, 2, 3, 4, 5], batch_size=1)
        self.assertAlmostEqual(batched, single, places=6)

    def test_auto_device_prefers_cuda_and_falls_back_to_cpu(self):
        for available, expected in ((True, "cuda"), (False, "cpu")):
            with patch("torch.cuda.is_available", return_value=available):
                self.assertEqual(prediction_device().type, expected)
        model = SequencePredictor()
        self.assertEqual(model.encoder.hidden_size, 256)
        self.assertEqual(model.position_head[0].out_features, 128)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_training_forecast_and_cpu_restore(self):
        config = EnvConfig(grid_size=5, uav_goal=(4, 4), max_steps=12)
        model, _ = fit_sequence_predictor(config, {"train": [1, 2], "validation": [3]},
                                          epochs=1, hidden_size=8, head_size=4)
        self.assertEqual(next(model.parameters()).device.type, "cuda")
        restored = SequencePredictor.from_state(model.export_state(), device="cpu")
        history = ((0, (2, 2)), (1, (2, 3)))
        np.testing.assert_allclose(model.predict_positions(history, 5, (0, 1, 3)),
                                   restored.predict_positions(history, 5, (0, 1, 3)), atol=1e-5)
