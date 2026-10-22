"""验证学得的车辆方向模型不在推理时读取未来真值。"""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from uav_demo.config import EnvConfig
from uav_demo.prediction import MotionPredictor, make_features


class MotionPredictionTests(unittest.TestCase):
    def uniform_predictor(self):
        model = MotionPredictor()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        return model

    def test_feature_inputs_are_observed_state_only(self):
        features = make_features((2, 3), (0, 1), (0, 4), 5)
        np.testing.assert_allclose(features, [0.5, 0.75, 0, 1, 0, 1, -0.5, 0.25])
        self.assertEqual(features.dtype, np.float32)

    def test_probability_mean_does_not_choose_argmax_route(self):
        model = self.uniform_predictor()
        # 四方向等概率，内部位置的期望位移为零；argmax 则会一直向北。
        result = model.predict_positions((2, 2), (1, 0), (4, 4), 5, steps=(1, 2, 4))
        np.testing.assert_allclose(result, [[2, 2], [2, 2], [2, 2]])
        # 左下角只允许北/东，边界 mask 后均值应是 (0.5, 0.5)。
        result = model.predict_positions((0, 0), (0, 0), (4, 4), 5, steps=(1,))
        np.testing.assert_allclose(result, [[0.5, 0.5]])

    def test_clock_goal_bounds_and_state_dict_roundtrip(self):
        model = self.uniform_predictor()
        with torch.no_grad():
            model.layers[-1].bias.copy_(torch.tensor([0., 20., 0., 0.]))
        result = model.predict_positions((1, 2), (1, 0), (4, 4), 5,
                                         steps=(0, 1, 2, 3), move_every=2, time=0)
        np.testing.assert_allclose(result, [[1, 2], [1, 2], [2, 2], [2, 2]], atol=1e-5)
        shifted = model.predict_positions((1, 2), (1, 0), (4, 4), 5,
                                          steps=(1,), move_every=2, time=1)
        np.testing.assert_allclose(shifted, [[2, 2]], atol=1e-5)
        fixed = model.predict_positions((4, 4), (1, 0), (4, 4), 5)
        np.testing.assert_allclose(fixed, [[4, 4]] * 3)
        many = model.predict_positions((4, 0), (1, 0), (0, 4), 5, steps=(30,))
        self.assertTrue(((many >= 0) & (many <= 4)).all())
        restored = MotionPredictor()
        restored.load_state_dict(model.state_dict())
        np.testing.assert_allclose(
            restored.predict_positions((1, 2), (1, 0), (4, 4), 5),
            model.predict_positions((1, 2), (1, 0), (4, 4), 5),
        )
        restored = MotionPredictor.from_state(model.export_state())
        np.testing.assert_allclose(
            restored.predict_positions((1, 2), (1, 0), (4, 4), 5),
            model.predict_positions((1, 2), (1, 0), (4, 4), 5),
        )
        invalid = model.export_state()
        invalid['directions'].reverse()
        with self.assertRaises(ValueError):
            MotionPredictor.from_state(invalid)

    def test_fit_only_uses_supplied_seeds_and_predict_never_reads_route(self):
        from uav_demo.vehicle import generate_vehicle_trajectory

        config = EnvConfig(grid_size=5, max_steps=20, uav_goal=(4, 4))
        with patch('uav_demo.prediction.generate_vehicle_trajectory', wraps=generate_vehicle_trajectory) as generator:
            model, metrics = MotionPredictor.fit(config, range(10, 20), epochs=2)
            self.assertEqual(generator.call_count, 10)
        self.assertEqual(metrics['fit_seeds'], list(range(10, 18)))
        self.assertEqual(metrics['holdout_seeds'], [18, 19])
        self.assertGreater(metrics['fit_moves'], 0)
        self.assertGreater(metrics['holdout_moves'], 0)
        self.assertTrue(np.isfinite(metrics['holdout_cross_entropy']))
        self.assertTrue(0 <= metrics['holdout_accuracy'] <= 1)
        with patch('uav_demo.prediction.generate_vehicle_trajectory', side_effect=AssertionError('future leak')):
            forecast = model.predict_positions((2, 1), (0, 1), (0, 4), 5)
        self.assertEqual(forecast.shape, (3, 2))
        self.assertTrue(np.isfinite(forecast).all())

    def test_invalid_training_seed_split_rejected(self):
        for seeds in ([], [1], [1, 1], [-1, 2]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                MotionPredictor.fit(EnvConfig(), seeds)


if __name__ == '__main__':
    unittest.main()
