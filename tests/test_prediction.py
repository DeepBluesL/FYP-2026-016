"""预测器统一接口：报告历史与环境一致、运动推导、按延迟和车辆时钟对齐预测步，且推理不读取未来真值。"""

import unittest
from unittest.mock import Mock

import numpy as np

from uav_demo.config import EnvConfig
from uav_demo.environment import GridWorldEnv
from uav_demo.linear_prediction import LinearPredictor
from uav_demo.prediction import forecast_vehicle, observed_motion, report_history


class PredictionInterfaceTests(unittest.TestCase):
    def test_report_history_matches_environment_observation(self):
        config = EnvConfig(observation_delay=3, car_route="random")
        env = GridWorldEnv(config)
        observation, _ = env.reset(seed=5)
        for _ in range(30):
            if observation["has_observation"]:
                self.assertEqual(report_history(env.car_trajectory, env.time, 3), observation["car_history"])
            observation, *_ = env.step(0)

    def test_motion_uses_report_times_and_preserves_direction_through_stops(self):
        np.testing.assert_array_equal(observed_motion(((0, (2, 2)),), 2), (0, 0))
        history = ((0, (1, 2)), (4, (3, 2)), (5, (3, 2)))
        np.testing.assert_array_equal(observed_motion(history, 2), (1, 0))
        np.testing.assert_array_equal(observed_motion(history, 1), (0.5, 0))

    def test_no_forecast_before_first_report(self):
        env = GridWorldEnv(EnvConfig(observation_delay=4))
        observation, _ = env.reset(seed=42)
        predictor = Mock()
        self.assertIsNone(forecast_vehicle(
            predictor, observation, env.config, (0, 1, 4)
        ))
        predictor.predict_positions.assert_not_called()

    def test_forecast_aligns_delay_and_slow_vehicle_clock(self):
        config = EnvConfig(grid_size=5, max_steps=30, uav_goal=(4, 4),
                           car_move_every=2, observation_delay=3)
        # 当前 t=5，报告来自 t=2；车辆只在偶数时刻移动：到 t=5 移动一次（t=4），到 t=6、7 各移动两次（t=4、6）。
        state = {"has_observation": True, "car_pos": (1, 2), "car_history": ((0, (0, 2)), (2, (1, 2))),
                 "time": 5, "observation_age": 3}
        forecast = forecast_vehicle(LinearPredictor(), state, config, (0, 1, 2))
        np.testing.assert_allclose(forecast, [[2, 2], [3, 2], [3, 2]])


if __name__ == '__main__':
    unittest.main()
