"""真实轨迹默认不进入观测；只有真实位置诊断以 expose_truth=True 建环境时才附带。"""

import unittest

import numpy as np

from uav_demo.agent import DQNAgent
from uav_demo.config import EnvConfig, TrainConfig
from uav_demo.environment import GridWorldEnv
from uav_demo.observation import encode_observation

FEATURES = TrainConfig().observation_features + ("rule_action",)


class TrueTrajectoryTests(unittest.TestCase):
    def test_observation_hides_truth_unless_switched_on(self):
        config = EnvConfig(observation_delay=3)
        hidden, _ = GridWorldEnv(config).reset(seed=1)
        self.assertNotIn("true_car_trajectory", hidden)
        env = GridWorldEnv(config, expose_truth=True)
        shown, _ = env.reset(seed=1)
        np.testing.assert_array_equal(shown["true_car_trajectory"], env.car_trajectory)
        # 开关只多一项真值，策略能编码的特征完全相同。
        np.testing.assert_array_equal(encode_observation(hidden, config), encode_observation(shown, config))

    def test_only_true_position_diagnostics_need_truth(self):
        self.assertFalse(TrainConfig().needs_true_trajectory)
        self.assertFalse(TrainConfig(rule_source="predictor").needs_true_trajectory)
        self.assertTrue(TrainConfig(rule_source="oracle").needs_true_trajectory)
        self.assertTrue(TrainConfig(prediction_model_type="oracle").needs_true_trajectory)
        self.assertFalse(TrainConfig(prediction_model_type="oracle", use_motion_prediction=False).needs_true_trajectory)

    def test_true_position_rule_reports_missing_switch(self):
        env_config = EnvConfig(observation_delay=0)
        config = TrainConfig(rule_source="oracle", rule_prior=0.1, use_motion_prediction=False,
                             observation_features=FEATURES)
        for expose in (True, False):
            env = GridWorldEnv(env_config, expose_truth=expose)
            observation, _ = env.reset(seed=2)
            agent = DQNAgent(encode_observation(observation, env_config, FEATURES).size, env.action_count, config,
                             env_config=env_config)
            if expose:
                self.assertEqual(agent.rule_target(observation), tuple(int(v) for v in env.car_trajectory[1]))
            else:
                with self.assertRaisesRegex(ValueError, "expose_truth"):
                    agent.rule_target(observation)


if __name__ == "__main__":
    unittest.main()
