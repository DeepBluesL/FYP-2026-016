"""离线对比脚本的计算部分：报告历史与环境一致、误差方向分量。网格推演本身的测试见 test_grid_prediction.py。"""

import math
import unittest

from compare_predictors import error_components, report_history
from uav_demo.config import EnvConfig
from uav_demo.environment import GridWorldEnv


class ReportHistoryTests(unittest.TestCase):
    def test_matches_environment_observation(self):
        config = EnvConfig(observation_delay=3, car_route="random")
        env = GridWorldEnv(config)
        observation, _ = env.reset(seed=5)
        for _ in range(30):
            if observation["has_observation"]:
                self.assertEqual(report_history(env.car_trajectory, env.time, 3), observation["car_history"])
            observation, *_ = env.step(0)


class ErrorComponentTests(unittest.TestCase):
    def test_hand_example(self):
        # 车头朝东，预测比真实少走 2 格、偏上 1 格；中心在报告位置的东北方。
        along, cross, center = error_components((5, 3), (7, 2), (0, 0), 1, (24.5, 24.5))
        self.assertAlmostEqual(along, -2.0)
        self.assertAlmostEqual(cross, 1.0)
        self.assertAlmostEqual(center, -1 / math.sqrt(2))

    def test_unknown_heading_keeps_only_center_part(self):
        along, cross, center = error_components((3, 0), (1, 0), (0, 0), None, (10, 0))
        self.assertIsNone(along)
        self.assertIsNone(cross)
        self.assertAlmostEqual(center, 2.0)


if __name__ == "__main__":
    unittest.main()
