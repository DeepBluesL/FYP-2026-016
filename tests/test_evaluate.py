"""验证评估指标包含初始状态。"""

import unittest

from evaluate import evaluate_episode
from uav_demo.config import EnvConfig
from uav_demo.environment import ACTION_NAMES, GridWorldEnv


class EvaluationTests(unittest.TestCase):
    def test_early_goal_includes_initial_position(self):
        class ScriptedAgent:
            def __init__(self):
                self.actions = iter([ACTION_NAMES.index("east_1")])

            def choose_action(self, observation, epsilon, mask):
                return next(self.actions, 0)

        for goal, expected in (((0, 0), True), ((4, 4), False)):
            with self.subTest(goal=goal):
                env = GridWorldEnv(EnvConfig(
                    grid_size=5, max_steps=8, uav_start=(0, 0),
                    uav_goal=goal, sensing_steps=5,
                ))
                result = evaluate_episode(ScriptedAgent(), env, seed=3_000_000)
                self.assertEqual(result["early_goal_before_sensing"], expected)


if __name__ == "__main__":
    unittest.main()
