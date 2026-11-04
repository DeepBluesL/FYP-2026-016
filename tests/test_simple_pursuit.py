"""贪心策略只接收车辆报告，首报前等待，并验证对比脚本的调用接口。"""

import json
import unittest

from compare_pursuit import run_episode
from uav_demo.config import EnvConfig
from uav_demo.environment import ACTION_NAMES, GridWorldEnv
from uav_demo.simple_pursuit import choose_action


class SimplePursuitTests(unittest.TestCase):
    def test_waits_until_first_report_and_ignores_hidden_current_position(self):
        config = EnvConfig(grid_size=5, max_steps=20, uav_goal=(4, 4),
                           observation_delay=3, sensing_radius=0)
        env = GridWorldEnv(config)
        observation, _ = env.reset(seed=42)
        for _ in range(3):
            self.assertEqual(choose_action(observation, config, env.valid_action_mask()), 0)
            observation, *_ = env.step(0)
        # 首报是右下角起点，策略应向东，不能追此时被改成北侧的真实车位。
        env.car_trajectory[env.time] = (0, 4)
        action = choose_action(env._observation(), config, env.valid_action_mask())
        self.assertEqual(ACTION_NAMES[action], "east_2")
        self.assertIsInstance(action, int)

    def test_finished_sensing_can_navigate_without_vehicle_report(self):
        env = GridWorldEnv(EnvConfig(observation_delay=5))
        env.reset(seed=42)
        env.sensing_complete = True
        action = choose_action(env._observation(), env.config, env.valid_action_mask())
        self.assertEqual(ACTION_NAMES[action], "north_east_2")

    def test_comparison_episode_uses_new_interface_and_serializable_actions(self):
        config = EnvConfig(grid_size=5, max_steps=20, uav_goal=(4, 4), observation_delay=3)
        row, episode = run_episode(None, config, 3_000_000, "greedy_pursuit")
        json.dumps(episode)
        self.assertEqual([record["action"] for record in episode["records"][1:4]], [0, 0, 0])
        self.assertFalse(episode["records"][0]["has_observation"])
        self.assertTrue(episode["records"][3]["has_observation"])
        self.assertLessEqual(row["steps"], config.max_steps)


if __name__ == "__main__":
    unittest.main()
