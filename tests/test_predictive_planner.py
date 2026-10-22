"""当前状态预测拦截的等待、阶段与边界行为。"""

import unittest

import numpy as np

from uav_demo.config import EnvConfig
from uav_demo.environment import ACTION_NAMES, GridWorldEnv
from uav_demo.predictive_planner import PredictivePlanner


class StraightPredictor:
    """测试模型仅按已观测速度外推，不持有环境或真实轨迹。"""

    def predict_positions(self, position, velocity, goal, grid_size, steps, move_every=1, time=0):
        moves = (time + np.asarray(steps)) // move_every - time // move_every
        return np.clip(np.asarray(position) + moves[:, None] * np.asarray(velocity), 0, grid_size - 1)


class PredictivePlannerTests(unittest.TestCase):
    def test_same_current_state_different_future_produces_same_action(self):
        config = EnvConfig()
        env = GridWorldEnv(config)
        observation, _ = env.reset(seed=42)
        first, second = PredictivePlanner(StraightPredictor(), config), PredictivePlanner(StraightPredictor(), config)
        expected = first.choose_action(observation, env.valid_action_mask())
        env.car_trajectory[1:] = (0, 0)
        self.assertEqual(second.choose_action(env._observation(), env.valid_action_mask()), expected)
        self.assertEqual(first.target, second.target)

    def test_straight_car_can_approach_hovering_uav_and_complete_sensing(self):
        config = EnvConfig(grid_size=15, max_steps=60, uav_start=(5, 4), uav_goal=(14, 14),
                           sensing_steps=5, sensing_radius=2)
        env = GridWorldEnv(config)
        env.reset(seed=42)
        env.car_trajectory[:, 0] = np.minimum(np.arange(61), 14)
        env.car_trajectory[:, 1] = 5
        env.car_goal = (14, 5)
        env.car_velocity = (1, 0)
        planner = PredictivePlanner(StraightPredictor(), config)
        pre_contact_hovers, first_contact = 0, None
        while True:
            action = planner.choose_action(env._observation(), env.valid_action_mask())
            _, _, terminated, truncated, info = env.step(action)
            if first_contact is None:
                if info['contact']:
                    first_contact = info['time']
                elif action == 0:
                    pre_contact_hovers += 1
            if terminated or truncated:
                break
        self.assertGreaterEqual(pre_contact_hovers, 2)
        self.assertTrue(info['success'])

    def test_success_or_failure_navigates_to_uav_goal_and_reset_clears_target(self):
        for flag in ('sensing_complete', 'sensing_failed'):
            with self.subTest(flag=flag):
                config = EnvConfig(grid_size=5, max_steps=20, uav_goal=(4, 4))
                env = GridWorldEnv(config)
                env.reset(seed=1)
                setattr(env, flag, True)
                planner = PredictivePlanner(StraightPredictor(), config)
                action = planner.choose_action(env._observation(), env.valid_action_mask())
                self.assertEqual(ACTION_NAMES[action], 'north_east_2')
                self.assertEqual(planner.target, (4, 4))
                planner.reset()
                self.assertIsNone(planner.target)
                self.assertIsNone(planner.target_time)

    def test_every_chosen_action_obeys_boundary_mask_with_slow_car(self):
        config = EnvConfig(grid_size=5, max_steps=30, uav_start=(4, 0), uav_goal=(4, 4), car_move_every=2)
        env = GridWorldEnv(config)
        observation, _ = env.reset(seed=2)
        planner = PredictivePlanner(StraightPredictor(), config)
        while True:
            mask = env.valid_action_mask()
            action = planner.choose_action(observation, mask)
            self.assertTrue(mask[action])
            observation, _, terminated, truncated, info = env.step(action)
            self.assertFalse(info['invalid_move'])
            if terminated or truncated:
                break


if __name__ == '__main__':
    unittest.main()
