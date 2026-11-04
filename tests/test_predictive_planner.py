"""当前状态预测拦截的等待、阶段与边界行为。"""

import unittest
from unittest.mock import Mock

import numpy as np

from uav_demo.config import EnvConfig
from uav_demo.environment import ACTION_NAMES, GridWorldEnv
from uav_demo.predictive_planner import PredictivePlanner


class StraightPredictor:
    """测试模型仅按已观测速度外推，不持有环境或真实轨迹。"""

    def predict_positions(self, position, velocity, grid_size, steps, move_every=1, time=0):
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
        env.car_observation_history = [(env.car_pos, env.car_velocity)]
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
                self.assertIsNone(planner.last_predictions)
                self.assertIsNone(planner.target_time)

    def test_every_chosen_action_obeys_boundary_mask_with_slow_car(self):
        config = EnvConfig(grid_size=5, max_steps=30, uav_start=(4, 0), uav_goal=(4, 4),
                           car_move_every=2, observation_delay=3)
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

    def test_waits_without_predicting_until_report_arrives(self):
        env = GridWorldEnv(EnvConfig(observation_delay=3))
        observation, _ = env.reset(seed=42)
        predictor = Mock(wraps=StraightPredictor())
        planner = PredictivePlanner(predictor, env.config)
        for _ in range(3):
            self.assertEqual(planner.choose_action(observation, env.valid_action_mask()), 0)
            predictor.predict_positions.assert_not_called()
            self.assertIsNone(planner.target)
            self.assertIsNone(planner.last_predictions)
            observation, *_ = env.step(0)
        planner.choose_action(observation, env.valid_action_mask())
        self.assertEqual(planner.last_predictions.shape,
                         (planner.horizon + env.config.sensing_steps, 2))
        predictor.predict_positions.assert_called_once()
        self.assertEqual(predictor.predict_positions.call_args.kwargs["time"], 0)
        self.assertEqual(predictor.predict_positions.call_args.kwargs["steps"][0], 3)


if __name__ == '__main__':
    unittest.main()
