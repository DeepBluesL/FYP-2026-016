"""外推贪心只用延迟报告：首报前悬停、按移动节拍外推、感知结束后去终点。"""

import unittest

from uav_demo.config import EnvConfig
from uav_demo.environment import ACTION_NAMES, GridWorldEnv
from uav_demo.extrapolate_pursuit import ExtrapolatePursuit, predicted_car
from uav_demo.simple_pursuit import choose_action as greedy_action


def report(car_pos, car_velocity, time, age):
    return {"car_pos": car_pos, "car_velocity": car_velocity, "time": time, "observation_age": age}


class ExtrapolatePursuitTests(unittest.TestCase):
    def test_extrapolates_to_next_step_counting_vehicle_move_phase(self):
        # 报告来自第 7 步，现在第 10 步；外推到第 11 步。
        every_step = EnvConfig(car_move_every=1)
        self.assertEqual(predicted_car(report((10, 10), (1, 0), 10, 3), every_step), (14, 10))
        # 每 2 步移动一格：第 8、10 步移动，共 2 格。
        every_two = EnvConfig(car_move_every=2, max_steps=200)  # 配置要求车辆能按时到站
        self.assertEqual(predicted_car(report((10, 10), (1, 0), 10, 3), every_two), (12, 10))
        # 方向未知（首报位移为 0）时就追报告位置；外推点不越出地图。
        self.assertEqual(predicted_car(report((10, 10), (0, 0), 10, 3), every_step), (10, 10))
        self.assertEqual(predicted_car(report((48, 3), (1, 0), 10, 3), every_step), (49, 3))

    def test_waits_until_first_report_then_uses_only_reported_state(self):
        config = EnvConfig(grid_size=5, max_steps=20, uav_goal=(4, 4),
                           observation_delay=3, sensing_radius=0)
        env = GridWorldEnv(config)
        policy = ExtrapolatePursuit(config)
        observation, _ = env.reset(seed=42)
        for _ in range(3):
            self.assertEqual(policy.choose_action(observation, env.valid_action_mask()), 0)
            observation, *_ = env.step(0)
        env.car_trajectory[env.time] = (0, 4)  # 改动真实车位不应影响只看报告的策略
        action = policy.choose_action(env._observation(), env.valid_action_mask())
        self.assertEqual(ACTION_NAMES[action], "east_2")

    def test_finished_sensing_navigates_like_greedy(self):
        env = GridWorldEnv(EnvConfig(observation_delay=5))
        env.reset(seed=42)
        env.sensing_complete = True
        observation, mask = env._observation(), env.valid_action_mask()
        self.assertEqual(ExtrapolatePursuit(env.config).choose_action(observation, mask),
                         greedy_action(observation, env.config, mask))


if __name__ == "__main__":
    unittest.main()
