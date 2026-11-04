import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from uav_delay.config import Config
from uav_delay.env import DelayEnv
from uav_delay.rules import chase, extrapolate_action


def scene(stationary=False):
    return dict(
        seed=-1, start=[50, 50], goal=[90, 50],
        route=[[50 + (0 if stationary else 5 * min(i, 20)), 50] for i in range(101)],
    )


def encoded(relative, view=50.0):
    relative = np.asarray(relative, dtype=float) / view
    return relative / max(1.0, np.linalg.norm(relative))


def create(delay=4, cfg=None, fixture=None):
    env = DelayEnv(delay, cfg)
    env.reset(options={"scenario": fixture or scene()})
    return env


class DelayTests(unittest.TestCase):
    def test_observation_is_exactly_d_steps_old(self):
        for delay in (0, 1, 4, 8):
            env = create(delay)
            for t in range(1, 12):
                obs, _, _, _, info = env.step([0, 0])
                self.assertEqual(obs.shape, (13,))
                if t < delay:  # 首次报告之前：没有车辆信息
                    self.assertIsNone(env.received_car())
                    np.testing.assert_array_equal(obs[6:10], [0, 0, 0, 0])
                    self.assertEqual(obs[12], 0.0)
                    self.assertIsNone(info["observed_step"])
                    continue
                seen = t - delay
                np.testing.assert_allclose(env.received_car()[0], [50 + 5 * seen, 50])
                np.testing.assert_allclose(obs[6:8], encoded([5 * seen, 0]), atol=1e-6)
                np.testing.assert_allclose(obs[8:10], [1 if seen > 0 else 0, 0], atol=1e-5)
                self.assertEqual(obs[12], 1.0)
                self.assertEqual(info["observed_step"], seen)
                self.assertEqual(info["observation_age"], delay)

    def test_first_report_arrives_at_step_d_with_step_zero_position(self):
        env = create(3)
        obs = env.observation()
        self.assertIsNone(env.received_car())  # 开局（d>0）不知道车在哪
        np.testing.assert_array_equal(obs[6:10], [0, 0, 0, 0])
        self.assertEqual(obs[12], 0.0)
        for _ in range(3):
            obs, *_ = env.step([0, 0])
        car, shift = env.received_car()  # t=3 收到 t=0 的位置，方向未知（位移 0）
        np.testing.assert_array_equal(car, [50, 50])
        np.testing.assert_array_equal(shift, [0, 0])
        self.assertEqual(obs[12], 1.0)
        no_delay = DelayEnv(0).reset(options={"scenario": scene()})[0]
        self.assertEqual(no_delay[12], 1.0)  # 无延迟时开局即有报告

    def test_rules_hover_before_first_report(self):
        env = create(4)
        for predict in (True, False):
            np.testing.assert_array_equal(chase(predict)(env.observation(), env), [0, 0])

    def test_car_stays_observed_after_minimum_collection(self):
        env = create(0, fixture=scene(stationary=True))
        env.collected = 5.0
        obs = env.observation()
        np.testing.assert_allclose(obs[6:8], [0, 0], atol=1e-6)

    def test_own_position_and_progress_are_current(self):
        env = create(4)
        obs, *_ = env.step([0.5, 0])
        np.testing.assert_allclose(obs[:2] * 245, [54, 50], atol=1e-5)
        np.testing.assert_array_equal(obs[6:8], [0, 0])  # 延迟 4，第 1 步还没有报告
        self.assertAlmostEqual(obs[4], 0.99, places=6)
        self.assertAlmostEqual(obs[5], 0.2, places=6)

    def test_delay_does_not_change_motion_or_reward(self):
        a, b = create(0), create(8)
        for _ in range(100):
            first, second = a.step([0.625, 0]), b.step([0.625, 0])
            np.testing.assert_array_equal(a.pos, b.pos)
            self.assertEqual(first[1], second[1])
            for key in ("collected", "steps", "travel", "success"):
                self.assertEqual(first[4][key], second[4][key])
            if first[2]:
                break

    def test_future_positions_do_not_leak(self):
        first, second = scene(), scene()
        first["start"] = second["start"] = [20, 200]
        second["route"][2:] = [[55, 50 + 5 * i] for i in range(1, 100)]
        a, b = create(4, fixture=first), create(4, fixture=second)
        for _ in range(5):  # 第 2 步起路线不同；延迟 4 步，前 5 步观测必须一致
            ra, rb = a.step([0.5, 0]), b.step([0.5, 0])
            np.testing.assert_array_equal(ra[0], rb[0])
            self.assertNotIn("car", ra[4])

    def test_approach_only_before_minimum_collection(self):
        diagonal = 245 * np.sqrt(2)
        fixture = scene(stationary=True)
        fixture["start"] = [20, 50]  # 车静止在 (50, 50)
        env = create(4, fixture=fixture)
        *_, info = env.step([1, 0])
        self.assertAlmostEqual(info["reward_parts"]["approach"], 8 / diagonal)
        env.collected = 5.0
        *_, info = env.step([1, 0])
        self.assertEqual(info["reward_parts"]["approach"], 0.0)

    def test_contact_inside_step_is_counted(self):
        fixture = scene(stationary=True)
        fixture["start"] = [45, 50]
        env = create(cfg=Config(radius=1.0), fixture=fixture)
        _, _, _, _, info = env.step([1, 0])
        self.assertEqual(info["steps"], 1)
        self.assertAlmostEqual(info["gained"], 0.25)

    def test_far_car_is_given_as_direction_only(self):
        fixture = scene(stationary=True)
        fixture["route"] = [[50, 200]] * 101
        obs = create(0, fixture=fixture).observation()
        np.testing.assert_allclose(obs[6:8], [0, 1], atol=1e-6)

    def test_failure_ends_as_soon_as_goal_is_unreachable_and_cancels_sensing(self):
        # 最少采集量设得达不到，无人机停在车上一直采集，直到来不及返程。
        env = create(cfg=Config(required=200.0), fixture=scene(stationary=True))
        done = False
        while not done:
            obs, _, done, truncated, info = env.step([0, 0])
            if not done:
                self.assertGreaterEqual(env.slack(), 0)
                self.assertAlmostEqual(obs[10], env.slack() / 100, places=6)
        self.assertFalse(truncated)
        self.assertEqual(info["steps"], 96)  # 剩 4 步 < 5 步，已不可能按时到达
        self.assertEqual(info["collected"], 96.0)
        self.assertEqual(info["score"], 0.0)
        self.assertAlmostEqual(info["total_reward"], -2.0)

    def test_slack_counts_full_speed_steps_to_goal_radius(self):
        env = create()
        self.assertEqual(env.slack(), 100 - 5)
        env.pos[:] = [86, 50]  # 已在终点范围内
        self.assertEqual(env.slack(), 100)

    def test_success_keeps_uncapped_sensing(self):
        fixture = scene(stationary=True)
        fixture["goal"] = [58, 50]
        env = create(fixture=fixture)
        for _ in range(20):
            env.step([0, 0])
        _, _, done, _, info = env.step([1, 0])
        self.assertTrue(done)
        self.assertTrue(info["success"])
        self.assertAlmostEqual(info["score"], info["collected"])
        self.assertGreater(info["collected"], 20.0)
        np.testing.assert_array_equal(env.pos, [58, 50])

    def test_goal_requires_minimum_collection(self):
        env = create()
        env.pos[:] = [93, 50]
        env.step([0, 0])
        self.assertFalse(env.done)

    def test_completed_progress_agrees_with_success_at_float_boundary(self):
        env = create()
        env.pos[:] = env.goal
        env.collected = 4.999999927608964
        self.assertEqual(env.observation()[5], 1.0)
        _, _, done, _, info = env.step([0, 0])
        self.assertTrue(info["sensing_complete"])
        self.assertTrue(info["success"])
        self.assertTrue(done)

    def test_auto_return_at_margin_after_minimum_collection_and_ignores_actions(self):
        fixture = scene(stationary=True)
        fixture["goal"] = [50, 130]  # 相距 80，直飞需 10 步
        env = create(fixture=fixture)
        env.step_count = 88  # 余量 2，但未采够：不返程
        env.step([1, 0])
        self.assertFalse(env.returning)
        np.testing.assert_allclose(env.pos, [58, 50])
        env.collected, env.step_count, env.pos[:] = 5.0, 87, [50, 50]  # 余量 3：继续跟车
        env.step([1, 0])
        self.assertFalse(env.returning)
        env.pos[:] = [50, 50]  # 第 88 步，余量 2：自动返程，忽略向东的动作
        obs, *_ = env.step([1, 0])
        self.assertTrue(env.returning)
        self.assertEqual(obs[11], 1.0)
        np.testing.assert_allclose(env.pos, [50, 58])
        while not env.done:
            *_, info = env.step([1, 0])
        self.assertTrue(info["success"])

    def test_chase_rules_aim_at_received_or_extrapolated_position(self):
        env = create(4)
        for _ in range(6):
            obs, *_ = env.step([0, 0])
        # 第 6 步收到第 2 步（x=60），位移 5；UAV 在 x=50。
        np.testing.assert_allclose(chase(False)(obs, env), [1, 0], atol=1e-6)
        env.pos[:] = [58, 50]  # 只追收到位置：差 2 m；外推 5 步到 x=85：差 27 m
        np.testing.assert_allclose(chase(False)(obs, env) * 8, [2, 0], atol=1e-6)
        np.testing.assert_allclose(chase(True)(obs, env), [1, 0], atol=1e-6)

    def test_residual_with_zero_correction_matches_extrapolation_rule(self):
        cfg = Config(residual=1.0)
        residual, plain = DelayEnv(4, cfg), DelayEnv(4, Config())
        obs, _ = residual.reset(seed=3)
        plain_obs, _ = plain.reset(seed=3)
        self.assertEqual(obs.shape, (15,))
        rule = chase(True)
        done = False
        while not done:
            np.testing.assert_allclose(obs[13:], extrapolate_action(residual), atol=1e-6)
            obs, _, done, _, info = residual.step([0, 0])
            plain_obs, _, _, _, plain_info = plain.step(rule(plain_obs, plain))
            np.testing.assert_array_equal(residual.pos, plain.pos)  # 两边调用同一个规则函数
        self.assertEqual(info["score"], plain_info["score"])

    def test_residual_correction_shifts_rule_action(self):
        fixture = scene(stationary=True)
        fixture["start"] = [20, 50]  # 车在东边 30 m，规则动作为 (1, 0)
        env = create(0, cfg=Config(residual=1.0), fixture=fixture)
        env.step([0, 1])  # 修正 (0, 1)：合成 (1, 1)，按最大速度缩放
        np.testing.assert_allclose(env.pos, [20 + 8 / np.sqrt(2), 50 + 8 / np.sqrt(2)])

    def test_destination_route_is_reproducible_and_moves_one_cell_per_step(self):
        cfg = Config()
        a, b = DelayEnv(4, cfg), DelayEnv(4, cfg)
        a.reset(seed=7)
        b.reset(seed=7)
        np.testing.assert_array_equal(a.route, b.route)
        steps = np.abs(np.diff(a.route, axis=0)).sum(axis=1)
        np.testing.assert_array_equal(steps, 5.0)
        self.assertTrue(np.all((a.route >= 0) & (a.route <= cfg.span)))
        np.testing.assert_array_equal(a.route, np.asarray(a.scene["route"]))
        self.assertEqual(len(a.route), cfg.duration + 1)

    def test_scene_sampling_and_gym_interfaces(self):
        first = DelayEnv(0, seed_range=(400_000, 402_000))
        second = DelayEnv(8, seed_range=(400_000, 402_000))
        for seed in (10, None, None):
            first.reset(seed=seed)
            second.reset(seed=seed)
            self.assertEqual(first.scene, second.scene)
            self.assertEqual(len(first.route), 101)
        for delay in (0, 4, 10):
            env = DelayEnv(delay)
            check_env(env, skip_render_check=True)
            obs, _ = env.reset(seed=42)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertEqual(env.action_space.shape, (2,))


if __name__ == "__main__":
    unittest.main()
