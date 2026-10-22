"""验证时钟、速度、采集和终止语义，不需要训练或绘图依赖。"""

import unittest

import numpy as np

from uav_demo.config import EnvConfig
from uav_demo.environment import ACTION_NAMES, GridWorldEnv
from uav_demo.vehicle import generate_vehicle_trajectory


class VehicleTests(unittest.TestCase):
    def test_seeded_routes_stay_in_bounds_reach_goal_and_obey_clock(self):
        for size, every in ((5, 1), (6, 3), (50, 1), (50, 2)):
            config = EnvConfig(
                grid_size=size, uav_goal=(size - 1, size - 1),
                car_move_every=every, max_steps=2 * (size - 1) * every + 1,
            )
            for seed in range(20):
                with self.subTest(size=size, every=every, seed=seed):
                    path, goal = generate_vehicle_trajectory(config, np.random.default_rng(seed))
                    replay, replay_goal = generate_vehicle_trajectory(config, np.random.default_rng(seed))
                    np.testing.assert_array_equal(path, replay)
                    self.assertEqual(goal, replay_goal)
                    self.assertEqual(path.shape, (config.max_steps + 1, 2))
                    self.assertTrue(np.issubdtype(path.dtype, np.integer))
                    self.assertTrue(np.all((0 <= path) & (path < size)))
                    self.assertGreaterEqual(path[0, 0], 2 * size // 3)
                    self.assertLess(path[0, 1], (size + 2) // 3)
                    self.assertEqual(tuple(path[-1]), goal)
                    self.assertEqual(goal, (0, size - 1))
                    self.assertNotEqual(tuple(path[0]), goal)
                    distances = np.abs(np.diff(path, axis=0)).sum(axis=1)
                    self.assertTrue(np.all(distances <= 1))
                    idle_ticks = np.arange(1, len(path)) % every != 0
                    self.assertTrue(np.all(distances[idle_ticks] == 0))
                    arrival = np.flatnonzero(np.all(path == goal, axis=1))[0]
                    self.assertTrue(np.all(path[arrival:] == goal))
                    moving_ticks = np.arange(1, arrival + 1) % every == 0
                    self.assertTrue(np.all(distances[:arrival][moving_ticks] == 1))

    def test_different_seeds_produce_different_scenarios(self):
        config = EnvConfig()
        first, _ = generate_vehicle_trajectory(config, np.random.default_rng(0))
        second, _ = generate_vehicle_trajectory(config, np.random.default_rng(1))
        self.assertFalse(np.array_equal(first, second))


class EnvironmentTests(unittest.TestCase):
    @staticmethod
    def action(name):
        return ACTION_NAMES.index(name)

    def make_env(self, **overrides):
        values = dict(grid_size=5, max_steps=20, uav_goal=(4, 4), sensing_radius=0.0)
        values.update(overrides)
        env = GridWorldEnv(EnvConfig(**values))
        env.reset(seed=42)
        return env

    def test_current_observation_is_independent_of_unseen_future_route(self):
        env = self.make_env()
        observation, _ = env.reset(seed=42)
        self.assertEqual(observation.dtype, np.float32)
        self.assertEqual(observation.shape, (20,))
        self.assertEqual(env.observation_size, 20)
        self.assertEqual(observation[12], 0)  # 采集失败标志
        # 当前状态相同，任意更换真正的未来轨迹，策略能看到的输入也必须相同。
        env.car_trajectory[1:] = (4, 0)
        np.testing.assert_array_equal(env._observation(), observation)
        next_observation, *_ = env.step(0)
        env.car_trajectory[2:] = (1, 3)
        np.testing.assert_array_equal(env._observation(), next_observation)
        self.assertAlmostEqual(float(next_observation[9]), 1 / 20)
        np.testing.assert_allclose(next_observation[13:15], (1.0, 0.0))
        np.testing.assert_allclose(next_observation[15:17], (1.0, 1.0))

    def test_current_observation_size_does_not_depend_on_episode_horizon(self):
        for horizon in (20, 100, 300):
            with self.subTest(horizon=horizon):
                env = self.make_env(max_steps=horizon)
                observation, _ = env.reset(seed=42)
                self.assertEqual(env.observation_size, 20)
                self.assertEqual(observation.shape, (20,))

    def test_car_velocity_uses_only_observed_moves_and_reset_clears_it(self):
        env = self.make_env(car_move_every=2)
        env.car_trajectory[:] = (2, 2)
        env.car_trajectory[:2] = (3, 1)
        env.car_trajectory[2:4] = (2, 1)
        np.testing.assert_array_equal(env._observation()[17:19], (0, 0))
        for expected, phase in (((0, 0), 0.5), ((-1, 0), 0), ((-1, 0), 0.5), ((0, 1), 0)):
            observation, _, _, _, info = env.step(0)
            np.testing.assert_array_equal(observation[17:19], expected)
            self.assertEqual(tuple(info["car_velocity"]), expected)
            self.assertEqual(observation[19], phase)
        observation, _ = env.reset(seed=42)
        np.testing.assert_array_equal(observation[17:19], (0, 0))

    def test_vehicle_does_not_depend_on_uav_actions(self):
        first, second = self.make_env(), self.make_env()
        actions = tuple(self.action(name) for name in ("north_2", "east_2", "south_2", "west_2"))
        for action in actions * 4:
            first.step(0)
            second.step(action)
            self.assertEqual(first.car_pos, second.car_pos)
        np.testing.assert_array_equal(first.car_trajectory, second.car_trajectory)

    def test_speed_changes_hover_and_rejected_boundary_displacement(self):
        env = self.make_env()
        env.step(self.action("north_2"))
        self.assertEqual((env.uav_pos, env.uav_speed), ((0, 2), 2))
        env.step(self.action("east_1"))
        self.assertEqual((env.uav_pos, env.uav_speed), ((1, 2), 1))
        env.step(0)
        self.assertEqual((env.uav_pos, env.uav_speed), ((1, 2), 1))
        env.step(self.action("east_2"))
        self.assertEqual((env.uav_pos, env.uav_speed), ((3, 2), 2))
        self.assertFalse(env.valid_action_mask()[self.action("east_2")])
        self.assertTrue(env.valid_action_mask()[0])
        _, _, _, _, info = env.step(self.action("east_2"))
        self.assertEqual(env.uav_pos, (3, 2))  # 整段取消，不裁剪到边缘。
        self.assertTrue(info["invalid_move"])
        self.assertEqual(env.travel_distance, 5.0)
        env.step(0)
        self.assertEqual(env.uav_speed, 2)  # 悬停保留名义速度。

    def test_invalid_actions_do_not_advance_time(self):
        env = self.make_env()
        for action in (-1, env.action_count, 1.5, True):
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.step(action)
            self.assertEqual(env.time, 0)
            self.assertEqual(env.uav_pos, (0, 0))

    def test_consecutive_reset_cumulative_retention_and_completion_latch(self):
        for mode, expected in (
            ("consecutive", [1, 2, 0, 1, 2, 3]),
            ("cumulative", [1, 2, 2, 3, 3, 3]),
        ):
            with self.subTest(mode=mode):
                env = self.make_env(sensing_mode=mode, sensing_steps=3)
                # 人工路线仅用于隔离采集逻辑，t=3 和 t=7 失去接触。
                env.car_trajectory[:] = (0, 0)
                env.car_trajectory[3] = (4, 4)
                env.car_trajectory[7] = (4, 4)
                actual = []
                for _ in range(6):
                    env.step(0)
                    actual.append(env.sensing_progress)
                self.assertEqual(actual, expected)
                self.assertTrue(env.sensing_complete)
                env.step(0)
                self.assertEqual(env.sensing_progress, 3)
                self.assertTrue(env.sensing_complete)

    def test_fast_move_senses_endpoint_once(self):
        cases = (
            ("north_2", (0, 1), (0, 2), 0),
            ("north_2", (0, 2), (0, 2), 1),
            ("north_east_2", (1, 1), (2, 2), 0),
            ("north_east_2", (2, 2), (2, 2), 1),
        )
        for name, car_position, destination, expected in cases:
            with self.subTest(action=name, car_position=car_position):
                env = self.make_env(sensing_steps=2)
                env.car_trajectory[:] = car_position
                env.step(self.action(name))
                self.assertEqual(env.uav_pos, destination)
                self.assertEqual(env.sensing_progress, expected)

    def test_reaching_goal_before_sensing_is_not_success(self):
        env = self.make_env(sensing_steps=2)
        env.car_trajectory[:] = (0, 0)
        for name in ("north_2", "north_2", "east_2", "east_2"):
            _, _, terminated, truncated, info = env.step(self.action(name))
        self.assertEqual(env.uav_pos, (4, 4))
        self.assertFalse(terminated or truncated)
        self.assertFalse(info["success"])
        self.assertEqual(info["phase"], "sensing")
        self.assertFalse(info["sensing_finished"])

    def test_completed_sensing_is_rewarded_only_once(self):
        env = self.make_env(sensing_steps=2)
        env.car_trajectory[:] = (0, 0)
        _, first_reward, *_ = env.step(0)
        _, completion_reward, *_ = env.step(0)
        _, later_reward, *_ = env.step(0)
        self.assertAlmostEqual(completion_reward - first_reward, env.config.sensing_complete_reward)
        self.assertAlmostEqual(later_reward, env.config.step_penalty)
        env.step(self.action("north_2"))
        self.assertTrue(env.sensing_complete)
        self.assertEqual(env.sensing_progress, 2)

    def test_success_on_final_tick_takes_precedence_over_timeout(self):
        env = self.make_env(sensing_steps=1)
        env.car_trajectory[:] = (4, 4)
        final_moves = [self.action(name) for name in ("north_2", "north_2", "east_2", "east_2")]
        for action in [self.action("hover")] * 16 + final_moves:
            _, _, terminated, truncated, info = env.step(action)
        self.assertEqual(env.time, 20)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["success"])
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_all_diagonal_directions_obey_grid_step_and_euclidean_distance(self):
        directions = {
            "north_east": (1, 1), "south_east": (1, -1),
            "north_west": (-1, 1), "south_west": (-1, -1),
        }
        for name, (dx, dy) in directions.items():
            for speed in (1, 2):
                with self.subTest(direction=name, speed=speed):
                    env = self.make_env(uav_start=(2, 2))
                    env.step(self.action(f"{name}_{speed}"))
                    self.assertEqual(env.uav_pos, (2 + speed * dx, 2 + speed * dy))
                    self.assertEqual(env.uav_speed, speed)
                    self.assertAlmostEqual(env.travel_distance, speed * np.sqrt(2))

    def test_diagonal_boundary_rejects_whole_displacement(self):
        env = self.make_env(uav_start=(3, 1))
        action = self.action("north_east_2")
        self.assertFalse(env.valid_action_mask()[action])
        _, _, _, _, info = env.step(action)
        self.assertEqual(env.uav_pos, (3, 1))
        self.assertEqual(env.uav_speed, 2)
        self.assertEqual(env.travel_distance, 0)
        self.assertTrue(info["invalid_move"])
        self.assertEqual(info["time"], 1)

    def test_action_mask_contains_all_seventeen_actions(self):
        env = self.make_env()
        mask = env.valid_action_mask()
        self.assertEqual(mask.shape, (17,))
        legal_names = {name for name, allowed in zip(ACTION_NAMES, mask) if allowed}
        self.assertEqual(legal_names, {
            "hover", "north_1", "east_1", "north_east_1",
            "north_2", "east_2", "north_east_2",
        })
        env.uav_pos = (2, 2)
        self.assertTrue(env.valid_action_mask().all())

    def test_timeout_and_reset_guard(self):
        env = self.make_env()
        env.car_trajectory[:] = (4, 4)
        for _ in range(20):
            _, _, terminated, truncated, info = env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertFalse(info["success"])
        with self.assertRaises(RuntimeError):
            env.step(0)
        env.reset(seed=0)
        env.step(0)
        self.assertEqual(env.time, 1)
        fresh = GridWorldEnv(env.config)
        with self.assertRaises(RuntimeError):
            fresh.step(0)

    def test_arrival_failure_is_once_and_cannot_collect_after_deadline(self):
        for mode in ("consecutive", "cumulative"):
            with self.subTest(mode=mode):
                env = self.make_env(sensing_mode=mode, step_penalty=0, progress_reward=0)
                env.car_goal = (0, 4)
                env.car_trajectory[:2] = (0, 3)
                env.car_trajectory[2:] = env.car_goal
                _, first_reward, terminated, truncated, _ = env.step(0)
                self.assertEqual(first_reward, 0)
                self.assertFalse(terminated or truncated)
                observation, reward, terminated, truncated, info = env.step(0)
                self.assertEqual(reward, env.config.sensing_failure_penalty)
                self.assertFalse(terminated or truncated)
                self.assertTrue(info["sensing_failed"])
                self.assertFalse(info["sensing_complete"])
                self.assertEqual(info["phase"], "to_goal")
                self.assertEqual(observation[12], 1)
                # 到站后即使重新接触车辆，也不能继续采集或再次扣失败罚分。
                env.uav_pos = env.car_goal
                for _ in range(env.config.sensing_steps):
                    _, reward, _, _, info = env.step(0)
                    self.assertEqual(reward, 0)
                    self.assertEqual(info["sensing_progress"], 0)
                env.step(self.action("east_2"))
                _, reward, terminated, truncated, info = env.step(self.action("east_2"))
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertTrue(info["uav_reached_goal"])
                self.assertFalse(info["success"])
                self.assertAlmostEqual(reward, 2 * env.config.movement_penalty)  # 只收飞行成本，不冒领成功奖励。
                env.reset(seed=42)
                self.assertFalse(env.sensing_failed or env.sensing_finished)

    def test_completion_before_or_on_car_arrival_is_not_failure(self):
        for required in (1, 2):
            with self.subTest(required=required):
                env = self.make_env(uav_start=(0, 4), sensing_radius=1, sensing_steps=required,
                                    step_penalty=0, progress_reward=0)
                env.car_goal = (0, 4)
                env.car_trajectory[:2] = (0, 3)
                env.car_trajectory[2:] = env.car_goal
                rewards = [env.step(0)[1], env.step(0)[1]]
                self.assertTrue(env.sensing_complete)
                self.assertFalse(env.sensing_failed)
                self.assertAlmostEqual(sum(rewards), required * env.config.contact_reward
                                       + env.config.sensing_complete_reward)

    def test_failed_phase_shapes_distance_toward_uav_goal(self):
        env = self.make_env()
        env.car_goal = (0, 4)
        env.car_trajectory[:] = env.car_goal
        env.step(0)  # 车辆已到站，触发失败。
        _, reward, _, _, _ = env.step(self.action("east_2"))
        expected = env.config.step_penalty + 2 * env.config.movement_penalty + (
            env.config.progress_reward * env.config.failed_return_progress_scale
            * (np.sqrt(32) - np.sqrt(20))
        )
        self.assertAlmostEqual(reward, expected)

    def test_hover_collects_as_well_as_unnecessary_center_chasing_with_less_cost(self):
        waiting = self.make_env(uav_start=(1, 1), sensing_radius=2, sensing_steps=3)
        chasing = self.make_env(uav_start=(1, 1), sensing_radius=2, sensing_steps=3)
        for env in (waiting, chasing):
            env.car_trajectory[:] = (3, 1)
        _, wait_reward, _, _, wait_info = waiting.step(self.action("hover"))
        _, move_reward, _, _, move_info = chasing.step(self.action("east_1"))
        self.assertEqual(wait_info["sensing_progress"], 1)
        self.assertEqual(move_info["sensing_progress"], 1)
        self.assertTrue(wait_info["contact"] and move_info["contact"])
        self.assertGreater(wait_reward, move_reward)
        self.assertAlmostEqual(wait_reward - move_reward, -waiting.config.movement_penalty)
        self.assertEqual(waiting.travel_distance, 0)
        self.assertEqual(chasing.travel_distance, 1)

    def test_waiting_on_future_path_is_legal_and_can_start_sensing(self):
        env = self.make_env(uav_start=(2, 2), sensing_radius=1, sensing_steps=3)
        env.car_trajectory[:] = (2, 2)
        env.car_trajectory[:3] = ((4, 3), (4, 2), (3, 2))
        hover = self.action("hover")
        self.assertTrue(env.valid_action_mask()[hover])
        _, _, _, _, first_info = env.step(hover)
        self.assertFalse(first_info["contact"])
        self.assertTrue(env.valid_action_mask()[hover])
        _, _, _, _, next_info = env.step(hover)
        self.assertTrue(next_info["contact"])
        self.assertEqual(next_info["sensing_progress"], 1)
        self.assertEqual(env.uav_pos, (2, 2))
        self.assertEqual(env.travel_distance, 0)

    def test_going_to_goal_then_waiting_for_sensing_failure_has_negative_return(self):
        # 固定先上后右的直接终点路线，不追车、不按未来路线挑动作。
        direct_actions = (
            [self.action("north_2")] * 24 + [self.action("north_1")]
            + [self.action("east_2")] * 24 + [self.action("east_1")]
        )
        for seed in range(10):
            with self.subTest(seed=seed):
                env = GridWorldEnv()
                env.reset(seed=seed)
                total_reward = 0.0
                for time in range(env.config.max_steps):
                    action = direct_actions[time] if time < len(direct_actions) else 0
                    _, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        break
                self.assertTrue(info["sensing_failed"])
                self.assertTrue(info["uav_reached_goal"])
                self.assertFalse(info["success"])
                self.assertLess(total_reward, 0)

    def test_failure_and_goal_on_last_tick_terminate_without_success(self):
        env = self.make_env(uav_start=(4, 4), step_penalty=0, progress_reward=0)
        env.car_goal = (0, 4)
        env.car_trajectory[:-1] = (0, 3)
        env.car_trajectory[-1] = env.car_goal
        for _ in range(env.config.max_steps):
            _, reward, terminated, truncated, info = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated or info["success"])
        self.assertTrue(info["sensing_failed"])
        self.assertEqual(reward, env.config.sensing_failure_penalty)


if __name__ == "__main__":
    unittest.main()
