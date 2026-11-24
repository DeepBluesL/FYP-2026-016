"""deploy/ 接口：只靠 ObservationBuilder 构造观测时，掩码和动作与原评估代码逐步一致。"""

import sys
import unittest
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))

from check import run_episode  # noqa: E402
from policy import ObservationBuilder, Policy  # noqa: E402
from uav_demo.environment import GridWorldEnv
from uav_demo.config import EnvConfig, TrainConfig
from uav_demo.agent import DQNAgent
from uav_demo.ppo import PPOAgent
from uav_demo.observation import DEFAULT_OBSERVATION_FEATURES, encode_observation  # noqa: E402


class DeployTests(unittest.TestCase):
    def test_ppo_checkpoint_matches_environment_through_report_builder(self):
        for auto_return in (False, True):
            config = EnvConfig(grid_size=5, max_steps=8, uav_goal=(4, 4),
                               observation_delay=2, auto_return=auto_return)
            agent = PPOAgent(18, 17, TrainConfig(use_motion_prediction=False, hidden_sizes=(8,)),
                             env_config=config)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "ppo.pt"
                agent.save(path, config)
                policy = Policy(path, delay=2)
                self.assertIsInstance(policy.agent, PPOAgent)
                self.assertEqual(run_episode(policy, GridWorldEnv(config), 3_000_000)[2], 0)

    def make_policy(self, delay):
        # Interface tests use current-format weights, not the archived trained models.
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        config = EnvConfig(observation_delay=delay, car_route="random", auto_return=True)
        env = GridWorldEnv(config)
        training = TrainConfig(hidden_sizes=(8,), use_motion_prediction=False,
                               observation_features=DEFAULT_OBSERVATION_FEATURES + ("rule_action",),
                               rule_prior=100.0)
        agent = DQNAgent(encode_observation(env.reset(seed=42)[0], config, training.observation_features).size,
                         env.action_count, training, env_config=config)
        path = Path(folder.name) / "policy.pt"
        agent.save(path, config)
        return Policy(path, delay=delay)

    def test_actions_match_original_evaluation(self):
        for delay in (4, 6, 8):
            policy = self.make_policy(delay)
            env = GridWorldEnv(policy.config)
            for seed in range(3_000_000, 3_000_005):
                with self.subTest(delay=delay, seed=seed):
                    self.assertEqual(run_episode(policy, env, seed)[2], 0)

    def test_builder_before_first_report_and_history(self):
        policy = self.make_policy(6)
        builder = ObservationBuilder(policy.config)
        observation = builder.build(3, (2, 2), 1, 0, False)
        self.assertFalse(observation["has_observation"])
        self.assertIsNone(observation["car_pos"])
        policy.act(observation)  # 首报前也能给出动作（规则为悬停）
        for t, position in enumerate([(10, 5), (11, 5), (11, 5)]):
            builder.report(t, position)
        observation = builder.build(8, (2, 2), 1, 0, False)
        self.assertNotIn("car_velocity", observation)
        self.assertEqual(observation["car_history"], ((0, (10, 5)), (1, (11, 5)), (2, (11, 5))))
        self.assertEqual(observation["observation_age"], 6)

    def test_projectairsim_example_grid_backend_matches_reference(self):
        import asyncio
        from projectairsim_example import GridBackend, run_episode
        from export_routes import run
        for delay in (4, 8):
            policy = self.make_policy(delay)
            env = GridWorldEnv(policy.config)
            for seed in range(3_000_000, 3_000_003):
                ref = run(env, seed, lambda o, m: policy.agent.choose_action(o, 0.0, m))
                car = [tuple(p) for p in env.car_trajectory]
                result = asyncio.run(run_episode(policy, GridBackend(), car))
                with self.subTest(delay=delay, seed=seed):
                    self.assertEqual(result, ref)

    def test_delay_must_match_model(self):
        policy = self.make_policy(6)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "d6.pt"
            policy.agent.save(path, policy.config)
            with self.assertRaisesRegex(ValueError, "delay=4"):
                Policy(path, delay=4)


if __name__ == "__main__":
    unittest.main()
