"""spec 分支的可选开关：势函数奖励、距离版规则先验、softmax 探索；默认关闭时行为不变。"""

import unittest

import numpy as np
import torch

from uav_demo.agent import DQNAgent
from uav_demo.config import EnvConfig, TrainConfig
from uav_demo.environment import GridWorldEnv
from uav_demo.extrapolate_pursuit import ExtrapolatePursuit
from uav_demo.observation import DEFAULT_OBSERVATION_FEATURES, encode_observation, feature_slice

DISTANCE_FEATURES = DEFAULT_OBSERVATION_FEATURES + ("rule_distance",)
RULE_FEATURES = DEFAULT_OBSERVATION_FEATURES + ("rule_action",)


def run_episode(config, actions_seed=0, seed=5):
    env = GridWorldEnv(config)
    env.reset(seed=seed)
    rng = np.random.default_rng(actions_seed)
    total, done = 0.0, False
    while not done:
        action = int(rng.choice(np.flatnonzero(env.valid_action_mask())))
        _, reward, terminated, truncated, _ = env.step(action)
        total += reward
        done = terminated or truncated
    return total


class PotentialRewardTests(unittest.TestCase):
    def test_shaping_telescopes_to_minus_initial_potential(self):
        """γ=1 时整局塑形之和只取决于初始势：与去掉塑形项的同一局相比，差值恰为 −Φ(s0)。"""
        for route in ("fixed", "random"):
            for actions_seed in range(3):
                shaped = EnvConfig(reward_v2=True, shaping_gamma=1.0, car_route=route, observation_delay=2)
                plain = EnvConfig(reward_v2=True, shaping_gamma=1.0, car_route=route, observation_delay=2,
                                  progress_reward=0.0, contact_reward=0.0)
                env = GridWorldEnv(shaped)
                env.reset(seed=5)
                initial = env._potential()
                difference = run_episode(shaped, actions_seed) - run_episode(plain, actions_seed)
                self.assertAlmostEqual(difference, -initial, places=6)

    def test_random_route_timeout_without_sensing_counts_as_failure(self):
        config = EnvConfig(reward_v2=True, shaping_gamma=1.0, car_route="random")
        env = GridWorldEnv(config)
        env.reset(seed=3)
        initial, total, done = env._potential(), 0.0, False
        while not done:
            _, reward, terminated, truncated, _ = env.step(0)  # 原地悬停，采不到
            total += reward
            done = terminated or truncated
        self.assertFalse(env.sensing_complete)
        expected = config.max_steps * config.step_penalty + config.sensing_failure_penalty - initial
        self.assertAlmostEqual(total, expected, places=6)


class DistancePriorTests(unittest.TestCase):
    def setUp(self):
        self.config = EnvConfig(observation_delay=2)
        env = GridWorldEnv(self.config)
        self.observation, _ = env.reset(seed=7)
        for _ in range(4):
            self.observation, *_ = env.step(0)

    def test_rule_action_has_zero_gap(self):
        features = encode_observation(self.observation, self.config, DISTANCE_FEATURES)
        gaps = features[feature_slice(DISTANCE_FEATURES, "rule_distance")]
        rule = ExtrapolatePursuit(self.config).choose_action(self.observation, np.ones(17, dtype=bool))
        self.assertEqual(gaps[rule], 0.0)
        self.assertTrue(np.all(gaps >= 0))
        self.assertGreater(gaps.max(), 0)

    def test_decision_q_subtracts_scaled_gap(self):
        train = TrainConfig(use_motion_prediction=False, observation_features=DISTANCE_FEATURES,
                            rule_prior=0.5, rule_prior_feature="rule_distance", hidden_sizes=(8,))
        agent = DQNAgent(encode_observation(self.observation, self.config, DISTANCE_FEATURES).size,
                         17, train, env_config=self.config)
        features = torch.as_tensor(agent._features(self.observation))[None]
        gaps = features[0, feature_slice(DISTANCE_FEATURES, "rule_distance")]
        expected = agent.online(features)[0] - 0.5 * gaps
        torch.testing.assert_close(agent._q(agent.online, features)[0], expected)

    def test_prior_feature_must_be_in_observation(self):
        with self.assertRaises(ValueError):
            TrainConfig(observation_features=RULE_FEATURES, rule_prior=1.0, rule_prior_feature="rule_distance")


class SoftmaxExplorationTests(unittest.TestCase):
    def test_softmax_exploration_follows_strong_prior_uniform_does_not(self):
        config = EnvConfig(observation_delay=2)
        env = GridWorldEnv(config)
        observation, _ = env.reset(seed=7)
        for _ in range(4):
            observation, *_ = env.step(0)
        rule = ExtrapolatePursuit(config).choose_action(observation, env.valid_action_mask())
        hits = {}
        for exploration in ("uniform", "softmax"):
            train = TrainConfig(use_motion_prediction=False, observation_features=RULE_FEATURES,
                                rule_prior=50.0, exploration=exploration, hidden_sizes=(8,))
            agent = DQNAgent(encode_observation(observation, config, RULE_FEATURES).size,
                             17, train, env_config=config)
            picks = [agent.choose_action(observation, epsilon=1.0, mask=env.valid_action_mask())
                     for _ in range(200)]
            hits[exploration] = np.mean(np.asarray(picks) == rule)
        self.assertGreater(hits["softmax"], 0.95)
        self.assertLess(hits["uniform"], 0.3)


if __name__ == "__main__":
    unittest.main()
