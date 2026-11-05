"""自动返程、外推规则动作特征与规则先验 Q 值：默认关闭时行为不变，开启后按规则起步。"""

import unittest

import numpy as np

from uav_demo.agent import DQNAgent
from uav_demo.config import EnvConfig, TrainConfig
from uav_demo.environment import GridWorldEnv
from uav_demo.extrapolate_pursuit import ExtrapolatePursuit
from uav_demo.observation import DEFAULT_OBSERVATION_FEATURES, encode_observation, feature_slice

RULE_FEATURES = DEFAULT_OBSERVATION_FEATURES + ("rule_action",)


class AutoReturnTests(unittest.TestCase):
    def test_mask_unchanged_by_default_and_single_shortest_action_when_enabled(self):
        for auto_return in (False, True):
            env = GridWorldEnv(EnvConfig(auto_return=auto_return))
            env.reset(seed=42)
            full = env.valid_action_mask()
            env.sensing_complete = True
            mask = env.valid_action_mask()
            if not auto_return:
                np.testing.assert_array_equal(mask, full)
                continue
            self.assertEqual(mask.sum(), 1)  # 只剩一条返程动作：(0,0) 去 (49,49) 应斜向 2 格
            self.assertEqual(int(np.flatnonzero(mask)[0]), 13)


class RuleFeatureTests(unittest.TestCase):
    def test_rule_action_feature_is_onehot_of_extrapolation_choice(self):
        config = EnvConfig(observation_delay=2)
        env = GridWorldEnv(config)
        observation, _ = env.reset(seed=7)
        for _ in range(4):
            observation, *_ = env.step(0)
        features = encode_observation(observation, config, RULE_FEATURES)
        onehot = features[feature_slice(RULE_FEATURES, "rule_action")]
        expected = ExtrapolatePursuit(config).choose_action(observation, np.ones(17, dtype=bool))
        self.assertEqual(onehot.sum(), 1.0)
        self.assertEqual(int(np.argmax(onehot)), expected)
        self.assertEqual(features.size, encode_observation(observation, config).size + 17)

    def test_rule_prior_requires_rule_feature(self):
        with self.assertRaises(ValueError):
            TrainConfig(rule_prior=5.0)

    def test_large_rule_prior_makes_untrained_agent_follow_rule(self):
        config = EnvConfig(observation_delay=2)
        train = TrainConfig(use_motion_prediction=False, observation_features=RULE_FEATURES,
                            rule_prior=1000.0, hidden_sizes=(8,))
        env = GridWorldEnv(config)
        observation, _ = env.reset(seed=7)
        size = encode_observation(observation, config, RULE_FEATURES).size
        agent = DQNAgent(size, env.action_count, train, env_config=config)
        rule = ExtrapolatePursuit(config)
        for _ in range(10):
            mask = env.valid_action_mask()
            action = agent.choose_action(observation, epsilon=0.0, mask=mask)
            self.assertEqual(action, rule.choose_action(observation, mask))
            observation, *_ = env.step(action)


class PersistentPriorTests(unittest.TestCase):
    def test_fitted_q_excludes_prior_so_deviation_cost_persists(self):
        """对同一条终止转移（执行规则动作、奖励 3）反复更新：网络直接拟合奖励，不吸收先验。"""
        import torch
        config = EnvConfig(observation_delay=2)
        train = TrainConfig(use_motion_prediction=False, observation_features=RULE_FEATURES,
                            rule_prior=5.0, hidden_sizes=(16,),
                            batch_size=1, learning_starts=0, learning_rate=0.01)
        env = GridWorldEnv(config)
        observation, _ = env.reset(seed=7)
        for _ in range(3):
            observation, *_ = env.step(0)
        agent = DQNAgent(encode_observation(observation, config, RULE_FEATURES).size,
                         env.action_count, train, env_config=config)
        action = ExtrapolatePursuit(config).choose_action(observation, np.ones(17, dtype=bool))
        agent.remember(observation, action, 3.0, observation, True, np.ones(17, dtype=bool))
        for _ in range(400):
            agent.update()
        features = torch.as_tensor(agent._features(observation))
        self.assertAlmostEqual(float(agent.online(features)[action]), 3.0, delta=0.05)
        # 选动作时其余动作仍比原始输出低 5 分。
        q = agent._q(agent.online, features[None])[0]
        other = (action + 1) % 17
        self.assertAlmostEqual(float(q[other]), float(agent.online(features)[other]) - 5.0, places=4)


class LegacyPriorModeTests(unittest.TestCase):
    def test_init_mode_rejected_only_when_prior_enabled(self):
        """旧模型记录了 rule_prior_mode=init：rule_prior=0 时先验未启用，应可加载；>0 时拒绝。"""
        import tempfile
        from pathlib import Path

        import torch
        config = EnvConfig(observation_delay=2)
        env = GridWorldEnv(config)
        observation, _ = env.reset(seed=7)
        for prior in (0.0, 2.0):
            train = TrainConfig(use_motion_prediction=False, observation_features=RULE_FEATURES,
                                rule_prior=prior, hidden_sizes=(8,))
            agent = DQNAgent(encode_observation(observation, config, RULE_FEATURES).size,
                             env.action_count, train, env_config=config)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "model.pt"
                agent.save(path, config)
                checkpoint = torch.load(path, weights_only=False)
                checkpoint["train_config"]["rule_prior_mode"] = "init"
                torch.save(checkpoint, path)
                if prior == 0.0:
                    DQNAgent.load(path)
                else:
                    with self.assertRaises(ValueError):
                        DQNAgent.load(path)


if __name__ == "__main__":
    unittest.main()
