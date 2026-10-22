"""验证 DQN 的学习、动作边界和模型保存，而不把短训练误认为收敛。"""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from uav_demo.agent import DQNAgent
from uav_demo.config import EnvConfig, TrainConfig
from uav_demo.environment import GridWorldEnv


class DQNTests(unittest.TestCase):
    def config(self, **changes):
        return replace(
            TrainConfig(batch_size=2, replay_capacity=16, learning_starts=0, hidden_sizes=(8, 8, 8)),
            **changes,
        )

    def temporary_directory(self):
        root = (Path(__file__).resolve().parents[1] / "outputs" / "tests").resolve()
        root.mkdir(parents=True, exist_ok=True)
        folder = tempfile.TemporaryDirectory(dir=root)
        self.assertTrue(Path(folder.name).resolve().is_relative_to(root))
        return folder

    def test_masked_actions_and_greedy_does_not_consume_randomness(self):
        agent = DQNAgent(2, 3, self.config())
        with torch.no_grad():
            for parameter in agent.online.parameters():
                parameter.zero_()
            agent.online.layers[-1].bias.copy_(torch.tensor([100.0, 2.0, 1.0]))
        observation = np.zeros(2, dtype=np.float32)
        mask = np.asarray([False, True, True])
        rng_before = deepcopy(agent.rng.bit_generator.state)
        self.assertEqual(agent.choose_action(observation, 0.0, mask), 1)
        self.assertEqual(agent.rng.bit_generator.state, rng_before)
        for _ in range(20):
            self.assertIn(agent.choose_action(observation, 1.0, mask), (1, 2))

    def test_target_ignores_invalid_next_actions_and_terminal_bootstrap(self):
        agent = DQNAgent(2, 3, self.config(gamma=0.5))
        with torch.no_grad():
            for network in (agent.online, agent.target):
                for parameter in network.parameters():
                    parameter.zero_()
            agent.target.layers[-1].bias.copy_(torch.tensor([100.0, 2.0, 0.0]))
        observation = np.zeros(2, dtype=np.float32)
        mask = np.asarray([False, True, True])
        agent.remember(observation, 1, 1.0, observation, False, mask)
        agent.remember(observation, 1, 1.0, observation, True, mask)
        # 非终止 target=1+.5*2=2；终止 target=1。Huber loss 均值为 1。
        self.assertAlmostEqual(agent.update(), 1.0, places=6)

    def test_double_dqn_selects_online_action_but_uses_target_value(self):
        agent = DQNAgent(2, 3, self.config(gamma=0.5))
        with torch.no_grad():
            for network in (agent.online, agent.target):
                for parameter in network.parameters():
                    parameter.zero_()
            # online 选择合法动作 1；target 的最大值在 2，不能用它来选动作。
            agent.online.layers[-1].bias.copy_(torch.tensor([100.0, 3.0, 0.0]))
            agent.target.layers[-1].bias.copy_(torch.tensor([100.0, 2.0, 10.0]))
        observation = np.zeros(2, dtype=np.float32)
        mask = np.asarray([False, True, True])
        for _ in range(2):
            agent.remember(observation, 2, 1.0, observation, False, mask)
        # 当前 Q(a=2)=0，target=1+.5*2=2，Huber loss=1.5。
        self.assertAlmostEqual(agent.update(), 1.5, places=6)

    def test_gradient_update_changes_weights_and_syncs_target(self):
        agent = DQNAgent(2, 3, self.config(target_update_every=1))
        observation = np.ones(2, dtype=np.float32)
        mask = np.ones(3, dtype=bool)
        for _ in range(2):
            agent.remember(observation, 1, 5.0, observation, True, mask)
        before = {name: value.clone() for name, value in agent.online.state_dict().items()}
        loss = agent.update()
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(agent.gradient_steps, 1)
        self.assertTrue(any(not torch.equal(before[name], value) for name, value in agent.online.state_dict().items()))
        for name, value in agent.online.state_dict().items():
            self.assertTrue(torch.equal(value, agent.target.state_dict()[name]))

    def test_checkpoint_roundtrip_preserves_policy_and_configuration(self):
        env_config = EnvConfig(sensing_mode="cumulative")
        env = GridWorldEnv(env_config)
        observation, _ = env.reset(seed=789)
        architecture = self.config(hidden_sizes=(12, 7), activation="tanh", leaky_relu_slope=0.2)
        agent = DQNAgent(env.observation_size, env.action_count, architecture)
        mask = env.valid_action_mask()
        with torch.no_grad():
            original_q = agent.online(torch.from_numpy(observation)).clone()
        # 在仓库 outputs 内创建临时目录，兼容 Windows 沙箱的可写范围。
        temporary_root = (Path(__file__).resolve().parents[1] / "outputs" / "tests").resolve()
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(temporary_root))
            path = Path(folder) / "model.pt"
            agent.save(path, env_config)
            loaded, loaded_env_config = DQNAgent.load(path)
        self.assertEqual(loaded_env_config, env_config)
        self.assertEqual(loaded.config.hidden_sizes, (12, 7))
        self.assertEqual(loaded.config.activation, "tanh")
        self.assertEqual(loaded.config.leaky_relu_slope, 0.2)
        self.assertEqual(loaded.choose_action(observation, mask=mask), agent.choose_action(observation, mask=mask))
        with torch.no_grad():
            self.assertTrue(torch.equal(original_q, loaded.online(torch.from_numpy(observation))))

    def test_prediction_features_and_policy_survive_checkpoint_roundtrip(self):
        from uav_demo.prediction import MotionPredictor

        env = GridWorldEnv()
        observation, _ = env.reset(seed=789)
        predictor = MotionPredictor()
        agent = DQNAgent(env.observation_size, env.action_count, self.config(),
                         motion_predictor=predictor, env_config=env.config)
        self.assertEqual(agent.network_input_size, 26)
        features = agent._features(observation)
        self.assertEqual(features.shape, (26,))
        np.testing.assert_array_equal(features[:20], observation)
        forecast = agent.predict_car(observation)
        np.testing.assert_allclose(
            features[20:].reshape(3, 2), forecast / (env.config.grid_size - 1) - observation[:2],
        )
        with torch.no_grad():
            original_q = agent.online(torch.from_numpy(features)).clone()
        with self.temporary_directory() as folder:
            path = Path(folder) / "forecast_model.pt"
            agent.save(path, env.config)
            loaded, restored_config = DQNAgent.load(path)
        self.assertEqual(restored_config, env.config)
        self.assertEqual(loaded.network_input_size, 26)
        np.testing.assert_array_equal(loaded.predict_car(observation), forecast)
        with torch.no_grad():
            torch.testing.assert_close(loaded.online(torch.from_numpy(loaded._features(observation))), original_q)
        self.assertEqual(loaded.choose_action(observation, mask=env.valid_action_mask()),
                         agent.choose_action(observation, mask=env.valid_action_mask()))
        # 修改不可见的未来路线，不影响相同当前观测的预测和决策。
        env.car_trajectory[1:] = (0, 0)
        np.testing.assert_array_equal(loaded._features(env._observation()), features)

    def test_legacy_checkpoints_require_retraining(self):
        with self.temporary_directory() as folder:
            path = Path(folder) / "legacy.pt"
            for action_count in (9, 17):
                with self.subTest(action_count=action_count):
                    # v1 未保存动作表；即使动作数量相同也不能证明语义一致。
                    torch.save({"format_version": 1, "action_count": action_count}, path)
                    with self.assertRaisesRegex(ValueError, "旧版模型.*重新训练"):
                        DQNAgent.load(path)

    def test_v2_v3_checkpoints_rejected_for_old_observation_semantics(self):
        env = GridWorldEnv()
        agent = DQNAgent(env.observation_size, env.action_count, self.config())
        with self.temporary_directory() as folder:
            path = Path(folder) / "legacy.pt"
            agent.save(path, env.config)
            checkpoint = torch.load(path, weights_only=True)
            for version in (2, 3):
                with self.subTest(version=version):
                    checkpoint["format_version"] = version
                    torch.save(checkpoint, path)
                    with self.assertRaisesRegex(ValueError, "旧版模型.*重新训练"):
                        DQNAgent.load(path)

    def test_old_v4_reward_fields_keep_original_environment_semantics(self):
        env = GridWorldEnv()
        agent = DQNAgent(env.observation_size, env.action_count, self.config())
        with self.temporary_directory() as folder:
            path = Path(folder) / "legacy_reward_v4.pt"
            agent.save(path, env.config)
            checkpoint = torch.load(path, weights_only=True)
            del checkpoint["env_config"]["progress_to_contact_boundary"]
            del checkpoint["env_config"]["movement_penalty"]
            torch.save(checkpoint, path)
            _, restored = DQNAgent.load(path)
        self.assertEqual(restored, replace(env.config, progress_to_contact_boundary=False,
                                          movement_penalty=0.0))

    def test_checkpoint_validates_action_semantics_observation_and_structure(self):
        env = GridWorldEnv()
        agent = DQNAgent(env.observation_size, env.action_count, self.config())
        with self.temporary_directory() as folder:
            path = Path(folder) / "model.pt"
            agent.save(path, env.config)
            original = torch.load(path, weights_only=True)
            self.assertEqual(original["format_version"], 4)
            cases = (
                ("count", "action_count", 9, "动作数量"),
                ("reordered_names", "action_names", original["action_names"][::-1], "动作数量"),
                ("reordered_vectors", "actions", original["actions"][::-1], "动作数量"),
                ("observation", "observation_size", env.observation_size + 1, "观测维度"),
                ("observation_mode", "observation_mode", "oracle", "观测模式"),
                ("prediction_features", "network_input_size", 26, "预测特征维度"),
            )
            for name, key, value, message in cases:
                with self.subTest(case=name):
                    invalid = deepcopy(original)
                    invalid[key] = value
                    torch.save(invalid, path)
                    with self.assertRaisesRegex(ValueError, message):
                        DQNAgent.load(path)

            # 参数形状与记录的层宽不匹配时，明确报告结构问题。
            invalid = deepcopy(original)
            invalid["train_config"]["hidden_sizes"] = (16, 8, 8)
            torch.save(invalid, path)
            with self.assertRaisesRegex(ValueError, "权重.*网络结构"):
                DQNAgent.load(path)

            # 激活函数没有 state_dict 参数，必须靠保存的配置恢复，不能缺省。
            invalid = deepcopy(original)
            del invalid["train_config"]["activation"]
            torch.save(invalid, path)
            with self.assertRaisesRegex(ValueError, "网络或训练配置"):
                DQNAgent.load(path)

    def test_save_rejects_model_environment_dimension_mismatch(self):
        with self.temporary_directory() as folder:
            path = Path(folder) / "invalid.pt"
            for observation_size, action_count in ((GridWorldEnv().observation_size, 9), (2, 17)):
                with self.subTest(observation_size=observation_size, action_count=action_count):
                    agent = DQNAgent(observation_size, action_count, self.config())
                    with self.assertRaisesRegex(ValueError, "不能保存"):
                        agent.save(path, EnvConfig())
                    self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
