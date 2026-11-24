import tempfile
import unittest
import csv
import math
from dataclasses import replace
from unittest.mock import patch
from pathlib import Path

import torch

from train import train
from uav_demo.config import EnvConfig, TrainConfig
from uav_demo.environment import GridWorldEnv
from uav_demo.ppo import PPOAgent, RolloutStep, trajectory_targets
from uav_demo.agent import DQNAgent


class PPOTests(unittest.TestCase):
    def test_best_checkpoint_keeps_selection_progress_and_complete_run_range(self):
        from evaluate import check_test_seeds
        import json
        env = EnvConfig(grid_size=5, max_steps=8, sensing_steps=8, uav_goal=(4, 4))
        metrics = dict(success_rate=0.0, sensing_completion_rate=0.0,
                       sensing_failure_rate=1.0, uav_goal_rate=0.0, mean_reward=-1.0,
                       mean_success_steps="", mean_travel_distance=0.0)
        for algorithm in ("ppo", "dqn"):
            cfg = TrainConfig(algorithm=algorithm, use_motion_prediction=False, total_steps=19,
                              eval_every_steps=5, rollout_steps=5, train_pi_iters=1, train_v_iters=1,
                              hidden_sizes=(8,), eval_episodes=1, test_seed=4_000_000, test_episodes=20)
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as folder:
                with patch("train.evaluate", return_value=metrics):
                    agent = train(env, cfg, folder)
                best, _ = type(agent).load(Path(folder) / "best_model.pt")
                metadata = best.run_metadata
                self.assertEqual(best.environment_steps, 5)
                self.assertEqual(metadata["selected_at_environment_steps"], 5)
                self.assertEqual(metadata["selected_at_episode"], 1)
                self.assertEqual(metadata["run_train_seed_start"], 42)
                self.assertEqual(metadata["run_train_seed_end_exclusive"], 45)
                self.assertTrue(metadata["run_complete"])
                sidecar = json.loads((Path(folder) / "run_metadata.json").read_text())
                self.assertEqual(sidecar["best_model.pt"], metadata)
                with self.assertRaisesRegex(ValueError, "重叠"):
                    check_test_seeds(best, 43, 1)
                with self.assertRaisesRegex(ValueError, "固定测试"):
                    check_test_seeds(best, 3_000_000, 1)
                check_test_seeds(best, 4_000_000, 20)
                with self.assertRaisesRegex(ValueError, "固定测试"):
                    check_test_seeds(best, 4_000_019, 2)

    def test_validation_alignment_and_reserved_test_range(self):
        env = EnvConfig(grid_size=5, max_steps=8, sensing_steps=8, uav_goal=(4, 4))
        cfg = TrainConfig(use_motion_prediction=False, total_steps=19, rollout_steps=5,
                          eval_every_steps=6, hidden_sizes=(8,), eval_episodes=1)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "整数倍"):
                train(env, cfg, folder)
            for seed in (42, cfg.eval_seed):
                with self.assertRaisesRegex(ValueError, "固定测试种子"):
                    train(env, replace(cfg, eval_every_steps=5, test_seed=seed), folder)
            agent = train(env, replace(cfg, eval_every_steps=None), folder)
            self.assertEqual(agent.config.eval_every_steps, 50)

    def test_independent_iterations_and_critic_metrics(self):
        env = GridWorldEnv()
        obs, _ = env.reset(seed=7)
        cfg = TrainConfig(use_motion_prediction=False, hidden_sizes=(8,),
                          train_pi_iters=1, train_v_iters=3, target_kl=100)
        agent = PPOAgent(18, 17, cfg, env_config=env.config)
        for _ in range(12):
            obs, _, _, _ = agent.collect_step(env, obs, env.valid_action_mask())
        batch = list(agent.rollout)
        features = torch.stack([s.features for s in batch])
        with torch.no_grad():
            last_value = agent.model.value(agent.features(obs)).item()
            _, returns = trajectory_targets(batch, last_value, cfg.gamma, cfg.gae_lambda)
            targets = torch.tensor(returns)
            before = agent.model.value(features)
            expected_loss = ((before - targets) ** 2).mean().item()
            expected_ev = (1 - (targets - before).var(unbiased=False) /
                           targets.var(unbiased=False)).item()
        metrics = agent.update(obs)
        self.assertEqual(metrics["policy_updates"], 1)
        self.assertEqual(metrics["value_updates"], 3)
        self.assertEqual(agent.gradient_steps, 4)
        self.assertAlmostEqual(metrics["value_loss_before"], expected_loss)
        self.assertAlmostEqual(metrics["explained_variance"], expected_ev)
        with torch.no_grad():
            after = ((agent.model.value(features) - targets) ** 2).mean().item()
        self.assertAlmostEqual(metrics["value_loss_after"], after)
        self.assertEqual(metrics["value_loss"], metrics["value_loss_after"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            agent.save(path, env.config)
            restored, _ = PPOAgent.load(path)
            self.assertEqual((restored.config.train_pi_iters, restored.config.train_v_iters), (1, 3))
            # 模拟独立字段出现之前的 checkpoint。
            saved = torch.load(path, weights_only=True)
            saved["train_config"].pop("train_pi_iters")
            saved["train_config"].pop("train_v_iters")
            saved["train_config"]["update_iters"] = 7
            torch.save(saved, path)
            restored, _ = PPOAgent.load(path)
            self.assertEqual((restored.config.train_pi_iters, restored.config.train_v_iters), (7, 7))

    def test_constant_returns_have_undefined_explained_variance(self):
        env = GridWorldEnv()
        obs, _ = env.reset(seed=7)
        cfg = TrainConfig(use_motion_prediction=False, hidden_sizes=(8,), update_iters=1)
        agent = PPOAgent(18, 17, cfg, env_config=env.config)
        obs, _, _, _ = agent.collect_step(env, obs, env.valid_action_mask())
        agent.rollout[-1].done = True
        agent.rollout[-1].reward = 0.0
        metrics = agent.update(obs)
        self.assertTrue(math.isnan(metrics["explained_variance"]))
        self.assertTrue(math.isfinite(metrics["value_loss_after"]))

    def test_agent_defaults_do_not_mutate_shared_configuration(self):
        cfg = TrainConfig(use_motion_prediction=False, hidden_sizes=(8,))
        dqn = DQNAgent(18, 17, cfg, env_config=EnvConfig())
        ppo = PPOAgent(18, 17, cfg, env_config=EnvConfig())
        self.assertIsNone(cfg.activation)
        self.assertEqual(cfg.algorithm, "dqn")
        self.assertIsInstance(dqn.online.layers[1], torch.nn.LeakyReLU)
        self.assertIsInstance(ppo.model.actor[1], torch.nn.Tanh)
        self.assertEqual(ppo.config.algorithm, "ppo")
        explicit = PPOAgent(18, 17, replace(cfg, activation="relu"), env_config=EnvConfig())
        self.assertIsInstance(explicit.model.actor[1], torch.nn.ReLU)

    def test_metrics_describe_final_policy(self):
        env = GridWorldEnv()
        obs, _ = env.reset(seed=7)
        agent = PPOAgent(18, 17, TrainConfig(use_motion_prediction=False, hidden_sizes=(8,),
                         update_iters=1, policy_lr=0.01), env_config=env.config)
        for _ in range(12):
            obs, _, _, _ = agent.collect_step(env, obs, env.valid_action_mask())
        batch = list(agent.rollout)
        metrics = agent.update(obs)
        with torch.no_grad():
            policy = agent.model.distribution(torch.stack([s.features for s in batch]),
                                              torch.stack([s.mask for s in batch]))
            logp = policy.log_prob(torch.tensor([s.action for s in batch]))
            kl = (torch.tensor([s.logp for s in batch]) - logp).mean().item()
        self.assertAlmostEqual(metrics["approx_kl"], kl)
        self.assertGreater(abs(kl), 1e-6)
        self.assertEqual(metrics["policy_updates"], 1)
        self.assertEqual(metrics["batch_steps"], 12)

    def test_equal_step_budget_validation_and_partial_rollout(self):
        env = EnvConfig(grid_size=5, max_steps=8, uav_goal=(4, 4), sensing_steps=8)
        for algorithm in ("dqn", "ppo"):
            cfg = TrainConfig(algorithm=algorithm, use_motion_prediction=False, episodes=1,
                              total_steps=19, eval_every_steps=10, eval_episodes=1,
                              rollout_steps=5, update_iters=1, hidden_sizes=(8,),
                              learning_starts=2, batch_size=2, train_every=1)
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as folder:
                agent = train(env, cfg, folder)
                with (Path(folder) / "eval.csv").open() as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual([int(r["environment_steps"]) for r in rows], [10, 19])
                with (Path(folder) / "training.csv").open() as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual(int(rows[-1]["episode_complete"]), 0)
                self.assertEqual(agent.environment_steps, 19)
                self.assertEqual(agent.config.episodes, 3)
                loaded, _ = type(agent).load(Path(folder) / "model.pt")
                self.assertEqual(loaded.environment_steps, 19)
                self.assertEqual(loaded.config.episodes, 3)
                if algorithm == "ppo":
                    with (Path(folder) / "updates.csv").open() as file:
                        rows = list(csv.DictReader(file))
                    self.assertEqual([int(r["batch_steps"]) for r in rows], [5, 5, 5, 4])
                    self.assertEqual([int(r["environment_steps"]) for r in rows], [5, 10, 15, 19])
                    self.assertEqual(agent.rollout, [])

    def test_terminal_and_rollout_cut_in_same_batch(self):
        # 第一局奖励 1, 2，价值 1, 2；第二局走一步就收满，下一状态价值 4。
        def step(value, reward, done):
            return RolloutStep(torch.zeros(1), torch.ones(1, dtype=torch.bool),
                               0, value, 0.0, reward, done)
        batch = [step(1, 1, False), step(2, 2, True), step(3, 4, False)]
        adv, returns = trajectory_targets(batch, last_value=4, gamma=0.5, gae_lambda=0.5)
        self.assertEqual(adv, [1, 0, 3])
        self.assertEqual(returns, [2, 2, 6])
        batch[1].reward = 4
        adv, returns = trajectory_targets(batch, 4, 0.5, 0.5)
        self.assertEqual(adv, [1.5, 2, 3])
        self.assertEqual(returns, [3, 4, 6])

    def test_update_uses_sampled_policy_and_changes_both_networks(self):
        env = GridWorldEnv()
        obs, _ = env.reset(seed=7)
        cfg = TrainConfig(algorithm="ppo", use_motion_prediction=False, update_iters=2)
        agent = PPOAgent(18, env.action_count, cfg, env_config=env.config)
        for _ in range(8):
            obs, _, _, _ = agent.collect_step(env, obs, env.valid_action_mask())
        batch = agent.rollout
        policy = agent.model.distribution(torch.stack([s.features for s in batch]),
                                          torch.stack([s.mask for s in batch]))
        ratio = (policy.log_prob(torch.tensor([s.action for s in batch]))
                 - torch.tensor([s.logp for s in batch])).exp()
        torch.testing.assert_close(ratio, torch.ones_like(ratio))
        actor = agent.model.actor[0].weight.detach().clone()
        critic = agent.model.critic[0].weight.detach().clone()
        agent.update(obs)
        self.assertFalse(torch.equal(actor, agent.model.actor[0].weight))
        self.assertFalse(torch.equal(critic, agent.model.critic[0].weight))
        self.assertEqual(agent.rollout, [])

    def test_train_flushes_partial_batch_and_restores_policy(self):
        env_config = EnvConfig(grid_size=5, max_steps=8, uav_goal=(4, 4), sensing_steps=8)
        cfg = TrainConfig(algorithm="ppo", use_motion_prediction=False, episodes=2,
                          rollout_steps=5, update_iters=1, eval_episodes=1,
                          hidden_sizes=(32, 16), activation="silu")
        with tempfile.TemporaryDirectory() as folder:
            agent = train(env_config, cfg, folder)
            saved, _ = PPOAgent.load(Path(folder) / "model.pt")
            import json
            counts = json.loads((Path(folder) / "parameter_counts.json").read_text())
            self.assertEqual(counts, saved.parameter_counts())
            self.assertEqual(saved.config.activation, "silu")
            self.assertEqual(saved.config.hidden_sizes, (32, 16))
        self.assertEqual(agent.environment_steps, 16)
        self.assertEqual(saved.gradient_steps, 8)  # 5+5+5+1，每批 actor/critic 各更新一次
        self.assertEqual(agent.rollout, [])
        for before, after in zip(agent.model.parameters(), saved.model.parameters()):
            torch.testing.assert_close(before, after)
