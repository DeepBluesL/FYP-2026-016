import unittest

import numpy as np
import torch

from uav_delay.config import Config
from uav_delay.evaluation import evaluate
from uav_delay.train import PPOConfig, advantages, torch_policy, train


class TorchPPOTests(unittest.TestCase):
    def test_gae_matches_hand_computation_and_stops_at_episode_end(self):
        ppo = PPOConfig(n_envs=1, n_steps=3, gamma=0.5, gae_lambda=0.5)
        buffer = dict(rewards=torch.tensor([[1.0], [2.0], [3.0]]),
                      values=torch.tensor([[0.5], [1.0], [1.5]]),
                      dones=torch.tensor([[0.0], [1.0], [0.0]]))
        adv, returns = advantages(buffer, torch.tensor([2.0]), ppo)
        # t=2：3 + 0.5×2 − 1.5 = 2.5；t=1 回合结束：2 − 1 = 1；t=0：(1 + 0.5×1 − 0.5) + 0.25×1 = 1.25
        np.testing.assert_allclose(adv.squeeze(-1), [1.25, 1.0, 2.5])
        np.testing.assert_allclose(returns.squeeze(-1), [1.75, 2.0, 4.0])

    def test_short_training_runs_end_to_end(self):
        torch.set_num_threads(1)
        ppo = PPOConfig(n_envs=2, n_steps=32, batch_size=16, n_epochs=2, log_std_init=-1.0)
        cfg = Config(residual=1.0)
        model, steps = train(4, cfg, 64, seed=0, ppo=ppo)
        self.assertEqual(steps, 64)
        result = evaluate(torch_policy(model), 4, cfg, range(3))
        self.assertEqual(result["episodes"], 3)


if __name__ == "__main__":
    unittest.main()
