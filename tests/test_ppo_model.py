import unittest

import torch

from uav_demo.environment import GridWorldEnv
from uav_demo.observation import encode_observation
from uav_demo.ppo_model import PPOActorCritic


class PPOModelTests(unittest.TestCase):
    def test_architecture_matches_dqn(self):
        from uav_demo.model import QNetwork
        for activation in ("relu", "leaky_relu", "tanh", "silu"):
            model = PPOActorCritic(18, 17, (32, 16), activation, 0.2)
            dqn = QNetwork(18, 17, (32, 16), activation, 0.2)
            self.assertEqual(repr(model.actor), repr(dqn.layers))
            model.actor.load_state_dict(dqn.layers.state_dict())
            features = torch.randn(3, 18)
            torch.testing.assert_close(model.actor(features), dqn(features))
            self.assertEqual(model.critic[-1].out_features, 1)

    def test_sample_one_gridworld_step(self):
        env = GridWorldEnv()
        observation, _ = env.reset(seed=7)
        features = torch.from_numpy(encode_observation(observation, env.config))
        mask = torch.from_numpy(env.valid_action_mask())
        model = PPOActorCritic(features.numel(), env.action_count)

        policy = model.distribution(features, mask)
        self.assertTrue((~mask).any())
        self.assertTrue((policy.probs[~mask] == 0).all())
        torch.testing.assert_close(policy.probs[mask].sum(), torch.tensor(1.0))
        action, value, logp = model.step(features, mask)
        self.assertTrue(mask[action])
        self.assertAlmostEqual(logp, policy.log_prob(torch.tensor(action)).item())
        self.assertAlmostEqual(value, model.value(features).item())
        _, _, _, _, info = env.step(action)
        self.assertFalse(info["invalid_move"])

    def test_per_state_masks_and_value_batch_shape(self):
        model = PPOActorCritic(2, 3)
        features = torch.zeros(2, 2)
        masks = torch.tensor([[True, False, False], [False, False, True]])
        policy = model.distribution(features, masks)
        torch.testing.assert_close(policy.probs, masks.float())
        torch.testing.assert_close(policy.log_prob(torch.tensor([0, 2])), torch.zeros(2))
        self.assertEqual(model.value(features[:1]).shape, (1,))
