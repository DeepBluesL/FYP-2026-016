"""接触概率（半径内概率和、各动作接触概率）、接触先验 ω、规则先验随训练减弱，以及接入训练与保存加载。"""

import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import torch

from evaluate import check_test_seeds
from train import train
from uav_demo.agent import DQNAgent
from uav_demo.config import EnvConfig, TrainConfig
from uav_demo.contact_prior import CONTACT_EPSILON, ContactModel, disk_sum
from uav_demo.environment import GridWorldEnv, preview_move
from uav_demo.grid_prediction import GridPredictor
from uav_demo.observation import encode_observation, feature_slice

FEATURES = TrainConfig().observation_features + ("rule_action",)


class PointMass:
    """测试用预测器：车下一步一定在 cell。"""
    needs_truth = False

    def __init__(self, cell, size=50):
        self.map = np.zeros((size, size))
        self.map[cell] = 1.0

    def distributions(self, history, grid_size, steps, move_every=1, goals=None):
        return np.stack([self.map for _ in steps])


def make_agent(env_config, config, contact_model):
    env = GridWorldEnv(env_config)
    size = encode_observation(env.reset(seed=0)[0], env_config, config.observation_features).size
    return DQNAgent(size, env.action_count, config, env_config=env_config, contact_model=contact_model)


def observed(env_config, steps=6, seed=2):
    env = GridWorldEnv(env_config)
    observation, _ = env.reset(seed=seed)
    for _ in range(steps):
        observation, *_ = env.step(0)
    return observation


class DiskSumTests(unittest.TestCase):
    def test_matches_brute_force_with_euclidean_radius(self):
        probability = np.random.default_rng(0).random((12, 12))
        for radius in (2.0, 1.5):
            summed = disk_sum(probability, radius)
            for x, y in ((0, 0), (5, 7), (11, 3)):
                expected = sum(probability[i, j] for i in range(12) for j in range(12)
                               if math.hypot(i - x, j - y) <= radius)
                self.assertAlmostEqual(summed[x, y], expected)


class ContactProbabilityTests(unittest.TestCase):
    def test_point_mass_gives_one_exactly_for_actions_ending_within_radius(self):
        config = EnvConfig(car_route="random", observation_delay=3)
        observation = observed(config)
        car = (observation["uav_pos"][0] + 2, observation["uav_pos"][1] + 1)
        probabilities = ContactModel(PointMass(car)).probabilities(observation, config)
        for action, p in enumerate(probabilities):
            position = preview_move(observation["uav_pos"], observation["uav_speed"], action, 50)[0]
            self.assertEqual(p, float(math.hypot(position[0] - car[0], position[1] - car[1]) <= 2))

    def test_zero_before_first_report_and_after_sensing(self):
        config = EnvConfig(car_route="random", observation_delay=3)
        model = ContactModel(PointMass((1, 1)))
        first, _ = GridWorldEnv(config).reset(seed=2)
        self.assertFalse(model.probabilities(first, config).any())
        finished = dict(observed(config), sensing_complete=True)
        self.assertFalse(model.probabilities(finished, config).any())

    def test_real_grid_probabilities_are_valid(self):
        config = EnvConfig(car_route="random", observation_delay=3)
        observation = observed(config, steps=12)
        probabilities = ContactModel(GridPredictor.base(config)).probabilities(observation, config)
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1 + 1e-9)))


class PriorTests(unittest.TestCase):
    def test_contact_prior_is_log_probability_gap_and_stays_out_of_the_network(self):
        env_config = EnvConfig(car_route="random", observation_delay=3)
        observation = observed(env_config)
        car = (observation["uav_pos"][0] + 3, observation["uav_pos"][1])
        model = ContactModel(PointMass(car))
        plain = make_agent(env_config, TrainConfig(use_motion_prediction=False), None)
        agent = make_agent(env_config, TrainConfig(use_motion_prediction=False, contact_prior=0.5), model)
        self.assertEqual(agent.network_input_size, plain.network_input_size)
        features = torch.as_tensor(agent._features(observation))
        self.assertEqual(features.numel(), agent.network_input_size + 17)
        p = model.probabilities(observation, env_config)
        log_p = np.log(p + CONTACT_EPSILON)
        with torch.no_grad():
            prior = (agent._q(agent.online, features) - agent.online(features[:agent.network_input_size])).numpy()
        np.testing.assert_allclose(prior, 0.5 * (log_p - log_p.max()), atol=1e-6)
        self.assertEqual(prior.max(), 0.0)

    def test_contact_probability_feature_goes_into_the_network(self):
        env_config = EnvConfig(car_route="random", observation_delay=3)
        observation = observed(env_config)
        model = ContactModel(PointMass((observation["uav_pos"][0] + 1, observation["uav_pos"][1])))
        features = TrainConfig().observation_features + ("contact_probability",)
        agent = make_agent(env_config, TrainConfig(use_motion_prediction=False, observation_features=features), model)
        values = agent._features(observation)
        self.assertEqual(values.size, agent.network_input_size)
        np.testing.assert_array_equal(values[feature_slice(features, "contact_probability")],
                                      model.probabilities(observation, env_config))

    def test_missing_contact_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "接触概率模型"):
            make_agent(EnvConfig(), TrainConfig(use_motion_prediction=False, contact_prior=0.1), None)

    def test_rule_prior_decays_linearly_to_zero(self):
        config = TrainConfig(use_motion_prediction=False, rule_prior=0.2, rule_prior_decay_steps=100,
                             observation_features=FEATURES)
        agent = make_agent(EnvConfig(), config, None)
        for steps, expected in ((0, 0.2), (50, 0.1), (100, 0.0), (500, 0.0)):
            agent.environment_steps = steps
            self.assertAlmostEqual(agent.current_rule_prior, expected)
        constant = make_agent(EnvConfig(), TrainConfig(use_motion_prediction=False, rule_prior=0.2,
                                                       observation_features=FEATURES), None)
        constant.environment_steps = 10_000
        self.assertEqual(constant.current_rule_prior, 0.2)


class IntegrationTests(unittest.TestCase):
    def test_configuration_flags(self):
        self.assertFalse(TrainConfig().uses_contact_model)
        self.assertTrue(TrainConfig(contact_prior=0.1).uses_contact_model)
        self.assertTrue(TrainConfig(contact_prior=0.1, contact_grid_turns="goal").needs_true_trajectory)
        self.assertFalse(TrainConfig(contact_prior=0.1).needs_true_trajectory)
        with self.assertRaisesRegex(ValueError, "重叠"):
            TrainConfig(use_motion_prediction=False, contact_prior=0.1, prediction_seed=0)
        for bad in ({"contact_prior": -1.0}, {"rule_prior_decay_steps": -5}, {"contact_grid_turns": "x"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                TrainConfig(**bad)

    def test_checkpoint_round_trip_keeps_contact_model(self):
        env_config = EnvConfig(car_route="random", observation_delay=4)
        config = TrainConfig(use_motion_prediction=False, contact_prior=0.2)
        agent = make_agent(env_config, config, ContactModel(GridPredictor("counted", (0.74, 0.11, 0.03, 0.12))))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            agent.save(path, env_config)
            loaded, _ = DQNAgent.load(path)
        np.testing.assert_allclose(loaded.contact_model.predictor.turn_probabilities, (0.74, 0.11, 0.03, 0.12))
        observation = observed(env_config, steps=10)
        np.testing.assert_array_equal(agent._features(observation), loaded._features(observation))
        with self.assertRaisesRegex(ValueError, "重叠"):
            check_test_seeds(loaded, config.prediction_seed, 1)

    def test_training_entrypoint_fits_contact_model(self):
        env_config = EnvConfig(car_route="random", max_steps=98)
        config = TrainConfig(episodes=1, eval_episodes=1, prediction_routes=10, prediction_seed=50_000,
                             use_motion_prediction=False, contact_prior=0.1, hidden_sizes=(8,))
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(StringIO()):
            agent = train(env_config, config, folder)
            metrics = json.loads((Path(folder) / "contact_metrics.json").read_text())
        self.assertEqual(metrics["turns"], "counted")
        self.assertIsInstance(agent.contact_model, ContactModel)


if __name__ == "__main__":
    unittest.main()
