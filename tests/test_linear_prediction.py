"""线性外推预测器与外推规则给出同一估计，并能作为普通预测器接入 DQN 和保存加载。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from uav_demo.agent import DQNAgent
from uav_demo.config import EnvConfig, TrainConfig
from uav_demo.environment import GridWorldEnv
from uav_demo.extrapolate_pursuit import predicted_car
from uav_demo.linear_prediction import LinearPredictor
from uav_demo.observation import encode_observation
from uav_demo.prediction import forecast_vehicle


def make_agent(env, env_config, config):
    size = encode_observation(env.reset(seed=0)[0], env_config, config.observation_features).size
    return DQNAgent(size, env.action_count, config, motion_predictor=LinearPredictor(), env_config=env_config)


class LinearPredictorTests(unittest.TestCase):
    def test_next_step_matches_extrapolation_rule_along_random_routes(self):
        for every in (1, 2):
            config = EnvConfig(observation_delay=6, car_route="random", car_move_every=every, max_steps=300)
            env, predictor = GridWorldEnv(config), LinearPredictor()
            for seed in range(5):
                observation, _ = env.reset(seed=seed)
                for _ in range(60):
                    if observation["has_observation"]:
                        forecast = forecast_vehicle(predictor, observation, config, (1,))[0]
                        self.assertEqual(tuple(int(v) for v in forecast), predicted_car(observation, config))
                    observation, *_ = env.step(0)

    def test_rule_from_linear_predictor_equals_default_linear_rule(self):
        env_config = EnvConfig(observation_delay=4, car_route="random")
        env = GridWorldEnv(env_config)
        features = TrainConfig().observation_features + ("rule_action",)
        linear, shared = (make_agent(env, env_config, TrainConfig(rule_source=source, rule_prior=0.1,
                                                                  observation_features=features))
                          for source in ("linear", "predictor"))
        observation, _ = env.reset(seed=3)
        for _ in range(40):
            mask = env.valid_action_mask()
            if observation["has_observation"]:
                self.assertEqual(linear.rule_action(observation, mask), shared.rule_action(observation, mask))
            observation, *_ = env.step(int(np.flatnonzero(mask)[-1]))

    def test_checkpoint_round_trip_keeps_linear_predictor(self):
        env_config = EnvConfig(observation_delay=4)
        env = GridWorldEnv(env_config)
        agent = make_agent(env, env_config, TrainConfig(prediction_model_type="linear"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            agent.save(path, env_config)
            loaded, _ = DQNAgent.load(path)
        self.assertIsInstance(loaded.motion_predictor, LinearPredictor)
        observation, _ = env.reset(seed=0)
        for _ in range(10):
            observation, *_ = env.step(0)
        np.testing.assert_array_equal(agent.predict_car(observation), loaded.predict_car(observation))


if __name__ == "__main__":
    unittest.main()
