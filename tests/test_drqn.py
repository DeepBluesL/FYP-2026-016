"""DRQN 对照组：隐状态按局重置、整局回放与更新、保存加载、接入训练入口。"""

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from evaluate import evaluate_episode
from train import train
from uav_demo.config import EnvConfig, TrainConfig
from uav_demo.drqn import DRQNAgent, load_agent
from uav_demo.environment import GridWorldEnv
from uav_demo.observation import encode_observation

ENV = EnvConfig(car_route="random", observation_delay=4, max_steps=98, auto_return=True)


def make_agent(**changes):
    config = TrainConfig(use_motion_prediction=False, hidden_sizes=(8,), **changes)
    size = encode_observation(GridWorldEnv(ENV).reset(seed=0)[0], ENV, config.observation_features).size
    return DRQNAgent(size, GridWorldEnv.action_count, config, env_config=ENV, hidden_size=16)


def play(agent, env, seed, epsilon=0.0, learn=False):
    observation, _ = env.reset(seed=seed)
    mask = env.valid_action_mask()
    while True:
        action = agent.choose_action(observation, epsilon, mask)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        next_mask = env.valid_action_mask()
        if learn:
            agent.remember(observation, action, reward, next_observation, terminated or truncated, next_mask)
        if terminated or truncated:
            return
        observation, mask = next_observation, next_mask


class DRQNTests(unittest.TestCase):
    def test_hidden_state_resets_at_episode_start(self):
        agent, env = make_agent(), GridWorldEnv(ENV)
        first, _ = env.reset(seed=3)
        action = agent.choose_action(first)
        for _ in range(5):
            agent.choose_action(env.step(0)[0])
        self.assertEqual(agent.choose_action(env.reset(seed=3)[0]), action)

    def test_whole_episodes_are_stored_and_update_runs(self):
        agent, env = make_agent(), GridWorldEnv(ENV)
        play(agent, env, seed=1, epsilon=1.0, learn=True)
        episode = agent.replay.episodes[0]
        self.assertEqual(len(episode["features"]), len(episode["actions"]) + 1)
        self.assertEqual(len(agent.replay), len(episode["actions"]))
        loss = agent.update()
        self.assertTrue(np.isfinite(loss))

    def test_end_to_end_only(self):
        for changes in ({"rule_prior": 0.1, "observation_features": TrainConfig().observation_features + ("rule_action",)},
                        {"contact_prior": 0.1}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "端到端"):
                make_agent(**changes)

    def test_checkpoint_round_trip_and_dispatch(self):
        agent, env = make_agent(), GridWorldEnv(ENV)
        play(agent, env, seed=2, epsilon=1.0, learn=True)
        agent.update()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            agent.save(path, ENV)
            loaded, env_config = load_agent(path)
        self.assertIsInstance(loaded, DRQNAgent)
        self.assertEqual(env_config, ENV)
        rows = [evaluate_episode(model, GridWorldEnv(ENV), 7) for model in (agent, loaded)]
        self.assertEqual(rows[0]["steps"], rows[1]["steps"])

    def test_training_entrypoint(self):
        config = TrainConfig(episodes=2, eval_episodes=1, use_motion_prediction=False, agent_type="drqn",
                             learning_starts=10, hidden_sizes=(8,))
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(StringIO()):
            agent = train(ENV, config, folder)
            loaded, _ = load_agent(Path(folder) / "best_model.pt")
        self.assertIsInstance(agent, DRQNAgent)
        self.assertGreater(agent.gradient_steps, 0)
        self.assertIsInstance(loaded, DRQNAgent)


if __name__ == "__main__":
    unittest.main()
