"""验证训练预热与训练/验证种子边界。"""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

from train import train
from uav_demo.config import EnvConfig, TrainConfig


class TrainingTests(unittest.TestCase):
    def test_warmup_can_exceed_replay_capacity(self):
        env_config = EnvConfig(
            grid_size=5, max_steps=8, uav_goal=(4, 4), sensing_steps=8,
        )
        config = TrainConfig(
            episodes=1, use_motion_prediction=False, batch_size=2,
            replay_capacity=4, learning_starts=6, train_every=1,
            eval_episodes=1, hidden_sizes=(8,),
        )
        root = (Path(__file__).resolve().parents[1] / "outputs" / "tests").resolve()
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(root))
            with redirect_stdout(StringIO()):
                agent = train(env_config, config, folder)
        self.assertEqual(agent.environment_steps, 8)
        self.assertEqual(agent.gradient_steps, 3)
        self.assertEqual(len(agent.replay), 4)

    def test_prediction_seed_ranges_are_independent_of_dqn_episode_count(self):
        config = TrainConfig(episodes=1, seed=42, eval_seed=43)
        self.assertEqual(config.prediction_routes, 1000)
        for changes in ({"prediction_seed": 42}, {"prediction_seed": 1_000_000},
                        {"prediction_seed": 41, "prediction_routes": 10}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "重叠"):
                TrainConfig(**changes)
        TrainConfig(episodes=1, prediction_seed=0, prediction_routes=10)


if __name__ == "__main__":
    unittest.main()
