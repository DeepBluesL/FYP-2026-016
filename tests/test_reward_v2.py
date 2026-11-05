"""简化奖励（reward_v2）：势函数塑形整局只差 −Φ(s0)；随机路线超时未采完按失败扣分。默认关闭时行为不变。"""

import unittest

import numpy as np

from uav_demo.config import EnvConfig
from uav_demo.environment import GridWorldEnv


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


if __name__ == "__main__":
    unittest.main()
