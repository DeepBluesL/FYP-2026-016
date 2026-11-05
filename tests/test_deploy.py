"""deploy/ 接口：只靠 ObservationBuilder 构造观测时，掩码和动作与原评估代码逐步一致。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))

from check import run_episode  # noqa: E402
from policy import ObservationBuilder, Policy  # noqa: E402
from uav_demo.environment import GridWorldEnv  # noqa: E402


class DeployTests(unittest.TestCase):
    def test_actions_match_original_evaluation(self):
        for delay in (4, 6, 8):
            policy = Policy(delay=delay)
            env = GridWorldEnv(policy.config)
            for seed in range(3_000_000, 3_000_005):
                with self.subTest(delay=delay, seed=seed):
                    self.assertEqual(run_episode(policy, env, seed)[2], 0)

    def test_builder_before_first_report_and_velocity(self):
        policy = Policy(delay=6)
        builder = ObservationBuilder(policy.config)
        observation = builder.build(3, (2, 2), 1, 0, False)
        self.assertFalse(observation["has_observation"])
        self.assertIsNone(observation["car_pos"])
        policy.act(observation)  # 首报前也能给出动作（规则为悬停）
        for t, position in enumerate([(10, 5), (11, 5), (11, 5)]):
            builder.report(t, position)
        observation = builder.build(8, (2, 2), 1, 0, False)
        self.assertEqual(observation["car_velocity"], (1, 0))  # 最近一次非零位移
        self.assertEqual(observation["observation_age"], 6)

    def test_projectairsim_example_grid_backend_matches_reference(self):
        import asyncio
        import json
        from projectairsim_example import ROUTES, GridBackend, run_episode
        data = json.loads(ROUTES.read_text(encoding="utf-8"))
        for delay in (4, 8):
            policy = Policy(delay=delay)
            reference = data["reference"][str(delay)]["cv_prediction_plus_rl"]
            for route, ref in list(zip(data["routes"], reference))[:10]:
                result = asyncio.run(run_episode(policy, GridBackend(), [tuple(p) for p in route["car"]]))
                with self.subTest(delay=delay, seed=route["seed"]):
                    self.assertEqual(result, ref)

    def test_delay_must_match_model(self):
        with self.assertRaises(ValueError):
            Policy(Path(__file__).resolve().parents[1] / "deploy" / "models" / "d6.pt", delay=4)


if __name__ == "__main__":
    unittest.main()
