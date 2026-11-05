"""导出测试路线供仿真平台回放：每条路线车每秒的格子坐标，以及我们在这条路线上的参考结果。

车辆轨迹只由路线种子决定，与延迟无关；参考结果按延迟 4 / 6 / 8 各算一份（deploy 模型和 CV prediction）。
用法：python deploy/export_routes.py [--episodes 200] [--output deploy/routes/test_routes.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy import Policy  # noqa: E402
from uav_demo.environment import GridWorldEnv  # noqa: E402
from uav_demo.extrapolate_pursuit import ExtrapolatePursuit  # noqa: E402


def run(env: GridWorldEnv, seed: int, choose) -> dict:
    observation, _ = env.reset(seed=seed)
    sensed_at = None
    while True:
        observation, _, terminated, truncated, info = env.step(choose(observation, env.valid_action_mask()))
        if sensed_at is None and info["sensing_complete"]:
            sensed_at = env.time
        if terminated or truncated:
            return {"success": bool(info["success"]), "steps": env.time, "sensed_at": sensed_at}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3_000_000)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "routes" / "test_routes.json")
    args = parser.parse_args()

    policies = {delay: Policy(delay=delay) for delay in (4, 6, 8)}
    config = policies[6].config
    seeds = list(range(args.seed, args.seed + args.episodes))
    routes = []
    for seed in seeds:
        env = GridWorldEnv(config)
        env.reset(seed=seed)
        routes.append({"seed": seed, "car": env.car_trajectory.tolist()})

    reference = {}
    for delay, policy in policies.items():
        env = GridWorldEnv(policy.config)
        rule = ExtrapolatePursuit(policy.config)
        reference[str(delay)] = {
            "cv_prediction_plus_rl": [run(env, s, lambda o, m: policy.agent.choose_action(o, 0.0, m)) for s in seeds],
            "cv_prediction": [run(env, s, rule.choose_action) for s in seeds],
        }

    data = {
        "description": "Car grid cell (x east, y north; 1 cell = 5 m) at every second t = 0..150 for each test route, "
                       "and reference results of the deploy models (d4/d6/d8.pt) and CV prediction on the same routes.",
        "grid_size": config.grid_size, "cell_m": 5, "step_s": 1, "max_steps": config.max_steps,
        "uav_start": list(config.uav_start), "uav_goal": list(config.uav_goal),
        "routes": routes, "reference": reference,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    for delay, ref in reference.items():
        rl = ref["cv_prediction_plus_rl"]
        mean = sum(r["steps"] if r["success"] else config.max_steps for r in rl) / len(rl)
        print(f"d={delay}: success {sum(r['success'] for r in rl) / len(rl):.1%}, mean completion {mean:.1f} steps")
    print(f"saved {args.output} ({args.output.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
