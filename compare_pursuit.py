"""在相同固定起点车辆路线下，对比已有 DQN 与追当前位置的贪心策略。"""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from evaluate import check_test_seeds
from uav_demo.agent import DQNAgent
from uav_demo.environment import GridWorldEnv
from uav_demo.simple_pursuit import choose_action


def run_episode(agent, config, seed, policy):
    env = GridWorldEnv(config)
    observation, info = env.reset(seed=seed)
    records = [{**info, "action": None, "reward": 0.0}]
    total_reward = 0.0
    sensing_step = None
    while True:
        action = (choose_action(env) if policy == "greedy_pursuit" else
                  agent.choose_action(observation, epsilon=0.0, mask=env.valid_action_mask()))
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        records.append({**info, "action": action, "reward": reward})
        if info["sensing_complete"] and sensing_step is None:
            sensing_step = info["time"]
        if terminated or truncated:
            break
    row = {"policy": policy, "seed": seed, "success": info["success"],
           "sensing_complete": info["sensing_complete"], "sensing_failed": info["sensing_failed"],
           "uav_reached_goal": info["uav_reached_goal"], "steps": info["time"],
           "sensing_complete_step": sensing_step, "travel_distance": info["travel_distance"],
           "total_reward": total_reward, "terminated": terminated, "truncated": truncated}
    episode = {"summary": row, "config": asdict(config), "car_goal": list(env.car_goal),
               "car_trajectory": env.car_trajectory.tolist(), "records": records}
    return row, episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/current_prediction_20261022/wait_train/best_model.pt"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4_000_000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fixed_corner_comparison"))
    parser.add_argument("--training-context", choices=("random_start", "fixed_corner", "unknown"), default="unknown",
                        help="明确记录 checkpoint 的训练起点设置，不从文件名猜测")
    args = parser.parse_args()
    agent, config = DQNAgent.load(args.checkpoint, device="cpu")
    check_test_seeds(agent, args.seed, args.episodes)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    policies = ("dqn", "greedy_pursuit")
    for index, seed in enumerate(range(args.seed, args.seed + args.episodes)):
        paired_episodes = []
        for policy in policies:
            row, episode = run_episode(agent, config, seed, policy)
            rows.append(row)
            paired_episodes.append(episode)
            if index == 0:
                folder = output / policy
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "episode.json").write_text(json.dumps(episode, indent=2), encoding="utf-8")
        # 确认两种策略确实面对完全相同的路线，而不只是相同种子。
        assert paired_episodes[0]["car_trajectory"] == paired_episodes[1]["car_trajectory"]
        assert paired_episodes[0]["car_trajectory"][0] == [config.grid_size - 1, 0]
        if (index + 1) % 10 == 0:
            print(f"Completed {index + 1}/{args.episodes} paired routes", flush=True)

    metrics = {}
    for policy in policies:
        selected = [row for row in rows if row["policy"] == policy]
        successful = [row for row in selected if row["success"]]
        metrics[policy] = {
            "success_rate": sum(row["success"] for row in selected) / len(selected),
            "sensing_completion_rate": sum(row["sensing_complete"] for row in selected) / len(selected),
            "uav_goal_rate": sum(row["uav_reached_goal"] for row in selected) / len(selected),
            "mean_success_steps": float(np.mean([row["steps"] for row in successful])) if successful else None,
            "mean_success_distance": float(np.mean([row["travel_distance"] for row in successful])) if successful else None,
            "mean_reward": float(np.mean([row["total_reward"] for row in selected])),
        }
    dqn_rows = [row for row in rows if row["policy"] == "dqn"]
    greedy_rows = [row for row in rows if row["policy"] == "greedy_pursuit"]
    common = [(d, g) for d, g in zip(dqn_rows, greedy_rows) if d["success"] and g["success"]]
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "training_context": args.training_context,
        "retrained": args.training_context == "fixed_corner",
        "note": "Fixed-corner evaluation. Training context is explicitly supplied by the caller. DQN retains its learned motion predictor; greedy uses only current position.",
        "config": asdict(config), "actual_car_start": [config.grid_size - 1, 0],
        "episodes_per_policy": args.episodes, "seed_start": args.seed,
        "seed_end_inclusive": args.seed + args.episodes - 1, "metrics": metrics,
        "both_success_count": len(common),
        "paired_mean_steps_dqn_minus_greedy": float(np.mean([d["steps"] - g["steps"] for d, g in common])) if common else None,
        "paired_mean_distance_dqn_minus_greedy": float(np.mean([d["travel_distance"] - g["travel_distance"] for d, g in common])) if common else None,
    }
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    # 与现有 demo 一致：独立绘图进程避免 PyTorch / Matplotlib 运行库冲突。
    for policy in policies:
        subprocess.run([sys.executable, "-m", "uav_demo.visualization", "--episode",
                        str(output / policy / "episode.json")], check=True,
                       cwd=Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
