import argparse
import numpy as np
from dataclasses import asdict
import json
from pathlib import Path

from uav_demo.config import EnvConfig
from uav_demo.environment import (
    GridWorldEnv, preview_move,
)


def choose_action(observation, config, mask):
    state = observation
    if not (state["sensing_complete"] or state["sensing_failed"]) and not state["has_observation"]:
        return 0

    # 感知阶段追最近收到的车位；感知结束后去 UAV 终点。
    target = (
        state["uav_goal"]
        if (state["sensing_complete"] or state["sensing_failed"])
        else state["car_pos"]
    )

    best_action, best_distance = 0, float("inf")

    # 枚举现有 17 个动作：悬停 + 八方向 × 步长 1/2。
    for action in np.flatnonzero(mask):
        position, _, invalid = preview_move(
            state["uav_pos"],
            state["uav_speed"],
            int(action),
            config.grid_size,
        )
        if invalid:
            continue

        # 比较执行动作后，到当前目标的距离平方。
        distance = sum(
            (a - b) ** 2 for a, b in zip(position, target)
        )
        if distance < best_distance:
            best_distance = distance
            best_action = int(action)

    return best_action


def main():
    parser = argparse.ArgumentParser(description="追踪最近收到的车辆位置，首报前悬停，并保存动画")
    parser.add_argument("--seed", type=int, default=3_000_000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/simple_pursuit"))
    parser.add_argument("--sensing-mode", choices=("consecutive", "cumulative"), default="consecutive")
    parser.add_argument("--observation-delay", type=int, default=0)
    args = parser.parse_args()
    config = EnvConfig(
        sensing_radius=2.0,
        sensing_steps=5,
        sensing_mode=args.sensing_mode,
        observation_delay=args.observation_delay,
    )
    env = GridWorldEnv(config)
    observation, info = env.reset(seed=args.seed)
    records = [{**info, "action": None, "reward": 0.0}]
    total_reward = 0.0

    while True:
        action = choose_action(observation, config, env.valid_action_mask())
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        records.append({**info, "action": action, "reward": reward})

        print(
            f"step={info['time']} "
            f"uav={info['uav_pos']} observed_car={info['observed_car_pos']} "
            f"sensing={info['sensing_progress']}/{config.sensing_steps} "
            f"phase={info['phase']}"
        )

        if terminated or truncated:
            print("任务成功：", info["success"])
            break

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    episode = {
        "summary": {
            "policy": "greedy_pursuit",
            "observation_mode": "delayed_vehicle",
            "observation_delay": config.observation_delay,
            "known_car_goal": False,
            "learned_motion_prediction": False,
            "seed": args.seed,
            "success": info["success"],
            "sensing_complete": info["sensing_complete"],
            "sensing_failed": info["sensing_failed"],
            "uav_reached_goal": info["uav_reached_goal"],
            "steps": info["time"],
            "travel_distance": info["travel_distance"],
            "total_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
        },
        "config": asdict(config),
        "car_goal": list(env.car_goal),
        "car_trajectory": env.car_trajectory.tolist(),
        "records": records,
    }
    episode_path = output_dir / "episode.json"
    episode_path.write_text(json.dumps(episode, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Episode data: {episode_path}")
    from uav_demo.visualization import render_episode
    paths = render_episode(
        config, env.car_trajectory, env.car_goal, records, output_dir,
        policy="greedy_pursuit",
    )
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
