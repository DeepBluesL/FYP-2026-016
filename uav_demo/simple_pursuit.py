import argparse
from dataclasses import asdict
import json
from pathlib import Path

from uav_demo.config import EnvConfig
from uav_demo.environment import GridWorldEnv, preview_move
from uav_demo.visualization import render_episode


def choose_action(env):
    # 感知阶段追当前车位；感知结束后去 UAV 终点。
    target = env.config.uav_goal if env.sensing_finished else env.car_pos

    best_action = 0
    best_distance = float("inf")

    # 枚举现有 17 个动作：悬停 + 八方向 × 步长 1/2。
    for action in range(env.action_count):
        position, _, invalid = preview_move(
            env.uav_pos,
            env.uav_speed,
            action,
            env.config.grid_size,
        )
        if invalid:
            continue

        # 比较执行动作后，到当前目标的距离平方。
        distance = sum(
            (a - b) ** 2 for a, b in zip(position, target)
        )
        if distance < best_distance:
            best_distance = distance
            best_action = action

    return best_action


def main():
    parser = argparse.ArgumentParser(description="追踪车辆当前位置，无需训练，并保存动画")
    parser.add_argument("--seed", type=int, default=3_000_000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/simple_pursuit"))
    parser.add_argument("--sensing-mode", choices=("consecutive", "cumulative"), default="consecutive")
    args = parser.parse_args()
    config = EnvConfig(
        sensing_radius=2.0,
        sensing_steps=5,
        sensing_mode=args.sensing_mode,
    )
    env = GridWorldEnv(config)
    _, info = env.reset(seed=args.seed)
    records = [{**info, "action": None, "reward": 0.0}]
    total_reward = 0.0

    while True:
        action = choose_action(env)
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        records.append({**info, "action": action, "reward": reward})

        print(
            f"step={info['time']} "
            f"uav={info['uav_pos']} car={info['car_pos']} "
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
            "observation_mode": "current_state",
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
    paths = render_episode(
        config, env.car_trajectory, env.car_goal, records, output_dir,
        policy="greedy_pursuit",
    )
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
