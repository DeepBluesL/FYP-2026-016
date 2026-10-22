"""运行一个完整 episode，保存轨迹、指标与动画。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

from uav_demo.config import EnvConfig
from uav_demo.environment import GridWorldEnv
from uav_demo.planner import OraclePlanner


def main() -> None:
    parser = argparse.ArgumentParser(description="50x50 UAV / car：学习策略与独立 Oracle 对照")
    parser.add_argument("--policy", choices=("oracle", "dqn", "predictive"), default="dqn")
    parser.add_argument("--seed", type=int, default=3_000_000)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/current_prediction_20261022/wait_train/best_model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--sensing-mode", choices=("consecutive", "cumulative"))
    parser.add_argument("--no-gif", action="store_true", help="只保存静态图和 JSON")
    parser.add_argument("--show", action="store_true", help="同时显示交互式动画窗口")
    args = parser.parse_args()

    controller = None
    if args.policy in ("dqn", "predictive"):
        from uav_demo.agent import DQNAgent

        try:
            agent, config = DQNAgent.load(args.checkpoint, device="cpu")
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
        if args.sensing_mode and args.sensing_mode != config.sensing_mode:
            parser.error("DQN 演示使用模型保存的采集模式，请加载对应模式的模型")
        if args.policy == "predictive":
            from uav_demo.predictive_planner import PredictivePlanner
            if agent.motion_predictor is None:
                parser.error("预测拦截策略需要包含车辆运动预测器的 checkpoint")
            controller = PredictivePlanner(agent.motion_predictor, config)
    else:
        config = EnvConfig(sensing_mode=args.sensing_mode or EnvConfig().sensing_mode)
        agent = OraclePlanner()

    env = GridWorldEnv(config)
    observation, info = env.reset(seed=args.seed)
    records = [{**info, "action": None, "reward": 0.0}]
    if args.policy != "oracle":
        predicted = agent.predict_car(observation)
        records[-1]["car_prediction"] = predicted.tolist() if predicted is not None else None
    total_reward = 0.0
    while True:
        if args.policy == "oracle":
            action = agent.choose_action(env)
        elif controller is not None:
            action = controller.choose_action(observation, env.valid_action_mask())
        else:
            action = agent.choose_action(observation, epsilon=0.0, mask=env.valid_action_mask())
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        records.append({**info, "action": action, "reward": reward})
        if args.policy != "oracle":
            predicted = agent.predict_car(observation)
            records[-1]["car_prediction"] = predicted.tolist() if predicted is not None else None
        if terminated or truncated:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "policy": "oracle_heuristic" if args.policy == "oracle" else args.policy,
        "observation_mode": "full_route_oracle" if args.policy == "oracle" else "current_state",
        "learned_motion_prediction": args.policy != "oracle" and agent.motion_predictor is not None,
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
    }
    episode = {
        "summary": summary,
        "config": asdict(config),
        "car_goal": list(env.car_goal),
        "car_trajectory": env.car_trajectory.tolist(),
        "records": records,
    }
    episode_path = (args.output_dir / "episode.json").resolve()
    episode_path.write_text(
        json.dumps(episode, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Episode data: {episode_path}", flush=True)
    # 独立进程只加载 NumPy/Matplotlib，避免 Conda MKL 与 PyTorch 的
    # OpenMP 动态库冲突；保存的 JSON 也能独立用于重新绘制回放。
    command = [
        sys.executable, "-m", "uav_demo.visualization", "--episode", str(episode_path),
    ]
    if args.no_gif:
        command.append("--no-gif")
    if args.show:
        command.append("--show")
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
