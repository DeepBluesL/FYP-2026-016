"""独立测试 DQN 或学习模型拦截控制器，记录感知、到达及严格的等车指标。"""

import argparse
import csv
import json
from pathlib import Path

from uav_demo.agent import DQNAgent
from uav_demo.environment import GridWorldEnv


def check_test_seeds(agent, seed: int, episodes: int) -> None:
    """区间均为左闭右开；预测器内部留出路线也不能作为最终测试路线。"""
    if seed < 0 or episodes < 1:
        raise ValueError("测试 seed 必须非负，episodes 必须为正整数")
    cfg = agent.config
    ranges = [("DQN 训练", cfg.seed, cfg.seed + cfg.episodes),
              ("验证", cfg.eval_seed, cfg.eval_seed + cfg.eval_episodes)]
    if agent.motion_predictor is not None:
        count = cfg.prediction_route_count
        ranges.append(("车辆预测器训练/内部验证", cfg.seed, cfg.seed + count))
    for name, start, end in ranges:
        if seed < end and seed + episodes > start:
            raise ValueError(f"测试种子与{name}区间 [{start}, {end}) 重叠")


def evaluate_episode(agent, env, seed: int, controller=None) -> dict:
    observation, info = env.reset(seed=seed)
    if controller is not None:
        controller.reset()
    first_contact = 0 if info["contact"] else None
    sensing_complete_step, pre_contact_hover_steps, hover_streak = None, 0, 0
    early_goal = info["uav_reached_goal"] and not info["sensing_finished"]
    waiting_contact, total_reward = False, 0.0
    while True:
        old_position = tuple(info["uav_pos"])
        action = (agent.choose_action(observation, epsilon=0.0, mask=env.valid_action_mask())
                  if controller is None else controller.choose_action(observation, env.valid_action_mask()))
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        stationary_hover = action == 0 and tuple(info["uav_pos"]) == old_position
        if first_contact is None:
            if info["contact"]:
                first_contact = info["time"]
                waiting_contact = stationary_hover and hover_streak >= 2
            else:
                pre_contact_hover_steps += int(stationary_hover)
                hover_streak = hover_streak + 1 if stationary_hover else 0
        if info["sensing_complete"] and sensing_complete_step is None:
            sensing_complete_step = info["time"]
        early_goal |= info["uav_reached_goal"] and not info["sensing_finished"]
        if terminated or truncated:
            return {"seed": seed, "success": info["success"],
                    "sensing_failed": info["sensing_failed"], "sensing_complete": info["sensing_complete"],
                    "uav_reached_goal": info["uav_reached_goal"], "steps": info["time"],
                    "travel_distance": info["travel_distance"], "total_reward": total_reward,
                    "first_contact_step": first_contact, "sensing_complete_step": sensing_complete_step,
                    "early_goal_before_sensing": early_goal, "pre_contact_hover_steps": pre_contact_hover_steps,
                    "waiting_contact": waiting_contact}


def main() -> None:
    parser = argparse.ArgumentParser(description="当前状态策略的独立测试，不使用 Oracle 控制动作。")
    parser.add_argument("--policy", choices=("dqn", "predictive"), default="dqn")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/current_prediction_20261022/wait_train/best_model.pt"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3_000_000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/current_prediction_20261022/wait_test"))
    args = parser.parse_args()
    try:
        agent, config = DQNAgent.load(args.checkpoint)
        check_test_seeds(agent, args.seed, args.episodes)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    env = GridWorldEnv(config)
    controller = None
    if args.policy == "predictive":
        from uav_demo.predictive_planner import PredictivePlanner
        if agent.motion_predictor is None:
            parser.error("预测拦截策略需要车辆运动预测器")
        controller = PredictivePlanner(agent.motion_predictor, config)
    rows = [evaluate_episode(agent, env, seed, controller) for seed in range(args.seed, args.seed + args.episodes)]
    rate = lambda field: sum(bool(row[field]) for row in rows) / len(rows)
    successful = [row for row in rows if row["success"]]
    summary = {"policy": args.policy, "known_future": False, "checkpoint": str(args.checkpoint.resolve()),
               "motion_prediction": agent.motion_predictor is not None, "episodes": len(rows),
               "seed_start": args.seed, "seed_end_inclusive": args.seed + args.episodes - 1,
               "success_rate": rate("success"), "sensing_completion_rate": rate("sensing_complete"),
               "sensing_failure_rate": rate("sensing_failed"), "uav_goal_rate": rate("uav_reached_goal"),
               "early_goal_before_sensing_rate": rate("early_goal_before_sensing"),
               "waiting_contact_rate": rate("waiting_contact"),
               "successful_waiting_contact_count": sum(row["waiting_contact"] for row in successful),
               "mean_success_steps": sum(row["steps"] for row in successful) / len(successful) if successful else None,
               "mean_reward": sum(row["total_reward"] for row in rows) / len(rows),
               "mean_travel_distance": sum(row["travel_distance"] for row in rows) / len(rows)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
