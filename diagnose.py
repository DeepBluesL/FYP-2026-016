"""诊断实验：逐步记录预测误差、规则动作是否正确、DQN 是否跟随规则，按车辆是否机动分组汇总。

真实车辆位置只用于事后分析，不进入策略输入。分组和指标在跑之前固定：
- 机动：从报告前一步到下一步之间，车辆的逐步位移有变化（转弯、掉头、停走）；否则为直行。
- 规则估计误差：规则追的下一步车位（线性外推、预测器或真实车位，见 rule_source）与真实下一步车位的欧氏距离。
- 规则差距：规则动作移动后离真实下一步车位的距离，减去用真实车位算出的同一规则动作的距离（>0 即规则走偏）。
- 跟随：DQN 实际动作等于规则动作。
只统计感知阶段且已收到报告的步；采集结束后的返程由 auto_return 接管，不计入。
用法：python diagnose.py --checkpoints outputs/diag_v9/lstm_d4_seed*/best_model.pt --output outputs/diag_v9/report_lstm_d4
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from evaluate import check_test_seeds
from uav_demo.agent import DQNAgent
from uav_demo.environment import GridWorldEnv, preview_move


def distance(first, second) -> float:
    return float(np.hypot(first[0] - second[0], first[1] - second[1]))


def greedy_action(env, target) -> int:
    """与外推规则相同的选动作方式，但目标由调用者给出（这里传入真实下一步车位）。"""
    best_action, best = 0, float("inf")
    for action in np.flatnonzero(env.valid_action_mask()):
        position, _, invalid = preview_move(env.uav_pos, env.uav_speed, int(action), env.config.grid_size)
        if not invalid and (gap := sum((a - b) ** 2 for a, b in zip(position, target))) < best:
            best_action, best = int(action), gap
    return best_action


def after_move(env, action):
    return preview_move(env.uav_pos, env.uav_speed, action, env.config.grid_size)[0]


def maneuvering(trajectory, start: int, end: int) -> bool:
    """[start, end] 内相邻位移是否不同（转弯、掉头或停走）；start 应含规则用来估速度的那一步。"""
    window = trajectory[max(start, 0):end + 1]
    moves = [tuple(b - a) for a, b in zip(window, window[1:])]
    return any(a != b for a, b in zip(moves, moves[1:]))


def raw_q(agent, observation) -> np.ndarray:
    """不含规则先验的 Q 值，用来看 DQN 自己有多想偏离规则。"""
    with torch.no_grad():
        features = torch.as_tensor(agent._features(observation), device=agent.device).unsqueeze(0)
        return agent.online(features[:, :agent.network_input_size]).squeeze(0).cpu().numpy()


def run_episode(agent, env, seed: int):
    observation, info = env.reset(seed=seed)
    config, steps = env.config, []
    first_contact = None
    while True:
        mask = env.valid_action_mask()
        action = agent.choose_action(observation, epsilon=0.0, mask=mask)
        t = env.time
        if observation["has_observation"] and not info["sensing_finished"]:
            true_next = tuple(int(v) for v in env.car_trajectory[min(t + 1, config.max_steps)])
            estimate = agent.rule_target(observation)
            rule_action = agent.rule_action(observation, mask)
            oracle_action = greedy_action(env, true_next)
            q = raw_q(agent, observation)
            row = {"seed": seed, "time": t, "age": observation["observation_age"],
                   # 规则的速度来自报告前一步到报告时刻的位移，所以窗口从 报告时刻−1 开始。
                   "maneuver": int(maneuvering(env.car_trajectory, t - observation["observation_age"] - 1, t + 1)),
                   "rule_estimate_error": distance(estimate, true_next),
                   "rule_gap": distance(after_move(env, rule_action), true_next)
                   - distance(after_move(env, oracle_action), true_next),
                   "follow": int(action == rule_action), "action": action, "rule_action": rule_action,
                   "oracle_action": oracle_action, "q_margin": float(q[action] - q[rule_action]),
                   "distance_to_car": distance(env.uav_pos, env.car_pos)}
            predicted = agent.predict_car(observation)
            for h, point in zip(agent.config.prediction_steps, predicted if predicted is not None else []):
                if t + h <= config.max_steps:
                    row[f"pred_error_h{h}"] = distance(point, env.car_trajectory[t + h])
            steps.append(row)
        observation, _, terminated, truncated, info = env.step(action)
        if first_contact is None and info["contact"]:
            first_contact = info["time"]
        if terminated or truncated:
            episode = {"seed": seed, "success": int(info["success"]), "steps": info["time"],
                       "completion": info["time"] if info["success"] else config.max_steps,
                       "first_contact": first_contact}
            return episode, steps


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def group_summary(rows, wrong_threshold: float) -> dict:
    """规则走偏 = rule_gap 超过阈值；跟随率按规则对/错分开算。"""
    wrong = [r for r in rows if r["rule_gap"] > wrong_threshold]
    right = [r for r in rows if r["rule_gap"] <= wrong_threshold]
    horizons = sorted({k for r in rows for k in r if k.startswith("pred_error_h")})
    return {"steps": len(rows), "rule_wrong_rate": len(wrong) / len(rows) if rows else None,
            "rule_estimate_error": mean([r["rule_estimate_error"] for r in rows]),
            "follow_rate_when_rule_right": mean([r["follow"] for r in right]),
            "follow_rate_when_rule_wrong": mean([r["follow"] for r in wrong]),
            "oracle_match_rate": mean([int(r["action"] == r["oracle_action"]) for r in rows]),
            **{h: mean([r.get(h) for r in rows]) for h in horizons}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3_000_000)
    parser.add_argument("--wrong-threshold", type=float, default=0.5, help="规则差距超过多少格算规则走偏")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    report = []
    for checkpoint in args.checkpoints:
        agent, config = DQNAgent.load(checkpoint)
        check_test_seeds(agent, args.seed, args.episodes)
        env = GridWorldEnv(config, expose_truth=agent.config.needs_true_trajectory)
        episodes, steps = [], []
        for seed in range(args.seed, args.seed + args.episodes):
            episode, rows = run_episode(agent, env, seed)
            episodes.append(episode)
            steps.extend(rows)
        name = checkpoint.parent.name
        columns = sorted({k for r in steps for k in r}, key=lambda k: (k.startswith("pred_error"), k))
        with (args.output / f"{name}_steps.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(steps)
        report.append({
            "checkpoint": str(checkpoint), "observation_delay": config.observation_delay,
            "rule_prior": agent.config.rule_prior, "rule_source": agent.config.rule_source,
            "motion_prediction": agent.motion_predictor is not None,
            "success_rate": mean([e["success"] for e in episodes]),
            "mean_completion": mean([e["completion"] for e in episodes]),
            "mean_first_contact": mean([e["first_contact"] for e in episodes]),
            "all": group_summary(steps, args.wrong_threshold),
            "straight": group_summary([r for r in steps if not r["maneuver"]], args.wrong_threshold),
            "maneuver": group_summary([r for r in steps if r["maneuver"]], args.wrong_threshold),
        })
        print(json.dumps(report[-1], indent=2, ensure_ascii=False), flush=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
