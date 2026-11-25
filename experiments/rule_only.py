"""不学习的规则基线：贪心直飞追各预测器给的下一步车位（t+1），看预测器本身对决策有多大用（没有 DQN、没有种子差异）。

预测器：直接追踪（追最近一次报告的车位，不预测）、线性预测、方向 LSTM 取 disk、目的地推断取 disk、真实位置（参考）。
环境与 run_queue.sh 训练时相同（随机路线、采完自动返程、势函数奖励、不计飞行距离代价）；200 条测试路线（种子 3000000 起），失败按 150 步计。
用法：python experiments/rule_only.py --delay 8 --lstm outputs/diag_v9/rulep_dirlstmdisk_d8_seed0/best_model.pt --output DIR
只跑部分预测器：--predictors latest linear（不含 dirlstmdisk 时不需要 --lstm）。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from offline_predictors import load_lstm  # noqa: E402
from uav_demo.config import EnvConfig  # noqa: E402
from uav_demo.environment import GridWorldEnv  # noqa: E402
from uav_demo.extrapolate_pursuit import ExtrapolatePursuit  # noqa: E402
from uav_demo.goal_inference import GoalFilterPredictor  # noqa: E402
from uav_demo.linear_prediction import LinearPredictor  # noqa: E402
from uav_demo.prediction import forecast_vehicle  # noqa: E402
from uav_demo.prediction_data import split_route_seeds  # noqa: E402

SEED, EPISODES = 3_000_000, 200
PREDICTORS = ("latest", "linear", "dirlstmdisk", "goalfilterdisk", "oracle")


def run(env, policy, model, seed: int) -> dict:
    observation, info = env.reset(seed=seed)
    while True:
        target = None
        if observation["has_observation"]:
            if model == "latest":
                target = tuple(int(v) for v in observation["car_pos"])
            elif model is None:
                target = tuple(int(v) for v in env.car_trajectory[min(env.time + 1, env.config.max_steps)])
            else:
                target = tuple(int(v) for v in forecast_vehicle(model, observation, env.config, (1,))[0])
        action = policy.choose_action(observation, env.valid_action_mask(), car_target=target)
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return {"success": int(info["success"]),
                    "completion": info["time"] if info["success"] else env.config.max_steps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=int, required=True)
    parser.add_argument("--lstm", type=Path, help="方向 LSTM：DQN 模型文件或单独保存的预测器")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--predictors", nargs="+", default=list(PREDICTORS), choices=PREDICTORS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "dirlstmdisk" in args.predictors and args.lstm is None:
        parser.error("跑 dirlstmdisk 需要 --lstm")
    config = EnvConfig(observation_delay=args.delay, car_route="random", auto_return=True, reward_v2=True,
                       movement_penalty=0.0)
    models = {}
    for name in args.predictors:
        if name == "dirlstmdisk":
            models[name] = load_lstm(args.lstm, args.device)
        elif name == "goalfilterdisk":
            models[name], _ = GoalFilterPredictor.fit(config, split_route_seeds(1000, start_seed=100_000, split_seed=42))
        else:
            models[name] = {"latest": "latest", "linear": LinearPredictor(), "oracle": None}[name]
        if name in ("dirlstmdisk", "goalfilterdisk"):
            models[name].point, models[name].radius = "disk", config.sensing_radius
    env, policy = GridWorldEnv(config), ExtrapolatePursuit(config)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        episodes = [run(env, policy, model, s) for s in range(SEED, SEED + EPISODES)]
        row = {"delay": args.delay, "predictor": name, "success_rate": float(np.mean([e["success"] for e in episodes])),
               "mean_completion": float(np.mean([e["completion"] for e in episodes])),
               "completion": [e["completion"] for e in episodes]}
        (args.output / f"d{args.delay}_{name}.json").write_text(json.dumps(row), encoding="utf-8")
        print(json.dumps({k: v for k, v in row.items() if k != "completion"}), flush=True)


if __name__ == "__main__":
    main()
