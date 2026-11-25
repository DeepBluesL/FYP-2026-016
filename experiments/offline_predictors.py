"""离线评估预测器（不训练 DQN）：点预测落在真车 2 格以内的比例，以及「2 格内概率」是否校准。

评估路线：2000 条随机路线（种子 4100000 起），与 DQN 训练、验证、测试及预测器训练路线都不重叠。
报告历史从起点 s 截断、不加延迟，往前推 k 步（d=8 时规则追的下一步就是 k=9）。
- 命中率：起点 s = 0/5/40/100，k = 9 / 13；方向 LSTM 和目的地推断各算 mode 与 disk（半径内概率最大点）两种取点。
- 校准：起点 s = 5/20/40/70/100，k = 9；在 disk 点处，抽样分布给出的「车在 2 格内」的概率 p 与实际是否命中。
  按 p 的分位数分 10 档，ECE = 各档 |平均 p − 命中率| 按样本数加权的平均（Guo et al., ICML 2017）。
方向 LSTM 读训练好的预测器（DQN 模型文件里保存的那个，或单独保存的预测器）；目的地推断在预测器训练路线上重新拟合，结果确定。
用法：python experiments/offline_predictors.py --lstm outputs/diag_v9/rulep_dirlstmdisk_d8_seed0/best_model.pt --output DIR
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uav_demo.config import EnvConfig  # noqa: E402
from uav_demo.direction_prediction import DirectionPredictor  # noqa: E402
from uav_demo.goal_inference import GoalFilterPredictor  # noqa: E402
from uav_demo.linear_prediction import LinearPredictor  # noqa: E402
from uav_demo.point_selection import disk_sum, point_positions  # noqa: E402
from uav_demo.prediction import report_history  # noqa: E402
from uav_demo.prediction_data import split_route_seeds  # noqa: E402
from uav_demo.vehicle import generate_vehicle_trajectory  # noqa: E402

CONFIG = EnvConfig(car_route="random")
SIZE, RADIUS = CONFIG.grid_size, CONFIG.sensing_radius
HIT_STARTS, HORIZONS = (0, 5, 40, 100), (9, 13)
CALIBRATION_STARTS, CALIBRATION_K = (5, 20, 40, 70, 100), 9
EVAL_SEED = 4_100_000


def load_lstm(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    return DirectionPredictor.from_state(saved.get("motion_predictor", saved), device=device)


def point_forecast(model, rule, history):
    """与训练和部署时相同的点预测：按 rule 在抽样直方图上取点。"""
    model.point, model.radius = rule, RADIUS
    return model.predict_positions(history, SIZE, HORIZONS)


def histogram(points):
    counts = np.zeros((SIZE, SIZE))
    np.add.at(counts, (points[:, 0], points[:, 1]), 1.0)
    return counts / len(points)


def calibration(rows):
    p, hit = np.asarray(rows).T
    edges = np.quantile(p, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
    table = [(float(p[bins == b].mean()), float(hit[bins == b].mean()), int((bins == b).sum()))
             for b in range(10) if (bins == b).any()]
    ece = sum(n * abs(mean_p - rate) for mean_p, rate, n in table) / len(p)
    return {"mean_p": float(p.mean()), "hit_rate": float(hit.mean()), "ece": ece, "bins": table}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lstm", type=Path, required=True, help="方向 LSTM：DQN 模型文件或单独保存的预测器")
    parser.add_argument("--routes", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    lstm = load_lstm(args.lstm, args.device)
    goal, fit = GoalFilterPredictor.fit(CONFIG, split_route_seeds(1000, start_seed=100_000, split_seed=42))
    samplers = {"dirlstm": lstm, "goalfilter": goal}
    points = {"linear": lambda h: LinearPredictor().predict_positions(h, SIZE, HORIZONS)}
    for name, model in samplers.items():
        for rule in ("mode", "disk"):
            points[f"{name}_{rule}"] = lambda h, m=model, r=rule: point_forecast(m, r, h)
    hits = {name: np.zeros((args.routes, len(HIT_STARTS), len(HORIZONS)), dtype=bool) for name in points}
    records = {name: [] for name in samplers}
    for r, seed in enumerate(range(EVAL_SEED, EVAL_SEED + args.routes)):
        route, _ = generate_vehicle_trajectory(CONFIG, np.random.default_rng(seed))
        for i, start in enumerate(HIT_STARTS):
            history = report_history(route, start, 0)
            for name, predict in points.items():
                for j, (point, k) in enumerate(zip(predict(history), HORIZONS)):
                    hits[name][r, i, j] = np.hypot(*(point - route[start + k])) <= RADIUS
        for start in CALIBRATION_STARTS:
            history, truth = report_history(route, start, 0), route[start + CALIBRATION_K]
            for name, model in samplers.items():
                probability = histogram(model.sample_positions(history, SIZE, (CALIBRATION_K,))[0])
                point = point_positions([probability], "disk", RADIUS)[0].astype(int)
                p = float(disk_sum(probability, RADIUS)[point[0], point[1]])
                records[name].append((p, float(np.hypot(*(point - truth)) <= RADIUS)))
        if (r + 1) % 500 == 0:
            print(f"评估 {r + 1}/{args.routes} 条路线", flush=True)

    rates = {name: [100 * float(h[:, :, j].mean()) for j in range(len(HORIZONS))] for name, h in hits.items()}
    calibrated = {name: calibration(rows) for name, rows in records.items()}
    lines = ["# 预测器离线评估", "",
             f"{args.routes} 条路线（种子 {EVAL_SEED} 起）。目的地推断拟合出的转向概率 "
             f"{np.round(fit['turn_probabilities'], 3).tolist()}、偏向 {fit['goal_bias']:.3f}。", "",
             f"## 点落在真车 2 格以内的比例（起点 s ∈ {HIT_STARTS}）", "",
             "| 预测器 | k=9 | k=13 |", "|---|---|---|"]
    lines += [f"| {name} | {rate[0]:.1f}% | {rate[1]:.1f}% |" for name, rate in rates.items()]
    lines += ["", f"## 校准（k={CALIBRATION_K}，起点 s ∈ {CALIBRATION_STARTS}，disk 点处「2 格内」的概率）", "",
              "| 预测器 | 平均预测概率 | 实际命中率 | ECE |", "|---|---|---|---|"]
    lines += [f"| {name} | {100 * c['mean_p']:.1f}% | {100 * c['hit_rate']:.1f}% | {100 * c['ece']:.1f} 个百分点 |"
              for name, c in calibrated.items()]
    text = "\n".join(lines) + "\n"
    (args.output / "report.md").write_text(text, encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps({"hit_rates": rates, "calibration": calibrated},
                                                         indent=2), encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
