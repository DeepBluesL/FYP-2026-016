"""离线对比预测器：在同一批观测上，让线性预测、LSTM、VAR 和网格推演（参考）各自预测下一步车位，并比较规则据此选的动作。

车辆路线只由路线种子决定、与无人机无关，所以任一时刻各预测器的输入（延迟报告历史）只取决于（路线, 时刻）：
- 全时段样本 B：每条测试路线的每个时刻 t（d ≤ t < max_steps）都预测一次，不经过任何策略，只看预测误差；
- 策略样本 A：用 --policy 的模型飞同样的路线，取采集阶段且已收到报告的步（口径同 diagnose.py），
  在当时的无人机状态下算各预测器的规则动作。
分组（一直直走 / 转过弯）、规则差距和走偏阈值与 diagnose.py 相同。误差 = 预测 − 真实，方向分量都以报告时刻已知的信息为基准：
- 沿车头：投影到报告时最后一次观测到的行驶方向，负数为落后；横向取绝对值；首报时朝向未知，不计；
- 朝中心：投影到报告位置指向地图中心的方向，正数为偏向中心。
真正的条件平均预测在样本 B 上这两个分量的平均应接近 0；按“未来是否转弯”分组后，各组本来就会有偏差。
参考行“网格均值 / 网格最可能格”用 uav_demo/grid_prediction.py 的基础转向概率版本（不含目的地偏向）。
另外统计 --policy 和 --own-flight 的模型自己飞时，采集计数增加的步里车处于“一直直走”时段的比例。
用法见 outputs/diag_v9/run_compare_d8.sh。
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from diagnose import after_move, distance, maneuvering, mean
from evaluate import check_test_seeds
from uav_demo.agent import DQNAgent
from uav_demo.environment import GridWorldEnv
from uav_demo.extrapolate_pursuit import ExtrapolatePursuit
from uav_demo.grid_prediction import GridPredictor, expected_positions, last_heading, most_likely_positions
from uav_demo.linear_prediction import LinearPredictor
from uav_demo.vehicle import DIRECTIONS

HORIZONS = (4, 8, 12)  # DQN 输入的三个预测点，相对当前时刻
GROUPS = {"all": lambda row: True, "straight": lambda row: not row["maneuver"],
          "turned": lambda row: bool(row["maneuver"])}
ERROR_METRICS = ("error", "hit", "along", "cross", "center", "to_grid")
RULE_METRICS = ("wrong", "contact")


def report_history(trajectory, time: int, delay: int) -> tuple:
    """t 时刻无人机已收到的报告历史，与 GridWorldEnv 观测里的 car_history 相同。"""
    return tuple((s, (int(trajectory[s][0]), int(trajectory[s][1]))) for s in range(time - delay + 1))


def error_components(predicted, true, report, heading, center):
    """误差 = 预测 − 真实，返回（沿车头分量, 横向绝对值, 朝中心分量）；朝向未知时前两项为 None。"""
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    toward = np.asarray(center, dtype=np.float64) - np.asarray(report, dtype=np.float64)
    center_part = float(error @ toward / np.linalg.norm(toward))
    if heading is None:
        return None, None, center_part
    forward = DIRECTIONS[heading]
    return float(error @ forward), float(abs(error @ (forward[1], -forward[0]))), center_part


def forecast_table(predictors, grid, trajectory, config) -> dict:
    """一条路线上每个时刻、每个预测器对报告后第 1..K 步的预测：{t: (报告历史, {名称: [K, 2]})}。"""
    delay = config.observation_delay
    steps = tuple(range(1, delay + max(HORIZONS) + 1))
    table = {}
    for t in range(delay, config.max_steps):
        history = report_history(trajectory, t, delay)
        forecasts = {name: np.asarray(predictor.predict_positions(history, config.grid_size, steps,
                                                                   config.car_move_every), dtype=np.float64)
                     for name, predictor in predictors.items()}
        maps = grid.distributions(history, config.grid_size, steps, config.car_move_every)
        forecasts["grid_mean"] = expected_positions(maps)
        forecasts["grid_mode"] = most_likely_positions(maps).astype(np.float64)
        table[t] = (history, forecasts)
    return table


def error_row(route, t, history, forecasts, trajectory, config) -> dict:
    """样本 B 的一行：下一步误差及其分量、往前推的误差曲线、DQN 三个预测点的误差。forecasts[name][i] 是报告后第 i+1 步。"""
    delay = config.observation_delay
    heading = last_heading(history, config.car_move_every)
    true_next = trajectory[t + 1]
    center = np.full(2, (config.grid_size - 1) / 2)
    row = {"route": route, "time": t, "heading_known": int(heading is not None),
           # 与 diagnose.py 相同：窗口从报告前一步开始，因为最后观测到的方向来自那一步的位移。
           "maneuver": int(maneuvering(trajectory, t - delay - 1, t + 1))}
    grid_next = forecasts["grid_mean"][delay]
    for name, points in forecasts.items():
        predicted = points[delay]  # 报告后第 d+1 步 = 当前时刻的下一步，即规则追的位置
        error = distance(predicted, true_next)
        along, cross, center_part = error_components(predicted, true_next, history[-1][1], heading, center)
        row.update({f"{name}_error": error, f"{name}_hit": int(error <= config.sensing_radius),
                    f"{name}_along": along, f"{name}_cross": cross, f"{name}_center": center_part,
                    f"{name}_to_grid": distance(predicted, grid_next)})
        for k in range(1, delay + 2):
            row[f"{name}_k{k}"] = distance(points[k - 1], trajectory[t - delay + k])
        for h in HORIZONS:
            if t + h <= config.max_steps:
                row[f"{name}_h{h}"] = distance(points[delay + h - 1], trajectory[t + h])
    return row


def rule_row(env, observation, mask, forecasts, threshold: float) -> dict:
    """样本 A 的一行：各预测器（及真实位置）作为规则目标时选的动作，是否走偏、移动后是否在采集半径内。"""
    config = env.config
    true_next = tuple(int(v) for v in env.car_trajectory[env.time + 1])
    pursuit = ExtrapolatePursuit(config)
    targets = {name: tuple(points[config.observation_delay]) for name, points in forecasts.items()}
    targets["oracle"] = true_next
    actions = {name: pursuit.choose_action(observation, mask, target) for name, target in targets.items()}
    best = distance(after_move(env, actions["oracle"]), true_next)
    row = {"distance_to_car": distance(env.uav_pos, env.car_pos)}
    for name, action in actions.items():
        gap = distance(after_move(env, action), true_next)
        row.update({f"{name}_action": action, f"{name}_wrong": int(gap - best > threshold),
                    f"{name}_contact": int(gap <= config.sensing_radius)})
    return row


def fly(agent, env, route, tables, b_rows, threshold, name):
    """飞一局。tables 不为 None 时记录策略样本 A；始终记录采集阶段每步的车辆时段及采集计数是否增加。"""
    config = env.config
    delay = config.observation_delay
    observation, info = env.reset(seed=route)
    pursuit_rows, sensing_rows = [], []
    linear_rule = agent.config.rule_source == "linear" or isinstance(agent.motion_predictor, LinearPredictor)
    while True:
        mask = env.valid_action_mask()
        # 自动返程时只剩一个合法动作，DQN 选什么都一样，跳过它省去预测器调用。
        action = int(np.flatnonzero(mask)[0]) if mask.sum() == 1 else agent.choose_action(observation, 0.0, mask)
        t, sensing = env.time, not info["sensing_finished"]
        if sensing and observation["has_observation"]:
            label = "turned" if maneuvering(env.car_trajectory, t - delay - 1, t + 1) else "straight"
            if tables is not None:
                row = {"policy": name} | b_rows[route, t] | rule_row(env, observation, mask, tables[route][t][1], threshold)
                own_rule = agent.rule_action(observation, mask)
                if linear_rule and row["linear_action"] != own_rule:
                    raise RuntimeError("查表得到的线性预测规则动作与策略自己的规则动作不一致")
                row["follow"] = int(action == own_rule)
                pursuit_rows.append(row)
        else:
            label = "no_report"
        progress = env.sensing_progress
        observation, _, terminated, truncated, info = env.step(action)
        if sensing:
            sensing_rows.append({"policy": name, "route": route, "time": t, "label": label,
                                 "increment": int(info["sensing_progress"] > progress), "final": 0})
        if terminated or truncated:
            break
    if info["sensing_complete"]:
        # 连续采集模式下，最后 sensing_steps 次计数增加就是完成采集的那段连续接触。
        for row in [r for r in sensing_rows if r["increment"]][-config.sensing_steps:]:
            row["final"] = 1
    episode = {"policy": name, "route": route, "success": int(info["success"]),
               "completion": info["time"] if info["success"] else config.max_steps}
    return episode, pursuit_rows, sensing_rows


def cluster_se(values, clusters):
    """按路线聚类的均值标准误：同一路线上的相邻步高度相关，不能当作独立样本。"""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return None
    sums = defaultdict(float)
    for value, cluster in zip(values - values.mean(), clusters):
        sums[cluster] += value
    groups = len(sums)
    if groups < 2:
        return None
    return float(np.sqrt(groups / (groups - 1) * sum(s * s for s in sums.values())) / len(values))


def summarize(rows, names, metrics) -> dict:
    keys = {k for r in rows for k in r}
    summary = {}
    for group, keep in GROUPS.items():
        subset = [r for r in rows if keep(r)]
        summary[group] = {"steps": len(subset), "heading_unknown": sum(1 - r["heading_known"] for r in subset)}
        for name in names:
            stats = {m: mean([r.get(f"{name}_{m}") for r in subset]) for m in metrics if f"{name}_{m}" in keys}
            if f"{name}_error" in keys:
                squared = mean([r[f"{name}_error"] ** 2 for r in subset])
                stats["rmse"] = None if squared is None else float(np.sqrt(squared))
                for key in sorted(k for k in keys if re.fullmatch(rf"{name}_[kh]\d+", k)):
                    stats[key[len(name) + 1:]] = mean([r.get(key) for r in subset])
            summary[group][name] = stats
    return summary


def paired_difference(rows, name, metric, baseline="linear") -> dict:
    """同一步上 预测器 − 基线 的差：均值与按路线聚类的标准误。"""
    diffs = [r[f"{name}_{metric}"] - r[f"{baseline}_{metric}"] for r in rows]
    return {"mean": mean(diffs), "se": cluster_se(diffs, [r["route"] for r in rows])}


def paired_differences(rows, names, metrics, baseline="linear") -> dict:
    result = {}
    for group, keep in GROUPS.items():
        subset = [r for r in rows if keep(r)]
        result[group] = {name: {m: paired_difference(subset, name, m, baseline) for m in metrics}
                         for name in names if name != baseline}
    return result


def add_family_means(summary, names) -> list[str]:
    """同一类预测器有多个种子时（如 lstm_seed0..2），补一行各项指标的种子平均。"""
    families = defaultdict(list)
    for name in names:
        if (match := re.fullmatch(r"(.+)_seed\d+", name)):
            families[match.group(1)].append(name)
    added = []
    for family, members in families.items():
        if len(members) < 2:
            continue
        for group in summary.values():
            metrics = group[members[0]].keys()
            group[f"{family}_mean"] = {m: mean([group[n].get(m) for n in members]) for m in metrics}
        added.append(f"{family}_mean")
    return added


def load_predictors(paths, env_config, seed, episodes):
    predictors = {"linear": LinearPredictor()}
    for path in paths:
        agent, config = load_checked(path, env_config, seed, episodes)
        predictor = agent.motion_predictor
        if isinstance(predictor, torch.nn.Module):
            predictor.to("cpu")
        kind = agent.config.prediction_model_type
        # VAR 是同一批路线上的最小二乘，种子不同系数也相同；相同就只算一次。
        twin = next((name for name, other in predictors.items() if kind == "var" and name.startswith("var")
                     and np.array_equal(other.coefficients, predictor.coefficients)), None)
        if twin is not None:
            print(f"{path} 的 VAR 系数与 {twin} 相同，跳过", flush=True)
            continue
        predictors[f"{kind}_seed{agent.config.seed}"] = predictor
    var_names = [name for name in predictors if name.startswith("var_seed")]
    if len(var_names) == 1:
        predictors["var"] = predictors.pop(var_names[0])
    return predictors


def load_checked(path, env_config, seed, episodes):
    agent, config = DQNAgent.load(path)
    check_test_seeds(agent, seed, episodes)
    if env_config is not None and config != env_config:
        raise ValueError(f"{path} 的环境配置与第一个策略不同，不能放在一起比较")
    return agent, config


def sensing_summary(episodes, sensing_rows) -> dict:
    """按模型族（去掉 _seedN）汇总：成功率、完成步数、采集阶段 / 计数增加的步 / 最终 5 步中车处于直走时段的比例。"""
    family = lambda name: re.sub(r"_seed\d+$", "", name)
    result = {}
    for key in sorted({family(e["policy"]) for e in episodes}):
        own = [r for r in sensing_rows if family(r["policy"]) == key]
        reported = [r for r in own if r["label"] != "no_report"]
        straight = lambda rows: mean([int(r["label"] == "straight") for r in rows])
        result[key] = {
            "episodes": sum(family(e["policy"]) == key for e in episodes),
            "success_rate": mean([e["success"] for e in episodes if family(e["policy"]) == key]),
            "mean_completion": mean([e["completion"] for e in episodes if family(e["policy"]) == key]),
            "sensing_steps": len(reported), "straight_share_all": straight(reported),
            "increments": sum(r["increment"] for r in reported),
            "straight_share_increment": straight([r for r in reported if r["increment"]]),
            "straight_share_final": straight([r for r in reported if r["final"]]),
            "increments_without_report": sum(r["increment"] for r in own if r["label"] == "no_report"),
        }
    return result


DISPLAY = {"linear": "线性预测", "var": "VAR", "grid_mean": "网格均值（参考）", "grid_mode": "网格最可能格（参考）",
           "oracle": "真实位置", "lstm_mean": "LSTM 三种子平均"}
GROUP_NAMES = {"all": "全体", "straight": "一直直走", "turned": "转过弯"}


def display(name):
    return DISPLAY.get(name, re.sub(r"^(\w+)_seed(\d+)$", lambda m: f"{m.group(1).upper()} s{m.group(2)}", name))


def fmt(value, digits=3, percent=False):
    if value is None:
        return "—"
    return f"{100 * value:.1f}%" if percent else f"{value:.{digits}f}"


def markdown_report(report) -> str:
    names, delay = report["predictors"], report["observation_delay"]
    lines = [f"# 预测器离线对比（d={delay}）", "",
             f"策略样本 A：{', '.join(report['policies'])} 各飞 {report['episodes']} 条测试路线，采集阶段且已收到报告的步。",
             f"全时段样本 B：同样 {report['episodes']} 条路线的每个时刻 t（{delay} ≤ t < max_steps），与策略无关。", ""]
    columns = [("error", "下一步误差", False), ("rmse", "均方根", False), ("hit", "误差≤2格", True),
               ("along", "沿车头(负=落后)", False), ("cross", "横向|·|", False), ("center", "朝中心(正=偏中心)", False),
               ("to_grid", "与网格均值距离", False)]
    for sample, title, rule in (("sample_a", "策略样本 A：下一步预测与规则动作", True),
                                ("sample_b", "全时段样本 B：下一步预测", False)):
        lines += [f"## {title}", ""]
        for group, label in GROUP_NAMES.items():
            stats = report[sample][group]
            header = [c[1] for c in columns] + (["规则选错率", "规则动作后在半径内"] if rule else [])
            lines += [f"### {label}（{stats['steps']} 步，其中首报朝向未知 {stats['heading_unknown']} 步）", "",
                      "| 预测器 | " + " | ".join(header) + " |", "|---" * (len(header) + 1) + "|"]
            for name in names + report["family_means"] + (["oracle"] if rule else []):
                row = stats[name]
                cells = [fmt(row.get(key), percent=pct) for key, _, pct in columns]
                if rule:
                    cells += [fmt(row.get("wrong"), percent=True), fmt(row.get("contact"), percent=True)]
                lines.append(f"| {display(name)} | " + " | ".join(cells) + " |")
            lines.append("")
    lines += ["## 与线性预测的逐步配对差（策略样本 A，预测器 - 线性预测，± 为按路线聚类的 1.96 倍标准误）", ""]
    diff_columns = (("error", "下一步误差", False), ("hit", "误差≤2格", True),
                    ("wrong", "规则选错率", True), ("contact", "规则动作后在半径内", True))
    for group, label in GROUP_NAMES.items():
        lines += [f"### {label}", "", "| 预测器 | " + " | ".join(c[1] for c in diff_columns) + " |",
                  "|---" * (len(diff_columns) + 1) + "|"]
        for name, stats in report["paired_vs_linear"][group].items():
            cells = []
            for key, _, pct in diff_columns:
                m, se = stats[key]["mean"], stats[key]["se"]
                scale, unit = (100, "%") if pct else (1, "")
                cells.append("—" if m is None else f"{scale * m:+.2f}{unit} ± {scale * 1.96 * (se or 0):.2f}{unit}")
            lines.append(f"| {display(name)} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## 往前推的误差曲线（样本 B，报告后第 k 步的平均误差；k=1 是 LSTM 被训练的任务，"
              f"k={delay + 1} 是规则追的下一步）", ""]
    ks = [f"k{k}" for k in range(1, delay + 2)]
    for group, label in GROUP_NAMES.items():
        lines += [f"### {label}", "", "| 预测器 | " + " | ".join(ks) + " |", "|---" * (len(ks) + 1) + "|"]
        for name in names + report["family_means"]:
            lines.append(f"| {display(name)} | " + " | ".join(fmt(report["sample_b"][group][name].get(k), 2) for k in ks) + " |")
        lines.append("")
    hs = [f"h{h}" for h in HORIZONS]
    lines += ["## DQN 输入的三个预测点（样本 B，相对当前时刻 h 步的平均误差）", "",
              "| 预测器 | " + " | ".join(f"{GROUP_NAMES[g]} {h}" for g in GROUP_NAMES for h in hs) + " |",
              "|---" * (len(hs) * len(GROUP_NAMES) + 1) + "|"]
    for name in names + report["family_means"]:
        lines.append(f"| {display(name)} | " + " | ".join(fmt(report["sample_b"][g][name].get(h), 2)
                                                         for g in GROUP_NAMES for h in hs) + " |")
    lines += ["", "## 采集发生在什么时段（各模型自己飞）", "",
              "| 模型 | 局数 | 成功率 | 平均完成步数 | 采集阶段直走占比 | 采集+1 的步中直走占比 | 最终 5 步中直走占比 | 无报告时的 +1 |",
              "|---|---|---|---|---|---|---|---|"]
    for family, stats in report["sensing"].items():
        lines.append(f"| {family} | {stats['episodes']} | {fmt(stats['success_rate'], percent=True)} | "
                     f"{fmt(stats['mean_completion'], 1)} | {fmt(stats['straight_share_all'], percent=True)} | "
                     f"{fmt(stats['straight_share_increment'], percent=True)} | "
                     f"{fmt(stats['straight_share_final'], percent=True)} | {stats['increments_without_report']} |")
    return "\n".join(lines) + "\n"


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, nargs="+", required=True, help="飞策略样本 A 的模型")
    parser.add_argument("--predictors", type=Path, nargs="*", default=[], help="取其中保存的预测器参与比较；线性预测总是包含")
    parser.add_argument("--own-flight", type=Path, nargs="*", default=[], help="只用于统计采集时段的其他模型")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3_000_000)
    parser.add_argument("--wrong-threshold", type=float, default=0.5, help="规则差距超过多少格算规则走偏")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policies = [(path.parent.name, *load_checked(path, None, args.seed, args.episodes)) for path in args.policy]
    env_config = policies[0][2]
    for _, _, config in policies:
        if config != env_config:
            raise ValueError("各策略的环境配置必须相同")
    predictors = load_predictors(args.predictors, env_config, args.seed, args.episodes)
    torch.set_num_threads(args.torch_threads)  # 加载 DQNAgent 时会把线程数设为 1
    grid = GridPredictor.base(env_config)  # 基础转向概率，只作参考
    env = GridWorldEnv(env_config)
    routes = range(args.seed, args.seed + args.episodes)

    tables, b_rows = {}, {}
    for index, route in enumerate(routes, start=1):
        env.reset(seed=route)
        trajectory = env.car_trajectory.copy()
        tables[route] = forecast_table(predictors, grid, trajectory, env_config)
        for t, (history, forecasts) in tables[route].items():
            b_rows[route, t] = error_row(route, t, history, forecasts, trajectory, env_config)
        if index % 20 == 0:
            print(f"预测表 {index}/{args.episodes} 条路线", flush=True)

    episodes, a_rows, sensing_rows = [], [], []
    flights = [(name, agent, True) for name, agent, _ in policies]
    flights += [(path.parent.name, load_checked(path, env_config, args.seed, args.episodes)[0], False)
                for path in args.own_flight]
    torch.set_num_threads(args.torch_threads)
    for name, agent, sample_a in flights:
        if isinstance(agent.motion_predictor, torch.nn.Module):
            agent.motion_predictor.to("cpu")
        for route in routes:
            episode, pursuit, sensing = fly(agent, env, route, tables if sample_a else None, b_rows,
                                            args.wrong_threshold, name)
            episodes.append(episode)
            a_rows += pursuit
            sensing_rows += sensing
        own = [e for e in episodes if e["policy"] == name]
        print(f"{name}: 成功率 {mean([e['success'] for e in own]):.3f}，"
              f"平均完成步数 {mean([e['completion'] for e in own]):.1f}", flush=True)

    names = list(predictors) + ["grid_mean", "grid_mode"]
    sample_b = summarize(list(b_rows.values()), names, ERROR_METRICS)
    sample_a = summarize(a_rows, names + ["oracle"], ERROR_METRICS + RULE_METRICS)
    family_means = add_family_means(sample_b, names)
    add_family_means(sample_a, names)
    report = {
        "observation_delay": env_config.observation_delay, "episodes": args.episodes, "seed": args.seed,
        "wrong_threshold": args.wrong_threshold, "policies": [name for name, *_ in policies],
        "predictor_checkpoints": [str(p) for p in args.predictors], "predictors": names,
        "family_means": family_means, "sample_a": sample_a, "sample_b": sample_b,
        "paired_vs_linear": paired_differences(a_rows, names, ("error", "hit", "wrong", "contact")),
        "follow_rate": mean([r["follow"] for r in a_rows]),
        "sensing": sensing_summary(episodes, sensing_rows),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "sample_b_steps.csv", list(b_rows.values()))
    write_csv(args.output / "sample_a_steps.csv", a_rows)
    write_csv(args.output / "sensing_steps.csv", sensing_rows)
    write_csv(args.output / "episodes.csv", episodes)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    text = markdown_report(report)
    (args.output / "report.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
