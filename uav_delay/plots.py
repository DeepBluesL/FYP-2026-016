"""论文用结果图：得分随延迟变化、学习曲线、轨迹对比、修正量随距离变化。输出到 outputs/figures/。"""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .compare import SEEDS, run_dir
from .env import DelayEnv
from .evaluation import DEVELOPMENT, VALIDATION, evaluate, load_config
from .rules import chase
from .train import load_run, torch_policy

OUT = Path("outputs/figures")
DELAYS = (2, 4, 6, 8)
# 默认分类色：1 蓝 = 规则，2 橙 = 规则 + PPO 修正，3 青 = 纯 PPO；文字用墨色，不用系列色。
RULE, RESIDUAL, PURE = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#2b2b29", "#6b6a64", "#e4e3de", "#fcfcfb"


def style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "lines.linewidth": 2,
    })


def score_vs_delay(plt):
    report = json.loads(Path("outputs/delay_compare/test.json").read_text(encoding="utf-8"))
    d = [r["delay"] for r in report]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot(d, [r["rule_mean"] for r in report], "-o", color=RULE, ms=8, label="规则（线性外推）")
    left.plot(d, [r["residual_mean"] for r in report], "-o", color=RESIDUAL, ms=8,
              label="规则 + PPO 修正（3 种子平均）")
    for r in report:
        left.scatter([r["delay"]] * len(SEEDS), r["seed_means"], s=18, color=RESIDUAL, alpha=0.45,
                     edgecolor=SURFACE, linewidth=1, zorder=3)
    left.set(xlabel="位置延迟 d（步）", ylabel="平均得分（采集量，失败记 0）", xticks=d,
             title="测试集 200 个场景的平均得分")
    left.legend(frameon=False)
    diff = [r["mean_difference"] for r in report]
    low = [r["mean_difference"] - r["ci95"][0] for r in report]
    high = [r["ci95"][1] - r["mean_difference"] for r in report]
    right.axhline(0, color=MUTED, linewidth=1)
    right.errorbar(d, diff, yerr=[low, high], fmt="o", color=RESIDUAL, ms=8, capsize=4,
                   elinewidth=2)
    for r in report:
        right.annotate(f"p={r['wilcoxon_p']:.1g}", (r["delay"], r["ci95"][1]),
                       textcoords="offset points", xytext=(0, 6), ha="center", color=MUTED,
                       fontsize=9)
    right.set(xlabel="位置延迟 d（步）", ylabel="得分差（修正 − 规则）", xticks=d,
              title="逐场景配对差值与 95% 置信区间")
    fig.tight_layout()
    fig.savefig(OUT / "score_vs_delay.png", dpi=200)


def learning_curves(plt):
    fig, axes = plt.subplots(1, len(DELAYS), figsize=(14, 3.6), sharey=False)
    for ax, delay in zip(axes, DELAYS):
        cfg, _ = load_config(run_dir(delay, 0))
        rule = evaluate(chase(True), delay, cfg, VALIDATION)["mean_score"]
        curves = []
        for seed in SEEDS:
            rows = json.loads((run_dir(delay, seed) / "curve.json").read_text())
            x = [r["timesteps"] / 1e6 for r in rows]
            curves.append([r["mean_score"] for r in rows])
            ax.plot(x, curves[-1], color=RESIDUAL, alpha=0.3, linewidth=1)
        ax.plot(x, np.mean(curves, axis=0), color=RESIDUAL, label="规则 + PPO 修正")
        pure = [Path(f"outputs/delay_torch_pure_d{delay}_seed{s}/curve.json") for s in SEEDS]
        if all(p.exists() for p in pure):
            rows = [json.loads(p.read_text()) for p in pure]
            ax.plot([r["timesteps"] / 1e6 for r in rows[0]],
                    np.mean([[r["mean_score"] for r in run] for run in rows], axis=0),
                    color=PURE, label="纯 PPO")
        ax.axhline(rule, color=RULE, linestyle="--", label="规则")
        ax.set(title=f"d = {delay}", xlabel="训练步数（百万）")
    axes[0].set_ylabel("验证集平均得分（32 局）")
    labels = {}
    for ax in axes:  # 汇总各子图的图例，避免遮挡数据；纯 PPO 只有 d=4、6
        for handle, label in zip(*ax.get_legend_handles_labels()):
            labels.setdefault(label, handle)
    fig.legend(labels.values(), labels.keys(), loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("学习曲线：细线为单个训练种子，粗线为 3 个种子平均", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(OUT / "learning_curves.png", dpi=200)


def rollout(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    path, done, first = [env.pos.copy()], False, None
    while not done:
        obs, _, done, _, info = env.step(policy(obs, env))
        path.append(env.pos.copy())
        if first is None and info["gained"] > 0:
            first = len(path) - 1
    return np.asarray(path), first, info


def trajectory(plt, delay=6):
    cfg, _, model = load_run(run_dir(delay, 1))
    rule_env, res_env = DelayEnv(delay, replace(cfg, residual=0.0)), DelayEnv(delay, cfg)
    best, chosen = -np.inf, None
    for seed in list(DEVELOPMENT)[:60]:  # 选一个两者都成功、修正提升最明显的开发场景
        _, _, r = rollout(rule_env, chase(True), seed)
        _, _, p = rollout(res_env, torch_policy(model), seed)
        if r["success"] and p["success"] and p["score"] - r["score"] > best:
            best, chosen = p["score"] - r["score"], seed
    rule_path, rule_first, rule_info = rollout(rule_env, chase(True), chosen)
    res_path, res_first, res_info = rollout(res_env, torch_policy(model), chosen)
    car = rule_env.route[: max(len(rule_path), len(res_path))]
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.plot(car[:, 0], car[:, 1], color=MUTED, linewidth=1.5, label="车辆真实路线")
    ax.plot(rule_path[:, 0], rule_path[:, 1], color=RULE,
            label=f"规则（得分 {rule_info['score']:.1f}，第 {rule_first} 步首次接触）")
    ax.plot(res_path[:, 0], res_path[:, 1], color=RESIDUAL,
            label=f"规则 + PPO 修正（得分 {res_info['score']:.1f}，第 {res_first} 步首次接触）")
    for path, first, color in ((rule_path, rule_first, RULE), (res_path, res_first, RESIDUAL)):
        ax.scatter(*path[first], s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
    ax.scatter(*rule_env.scene["start"], marker="s", s=80, color=INK, zorder=4)
    ax.scatter(*rule_env.goal, marker="*", s=200, color=INK, zorder=4)
    ax.scatter(*car[0], marker="^", s=80, color=MUTED, zorder=4)
    ax.annotate("起点", rule_env.scene["start"], xytext=(-8, 8), textcoords="offset points",
                ha="right")
    ax.annotate("终点", rule_env.goal, xytext=(6, 6), textcoords="offset points")
    ax.annotate("车辆出发", car[0], xytext=(-6, 10), textcoords="offset points", color=MUTED,
                ha="right")
    ax.set(xlim=(0, cfg.span), ylim=(0, cfg.span), aspect="equal", xlabel="x（m）", ylabel="y（m）",
           title=f"开发场景 {chosen}，d = {delay}：前 60 个开发场景中提升最大的示例\n（圆点为首次接触位置）")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "trajectory_d6.png", dpi=200)


def correction_vs_distance(plt):
    bins = [(0, 15, "< 15 m"), (15, 40, "15–40 m"), (40, 1e9, "> 40 m")]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    width = 0.18
    for i, delay in enumerate(DELAYS):
        sizes = {label: [] for *_, label in bins}
        for seed in SEEDS:
            cfg, _, model = load_run(run_dir(delay, seed))
            env = DelayEnv(delay, cfg)
            for scene in list(DEVELOPMENT)[:40]:
                obs, _ = env.reset(seed=scene)
                done = False
                while not done:
                    action = model.act(obs)
                    if not env.returning and env.has_report():
                        distance = np.linalg.norm(env.received_car()[0] - env.pos)
                        label = next(lab for lo, hi, lab in bins if lo <= distance < hi)
                        sizes[label].append(np.linalg.norm(action))
                    obs, _, done, _, _ = env.step(action)
        x = np.arange(len(bins)) + (i - 1.5) * width
        ax.bar(x, [np.mean(sizes[lab]) for *_, lab in bins], width * 0.9, color=RESIDUAL,
               alpha=0.35 + 0.2 * i, label=f"d = {delay}")
    ax.set(xticks=range(len(bins)), xticklabels=[lab for *_, lab in bins],
           xlabel="收到的车辆位置与无人机的距离", ylabel="PPO 修正量平均幅度",
           title="离车较远（> 40 m）时修正最大（开发场景 40 局 × 3 种子）")
    ax.legend(frameon=False, ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "correction_vs_distance.png", dpi=200)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    style(plt)
    for draw in (score_vs_delay, learning_curves, trajectory, correction_vs_distance):
        draw(plt)
        print("saved", draw.__name__)


if __name__ == "__main__":
    main()
