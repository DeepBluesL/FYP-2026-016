"""组间比较：每组若干种子，在 200 条测试路线（种子 3000000 起）上逐局评估，失败记 150 步。

逐局结果缓存在各模型目录的 test_episodes.csv，重复分析不再重跑。判定规则（事先写在各步的 criteria 文件里）：
- 每组 3 个种子：两组种子的平均完成步数完全分开，且按路线配对的差值 95% 区间不含 0 → 有差别；
  种子交错且平均差在 ±5 步以内 → 差不多；其余 → 不确定（同第 3 步）。
- 每组 ≥5 个种子：种子平均完成步数的置换检验（双侧）p < 0.05，且按路线配对的区间不含 0 → 有差别；
  p ≥ 0.05 且平均差在 ±5 步以内 → 差不多；其余 → 不确定。
用法：python experiments/analyze_groups.py --baseline rulep_linear_d8 --groups rulep_dirlstmdisk_d8 --seeds 0 1 2 --output DIR
组名可写成 名称@种子列表（如 plain_d8@0,1,2,3），覆盖 --seeds。
"""
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import check_test_seeds, evaluate_episode  # noqa: E402
from uav_demo.agent import DQNAgent  # noqa: E402
from uav_demo.environment import GridWorldEnv  # noqa: E402

SEED, EPISODES = 3_000_000, 200
RUNS = ROOT / "outputs" / "diag_v9"  # run_queue.sh 的输出位置


def episodes(name):
    """单个模型在测试路线上的逐局结果，[(路线, 成功, 完成步数)]；有缓存就读缓存。"""
    cache = RUNS / name / "test_episodes.csv"
    if cache.exists():
        with cache.open(encoding="utf-8") as file:
            rows = [(int(r["route"]), int(r["success"]), int(r["completion"])) for r in csv.DictReader(file)]
        if len(rows) == EPISODES and rows[0][0] == SEED:
            return rows
    agent, config = DQNAgent.load(RUNS / name / "best_model.pt")
    check_test_seeds(agent, SEED, EPISODES)
    env = GridWorldEnv(config, expose_truth=agent.config.needs_true_trajectory)
    rows = []
    for route in range(SEED, SEED + EPISODES):
        result = evaluate_episode(agent, env, route)
        rows.append((route, int(result["success"]), result["steps"] if result["success"] else config.max_steps))
    with cache.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("route", "success", "completion"))
        writer.writerows(rows)
    return rows


def permutation_p(first, second):
    """两组种子均值差的双侧置换检验；组合数不多时穷举，否则抽 20000 次。"""
    pooled, n = np.asarray(first + second, dtype=float), len(first)
    observed = abs(np.mean(first) - np.mean(second))
    combos = list(itertools.combinations(range(len(pooled)), n)) if len(pooled) <= 20 else None
    if combos is None:
        rng = np.random.default_rng(0)
        combos = [rng.choice(len(pooled), n, replace=False) for _ in range(20_000)]
    hits = 0
    for combo in combos:
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(combo)] = True
        hits += abs(pooled[mask].mean() - pooled[~mask].mean()) >= observed - 1e-12
    return hits / len(combos)


def verdict(group_seeds, base_seeds, diff, half_width, p):
    excludes_zero = abs(diff) > half_width
    if len(group_seeds) <= 3 or len(base_seeds) <= 3:
        different = max(group_seeds) < min(base_seeds) or min(group_seeds) > max(base_seeds)
    else:
        different = p < 0.05
    if different and excludes_zero:
        return "有差别（更好）" if diff < 0 else "有差别（更差）"
    if not different and abs(diff) <= 5:
        return "差不多"
    return "不确定"


def parse(spec, default_seeds):
    name, _, seeds = spec.partition("@")
    return name, ([int(s) for s in seeds.split(",")] if seeds else list(default_seeds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--title", default="组间比较")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    specs = [parse(s, args.seeds) for s in [args.baseline] + args.groups]
    results = {}
    for name, seeds in specs:
        for seed in seeds:
            run = f"{name}_seed{seed}"
            results[run] = episodes(run)
            print(f"{run}: 完成步数 {np.mean([r[2] for r in results[run]]):.1f}，"
                  f"成功率 {np.mean([r[1] for r in results[run]]):.3f}", flush=True)

    def per_route(name, seeds):
        return np.mean([[r[2] for r in results[f"{name}_seed{s}"]] for s in seeds], axis=0)

    def seed_means(name, seeds, column=2):
        return [float(np.mean([r[column] for r in results[f"{name}_seed{s}"]])) for s in seeds]

    base_name, base_seeds_list = specs[0]
    base_route, base_seeds = per_route(base_name, base_seeds_list), seed_means(base_name, base_seeds_list)
    lines = [f"# {args.title}（200 条测试路线，失败记 150 步）", "",
             f"| 组 | 种子数 | 各种子完成步数 | 平均 | 成功率 | 与 {base_name} 之差（95% 区间，按路线配对） | 置换检验 p | 判定 |",
             "|---|---|---|---|---|---|---|---|"]
    summary = {}
    for name, seeds in specs:
        values = seed_means(name, seeds)
        differences = per_route(name, seeds) - base_route
        diff = float(differences.mean())
        half = float(1.96 * differences.std(ddof=1) / np.sqrt(len(differences)))
        p = permutation_p(values, base_seeds) if name != base_name else None
        judged = "—" if name == base_name else verdict(values, base_seeds, diff, half, p)
        success = float(np.mean(seed_means(name, seeds, column=1)))
        summary[name] = {"seeds": seeds, "seed_completion": values, "mean_completion": float(np.mean(values)),
                         "success_rate": success, "diff": diff, "ci_half_width": half, "p": p, "verdict": judged}
        diff_text = "—" if name == base_name else f"{diff:+.1f}（{diff - half:+.1f} ~ {diff + half:+.1f}）"
        lines.append(f"| {name} | {len(seeds)} | {' / '.join(f'{v:.1f}' for v in values)} | {np.mean(values):.1f} | "
                     f"{100 * success:.1f}% | {diff_text} | {'—' if p is None else f'{p:.3f}'} | {judged} |")
    text = "\n".join(lines) + "\n"
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output / "report.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
