"""学习速度：读各模型目录的 eval.csv（每 100 局一次验证），画学习曲线，并比较前若干局的平均验证完成步数。

验证完成步数 = 成功率 × 成功局的平均步数 + (1 − 成功率) × 150（失败记 150 步，同测试）。
学习速度取每个种子在 [--start, --until] 局各次验证的平均，越小说明学得越快。判定（事先写在 step15 的 criteria 里）：
每组 ≤3 个种子时，两组种子完全分开 → 有差别；更多种子时，双侧置换检验 p < 0.05 → 有差别。
不算有差别且平均差在 ±5 步以内 → 差不多；其余 → 不确定。
用法：python experiments/learning_curves.py --groups ls_pure_d8 ls_point_d8 ls_rule_d8 ls_rulep_d8 \
        --pairs ls_rulep_d8:ls_point_d8 ls_rulep_d8:ls_rule_d8 --seeds 0 1 2 --output DIR
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_groups import RUNS, permutation_p  # noqa: E402


def curve(run):
    """[(局数, 验证完成步数)]。"""
    max_steps = json.loads((RUNS / run / "config.json").read_text(encoding="utf-8"))["environment"]["max_steps"]
    with (RUNS / run / "eval.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    out = []
    for r in rows:
        success = float(r["success_rate"])
        steps = float(r["mean_success_steps"]) if r["mean_success_steps"] else 0.0
        out.append((int(r["episode"]), success * steps + (1 - success) * max_steps))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--pairs", nargs="*", default=[], help="组A:组B，报告 A − B")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--until", type=int, default=1500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    speed, curves = {}, {}
    for group in args.groups:
        runs = [curve(f"{group}_seed{s}") for s in args.seeds]
        episodes = [e for e, _ in runs[0]]
        curves[group] = (episodes, np.mean([[v for _, v in c] for c in runs], axis=0))
        speed[group] = [float(np.mean([v for e, v in c if args.start <= e <= args.until])) for c in runs]

    plt.figure(figsize=(6, 4))
    for group, (episodes, values) in curves.items():
        plt.plot(episodes, values, label=group)
    plt.xlabel("训练局数")
    plt.ylabel("验证完成步数（失败记 150）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output / "learning_curves.png", dpi=150)

    lines = [f"# 学习速度：第 {args.start}–{args.until} 局的平均验证完成步数（{len(args.seeds)} 个种子）", "",
             "| 组 | 各种子 | 平均 |", "|---|---|---|"]
    lines += [f"| {g} | {' / '.join(f'{v:.1f}' for v in vals)} | {np.mean(vals):.1f} |" for g, vals in speed.items()]
    lines += ["", "| 比较 | 平均差 | 置换检验 p | 判定 |", "|---|---|---|---|"]
    for pair in args.pairs:
        a, b = pair.split(":")
        diff, p = float(np.mean(speed[a]) - np.mean(speed[b])), permutation_p(speed[a], speed[b])
        if len(args.seeds) <= 3:
            different = max(speed[a]) < min(speed[b]) or min(speed[a]) > max(speed[b])
        else:
            different = p < 0.05
        judged = ("有差别（更快）" if diff < 0 else "有差别（更慢）") if different else ("差不多" if abs(diff) <= 5 else "不确定")
        lines.append(f"| {a} − {b} | {diff:+.1f} | {p:.3f} | {judged} |")
    text = "\n".join(lines) + "\n"
    (args.output / "report.md").write_text(text, encoding="utf-8")
    (args.output / "speed.json").write_text(json.dumps(speed, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
