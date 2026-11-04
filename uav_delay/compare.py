"""规则与“规则 + PPO 修正”的逐场景配对比较：均值差、bootstrap 置信区间、Wilcoxon 符号秩检验。"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .evaluation import DEVELOPMENT, TEST, evaluate, load_config
from .rules import chase
from .train import load_run, torch_policy

SEEDS = (0, 1, 2)


def run_dir(delay, seed):
    return Path("outputs") / f"delay_torch_resid_d{delay}_seed{seed}"


def scores(job):
    """返回一组策略在给定场景上的逐场景得分；环境配置取自训练时保存的 config.json。"""
    kind, delay, seed, split = job
    seeds = TEST if split == "test" else DEVELOPMENT
    if kind == "rule":
        # 规则在种子 0 的训练环境上评估；main 已检查各种子环境一致。
        cfg, _ = load_config(run_dir(delay, 0))
        cfg, policy = replace(cfg, residual=0.0), chase(True)
    else:
        cfg, saved_delay, model = load_run(run_dir(delay, seed))
        assert saved_delay == delay, (saved_delay, delay)
        policy = torch_policy(model)
    rows = evaluate(policy, delay, cfg, seeds)["episodes_detail"]
    return job, [row["score"] for row in rows]


def paired_stats(rule, residual, rng):
    from scipy.stats import wilcoxon
    diff = residual - rule
    boot = rng.choice(diff, size=(10_000, len(diff))).mean(axis=1)
    nonzero = np.count_nonzero(diff)
    p = float(wilcoxon(residual, rule, zero_method="pratt").pvalue) if nonzero else 1.0
    return dict(
        rule_mean=float(rule.mean()), residual_mean=float(residual.mean()),
        mean_difference=float(diff.mean()),
        ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        wilcoxon_p=p, scenes_better=int((diff > 0).sum()), scenes_worse=int((diff < 0).sum()),
        scenes_equal=int((diff == 0).sum()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delays", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--split", choices=("development", "test"), default="test")
    args = parser.parse_args()
    for d in args.delays:
        configs = {load_config(run_dir(d, s))[0] for s in SEEDS}
        if len(configs) != 1:
            raise ValueError(f"delay {d}: training seeds used different environment configs")

    jobs = [("rule", d, None, args.split) for d in args.delays]
    jobs += [("residual", d, s, args.split) for d in args.delays for s in SEEDS]
    with ProcessPoolExecutor() as pool:
        results = dict(pool.map(scores, jobs))

    rng = np.random.default_rng(0)
    report = []
    for d in args.delays:
        rule = np.asarray(results[("rule", d, None, args.split)])
        per_seed = [np.asarray(results[("residual", d, s, args.split)]) for s in SEEDS]
        # 每个场景取三个训练种子的平均，作为该场景上的残差策略得分。
        summary = paired_stats(rule, np.mean(per_seed, axis=0), rng)
        summary.update(delay=d, split=args.split,
                       seed_means=[float(x.mean()) for x in per_seed],
                       seed_wilcoxon_p=[paired_stats(rule, x, rng)["wilcoxon_p"] for x in per_seed])
        print(json.dumps(summary, ensure_ascii=False))
        report.append(summary)
    output = Path("outputs/delay_compare")
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{args.split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
