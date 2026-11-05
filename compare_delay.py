"""外推贪心与 DQN（多个训练种子）在同一批测试路线上逐局配对比较。

完成时间：成功回合取实际步数，失败回合记为 max_steps（越小越好）。
差值 = 外推贪心 − DQN（正数表示 DQN 更快）；95% 置信区间为路线 bootstrap，p 为 Wilcoxon 符号秩检验。
"""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate import check_test_seeds, evaluate_episode
from uav_demo.agent import DQNAgent
from uav_demo.environment import GridWorldEnv
from uav_demo.extrapolate_pursuit import ExtrapolatePursuit


def completion_times(rows, max_steps):
    return np.asarray([row["steps"] if row["success"] else max_steps for row in rows], dtype=float)


def paired(rule, dqn, rng):
    from scipy.stats import wilcoxon
    diff = rule - dqn
    boot = rng.choice(diff, size=(10_000, len(diff))).mean(axis=1)
    p = float(wilcoxon(rule, dqn, zero_method="pratt").pvalue) if np.any(diff) else 1.0
    return {"mean_difference": float(diff.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "wilcoxon_p": p, "dqn_faster": int((diff > 0).sum()), "dqn_slower": int((diff < 0).sum()),
            "equal": int((diff == 0).sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True, help="各训练种子的 best_model.pt")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    loaded = [DQNAgent.load(path) for path in args.checkpoints]
    configs = {json.dumps(vars(config), sort_keys=True, default=list) for _, config in loaded}
    if len(configs) != 1:
        parser.error("各训练种子的环境配置不一致，不能配对比较")
    config = loaded[0][1]
    seeds = range(args.seed, args.seed + args.episodes)
    for agent, _ in loaded:
        check_test_seeds(agent, args.seed, args.episodes)

    env = GridWorldEnv(config)
    rule_rows = [evaluate_episode(None, env, seed, ExtrapolatePursuit(config)) for seed in seeds]
    dqn_rows = [[evaluate_episode(agent, env, seed) for seed in seeds] for agent, _ in loaded]

    rule = completion_times(rule_rows, config.max_steps)
    per_seed = [completion_times(rows, config.max_steps) for rows in dqn_rows]
    rng = np.random.default_rng(0)
    report = {
        "observation_delay": config.observation_delay, "auto_return": config.auto_return,
        "episodes": args.episodes, "seed_start": args.seed,
        "rule_success_rate": float(np.mean([row["success"] for row in rule_rows])),
        "rule_mean_completion": float(rule.mean()),
        "dqn_success_rates": [float(np.mean([row["success"] for row in rows])) for rows in dqn_rows],
        "dqn_mean_completion": [float(times.mean()) for times in per_seed],
        # 每条路线取各种子平均，作为 DQN 在该路线上的完成时间。
        "paired_mean_over_seeds": paired(rule, np.mean(per_seed, axis=0), rng),
        "paired_per_seed": [paired(rule, times, rng) for times in per_seed],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
