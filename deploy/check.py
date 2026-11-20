"""接入示例 + 一致性检查：把训练环境当成"仿真平台"，只通过 ObservationBuilder 给 Policy 喂数据。

逐步核对：动作掩码、所选动作都与原评估代码（env 观测 + env 掩码）完全一致；并输出 200 条测试路线的成功率和完成时间。
用法：python deploy/check.py --delay 6 [--episodes 200]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy import ObservationBuilder, Policy  # noqa: E402
from uav_demo.environment import GridWorldEnv  # noqa: E402


def run_episode(policy: Policy, env: GridWorldEnv, seed: int) -> tuple[bool, int, int]:
    """返回 (是否成功, 用的步数, 与原评估代码不一致的步数)。"""
    env_observation, _ = env.reset(seed=seed)
    builder = ObservationBuilder(policy.config)
    true_car = [env.car_pos]  # 仿真端知道的真实车辆轨迹；报告晚 d 步到达
    mismatches = 0
    while True:
        t = env.time
        if t - policy.delay >= 0:
            builder.report(t - policy.delay, true_car[t - policy.delay])
        observation = builder.build(t, env.uav_pos, env.uav_speed, env.sensing_progress,
                                    env.sensing_complete, env.sensing_failed)
        action = policy.act(observation)
        expected = policy.agent.choose_action(env_observation, 0.0, env.valid_action_mask())
        mismatches += int(action != expected
                          or not np.array_equal(policy.valid_mask(observation), env.valid_action_mask()))
        env_observation, _, terminated, truncated, info = env.step(action)
        true_car.append(env.car_pos)
        if terminated or truncated:
            return bool(info["success"]), env.time, mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=int, choices=(4, 6, 8), default=6)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3_000_000)
    args = parser.parse_args()

    policy = Policy(delay=args.delay)
    env = GridWorldEnv(policy.config)
    results = [run_episode(policy, env, seed) for seed in range(args.seed, args.seed + args.episodes)]
    success = np.mean([r[0] for r in results])
    completion = np.mean([steps if ok else policy.config.max_steps for ok, steps, _ in results])
    mismatches = sum(r[2] for r in results)
    print(f"d={args.delay}  路线 {args.episodes}  成功率 {success:.1%}  平均完成时间 {completion:.1f} 步  "
          f"与原评估不一致的步数 {mismatches}")
    if mismatches:
        sys.exit(1)


if __name__ == "__main__":
    main()
