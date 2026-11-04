"""场景划分、读取训练配置、统一的回合评估。PPO 与规则基线共用同一评估函数。"""

from dataclasses import fields
import json
from pathlib import Path

import numpy as np

from .config import Config
from .env import DelayEnv

# 场景种子划分：训练、验证（学习曲线）、开发（调试与分析）、测试（只用于最终比较）。
TRAINING = (400_000, 402_000)
VALIDATION = range(410_000, 410_032)
DEVELOPMENT = range(430_000, 430_200)
TEST = range(420_000, 420_200)


def load_config(run_dir):
    """读取训练时保存的环境配置和延迟，保证评估环境与训练环境一致。"""
    metadata = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
    environment = dict(metadata["environment"])
    # 旧配置单独保存了车辆行驶步数，现与回合步数合并，只接受两者相等的旧配置。
    if environment.pop("car_duration", environment["duration"]) != environment["duration"]:
        raise ValueError(f"{run_dir}: car_duration differs from duration")
    unknown = set(environment) - {field.name for field in fields(Config)}
    if unknown:
        raise ValueError(f"{run_dir}: unknown config keys {sorted(unknown)}")
    environment["turns"] = tuple(environment["turns"])
    return Config(**environment), metadata["delay"]


def evaluate(policy, delay, cfg, seeds):
    """policy(obs, env) -> action；在给定场景上逐局运行并汇总。"""
    env = DelayEnv(delay, cfg)
    rows = []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        done = False
        while not done:
            obs, _, done, _, info = env.step(policy(obs, env))
        rows.append(info)
    steps = [row["steps"] for row in rows if row["success"]]
    return dict(
        episodes=len(rows),
        mean_score=float(np.mean([row["score"] for row in rows])),
        success_rate=float(np.mean([row["success"] for row in rows])),
        sensing_completion_rate=float(np.mean([row["sensing_complete"] for row in rows])),
        mean_collected=float(np.mean([row["collected"] for row in rows])),
        mean_success_steps=float(np.mean(steps)) if steps else None,
        episodes_detail=rows,
    )
