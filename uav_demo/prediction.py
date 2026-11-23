"""预测器的统一调用接口：从已收到的历史定位报告预测车辆未来位置。

各预测器（linear_prediction、direction_prediction、goal_inference、oracle_prediction）都实现
predict_positions(history, grid_size, steps, move_every)，只读报告历史，不读真实未来路线（oracle 除外，仅作上限）。
"""

from __future__ import annotations

import numpy as np


def report_history(trajectory, time: int, delay: int) -> tuple:
    """t 时刻无人机已收到的报告历史，与 GridWorldEnv 观测里的 car_history 相同；供离线评估预测器使用。"""
    return tuple((s, (int(trajectory[s][0]), int(trajectory[s][1]))) for s in range(time - delay + 1))


def observed_motion(history, move_every):
    """从已收到的 (报告时间, 位置) 推导每次移动的位移；首报方向未知。

    沿用最近非零运动；报告有间隔时按其中的可移动次数取平均。
    """
    for (t0, p0), (t1, p1) in reversed(list(zip(history, history[1:]))):
        displacement = np.asarray(p1, dtype=np.float32) - p0
        moves = t1 // move_every - t0 // move_every
        if moves > 0 and np.any(displacement):
            return displacement / moves
    return np.zeros(2, dtype=np.float32)


def forecast_vehicle(predictor, state, config, steps):
    """steps 相对当前时刻；无有效报告时返回 None，不预测占位坐标。"""
    steps = tuple(steps)

    if not steps or any(
        isinstance(h, bool)
        or not isinstance(h, (int, np.integer))
        or h < 0
        for h in steps
    ):
        raise ValueError("`steps` must be a non-empty sequence of non-negative integers")

    if not state["has_observation"]:
        return None

    age = state["observation_age"]
    return predictor.predict_positions(
        state["car_history"],
        config.grid_size,
        steps=tuple(age + h for h in steps),
        move_every=config.car_move_every,
    )
