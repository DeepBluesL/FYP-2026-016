"""规则策略：只使用 env.received_car()，即 d 步前的车辆位置与位移，不看车辆真实位置。
首次报告到达之前没有车辆信息，规则原地悬停。

线性外推规则同时用作基线和残差控制里的“规则动作”，两处共用同一个函数。
"""

import numpy as np


def toward(target, env):
    """飞向目标点的动作：远处为单位方向（全速），不足一步时按距离缩小，正好到达。"""
    aim = target - env.pos
    return aim / max(env.cfg.speed, float(np.linalg.norm(aim)))


def extrapolate_action(env):
    """线性外推：假设车辆保持上一步位移，飞向 收到位置 + (d+1)×位移。"""
    report = env.received_car()
    if report is None:
        return np.zeros(2)
    car, shift = report
    return toward(car + (env.delay + 1) * shift, env)


def received_action(env):
    """不做预测，直接飞向收到的车辆位置。"""
    report = env.received_car()
    return np.zeros(2) if report is None else toward(report[0], env)


def chase(predict=True):
    """返回评估用的策略函数 policy(obs, env)；返程由环境自动完成。"""
    action = extrapolate_action if predict else received_action
    return lambda obs, env: action(env)
