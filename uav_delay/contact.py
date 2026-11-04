"""采集接触时长：一步内无人机与车辆都做直线运动，计算两者距离不超过采集半径的时长。"""

import numpy as np


def contact_time(relative_position, relative_velocity, radius, dt):
    """解二次不等式 ||相对位置 + 相对速度*t||² <= 半径²，返回接触时长。"""
    a = float(relative_velocity @ relative_velocity)
    c = float(relative_position @ relative_position - radius * radius)
    if a < 1e-14:
        return dt if c <= 1e-10 else 0.0
    b = 2 * float(relative_position @ relative_velocity)
    discriminant = b * b - 4 * a * c
    if discriminant <= 0:
        return 0.0  # 只相切一瞬间，接触时长为 0
    root = np.sqrt(discriminant)
    return max(0.0, min(dt, (-b + root) / (2 * a)) - max(0.0, (-b - root) / (2 * a)))
