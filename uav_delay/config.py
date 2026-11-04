"""实验参数：场景尺度、车辆运动、任务规则、观测归一化、奖励权重。每步按 1 s，距离单位 m。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # 场景尺度
    span: float = 245.0  # 正方形地图边长
    spacing: float = 5.0  # 车辆每步移动距离
    speed: float = 8.0  # UAV 每步最大移动距离
    radius: float = 10.0  # 有效采集范围
    goal_radius: float = 4.0  # 终点范围
    duration: int = 100  # 回合步数；车辆全程行驶、全程可采集

    # 车辆运动
    turns: tuple = (0.85, 0.07, 0.01, 0.07)  # 相对当前朝向：直行、右转、掉头、左转
    car_goal_bias: float = 1.0  # 车辆偏向自己的目的地行驶的强度（目的地不告诉无人机）

    # 任务规则
    required: float = 5.0  # 成功所需的最少采集量（完整接触步）
    return_margin: int = 2  # 时间余量降到此值时自动返程

    # 观测与控制
    car_view: float = 50.0  # 收到的车辆相对位置按此距离归一化，超出时只保留方向
    residual: float = 0.0  # >0 时策略输出对规则动作的修正量，最大幅度为此值

    # 奖励（只用于训练）
    sensing: float = 0.2  # 每个完整接触步的奖励；失败时全部扣回
    approach: float = 1.0  # 采够最少量之前，每步靠近车辆的距离 / 地图对角线 的权重
