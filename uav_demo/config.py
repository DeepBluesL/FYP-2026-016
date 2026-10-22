"""集中管理场景参数和训练参数；修改默认值后重新训练即可。"""

from dataclasses import dataclass
import math


ACTIVATION_NAMES = ("relu", "leaky_relu", "tanh", "silu")


def validate_network_settings(hidden_sizes, activation, leaky_relu_slope) -> tuple[int, ...]:
    """配置和独立构造 QNetwork 共用的结构检查，不需要导入 PyTorch。"""
    if not isinstance(hidden_sizes, (tuple, list)) or not hidden_sizes:
        raise ValueError("hidden_sizes 必须是非空的正整数列表或元组")
    if any(isinstance(width, bool) or not isinstance(width, int) or width < 1 for width in hidden_sizes):
        raise ValueError("hidden_sizes 中每层宽度必须是正整数")
    if activation not in ACTIVATION_NAMES:
        raise ValueError(f"activation 必须是 {ACTIVATION_NAMES} 中的一种")
    if (
        isinstance(leaky_relu_slope, bool)
        or not isinstance(leaky_relu_slope, (int, float))
        or not math.isfinite(leaky_relu_slope)
        or not 0 <= leaky_relu_slope <= 1
    ):
        raise ValueError("leaky_relu_slope 必须是 [0, 1] 内的有限数")
    return tuple(hidden_sizes)


@dataclass
class EnvConfig:
    grid_size: int = 50
    max_steps: int = 300
    uav_start: tuple[int, int] = (0, 0)
    uav_goal: tuple[int, int] = (49, 49)
    initial_speed: int = 1

    car_move_every: int = 1  # 每隔多少个环境步移动 1 格；1 为每步一格，2 为两步一格
    car_spawn_region: str = "bottom_right"  # 兼容旧 checkpoint 的字段；当前统一固定右下角，随机起点逻辑已注释
    car_straight_prob: float = 0.65
    car_turn_prob: float = 0.15  # 左、右转各自的基础概率
    car_reverse_prob: float = 0.05
    car_goal_bias: float = 1.0  # 用 exp(bias * 靠近终点的距离) 加权方向

    sensing_radius: float = 2.0
    sensing_steps: int = 5
    sensing_mode: str = "consecutive"  # 或 cumulative

    step_penalty: float = -0.1
    progress_reward: float = 0.2
    progress_to_contact_boundary: bool = True  # 到感知范围即可，范围内不奖励继续贴近车中心
    movement_penalty: float = -0.1  # 每格实际飞行距离的小代价，允许悬停等车节省路程
    contact_reward: float = 2.0
    sensing_complete_reward: float = 50.0
    sensing_failure_penalty: float = -100.0  # 车辆到终点时仍未完成采集，仅扣一次
    failed_return_progress_scale: float = 0.1  # 失败后的导航塑形弱化，不让返航抵消失败代价
    success_reward: float = 100.0
    invalid_move_penalty: float = -1.0
    timeout_penalty: float = -10.0

    def __post_init__(self) -> None:
        if not isinstance(self.grid_size, int) or self.grid_size < 5:
            raise ValueError("grid_size 必须是 >= 5 的整数")
        if not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError("max_steps 必须是正整数")
        if not isinstance(self.car_move_every, int) or self.car_move_every < 1:
            raise ValueError("car_move_every 必须是正整数")
        if self.car_spawn_region not in ("bottom_right", "mixed"):
            raise ValueError("car_spawn_region 只能取 bottom_right 或 mixed")
        if self.max_steps // self.car_move_every < 2 * (self.grid_size - 1):
            raise ValueError("时间上限太短：需保证车辆可从任意起点到达终点")
        for name in ("uav_start", "uav_goal"):
            position = tuple(getattr(self, name))  # JSON 的 list 也可还原
            if len(position) != 2 or any(
                not isinstance(v, int) or not 0 <= v < self.grid_size for v in position
            ):
                raise ValueError(f"{name} 必须是地图内的整数坐标")
            setattr(self, name, position)
        if self.initial_speed not in (1, 2):
            raise ValueError("initial_speed 只能取 1 或 2")
        probabilities = (self.car_straight_prob, self.car_turn_prob, self.car_reverse_prob)
        if any(not math.isfinite(p) or p <= 0 for p in probabilities):
            raise ValueError("车辆各基础转向概率必须为有限正数")
        if not math.isclose(self.car_straight_prob + 2 * self.car_turn_prob + self.car_reverse_prob, 1.0):
            raise ValueError("直行 + 左转 + 右转 + 掉头的基础概率之和必须为 1")
        if not math.isfinite(self.car_goal_bias) or not 0 <= self.car_goal_bias <= 20:
            raise ValueError("car_goal_bias 必须在 [0, 20] 范围内")
        if not math.isfinite(self.sensing_radius) or self.sensing_radius < 0:
            raise ValueError("sensing_radius 必须是非负有限数")
        if not isinstance(self.sensing_steps, int) or not 1 <= self.sensing_steps <= self.max_steps:
            raise ValueError("sensing_steps 必须为 1 到 max_steps 之间的整数")
        if self.sensing_mode not in ("consecutive", "cumulative"):
            raise ValueError("sensing_mode 只能取 consecutive 或 cumulative")
        if not math.isfinite(self.sensing_failure_penalty) or self.sensing_failure_penalty >= 0:
            raise ValueError("sensing_failure_penalty 必须是有限负数")
        if not 0 <= self.failed_return_progress_scale <= 1:
            raise ValueError("failed_return_progress_scale 必须在 [0, 1] 内")
        if not isinstance(self.progress_to_contact_boundary, bool):
            raise ValueError("progress_to_contact_boundary 必须为布尔值")
        if not math.isfinite(self.movement_penalty) or self.movement_penalty > 0:
            raise ValueError("movement_penalty 必须是有限非正数")


@dataclass
class TrainConfig:
    episodes: int = 1500
    batch_size: int = 64
    learning_rate: float = 0.0005
    gamma: float = 0.99
    replay_capacity: int = 50_000
    learning_starts: int = 1000
    train_every: int = 4
    target_update_every: int = 250  # 梯度更新次数
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 60_000
    hidden_sizes: tuple[int, ...] = (128, 128, 128)  # 每个元素定义一个隐藏层的宽度
    activation: str = "leaky_relu"
    leaky_relu_slope: float = 0.01
    seed: int = 42
    device: str = "cpu"
    torch_threads: int = 1
    eval_every: int = 100
    eval_episodes: int = 10
    eval_seed: int = 1_000_000
    log_every: int = 25
    use_motion_prediction: bool = True
    prediction_routes: int = 300
    prediction_epochs: int = 20

    def __post_init__(self) -> None:
        positive_fields = (
            "episodes", "batch_size", "replay_capacity", "train_every", "target_update_every",
            "epsilon_decay_steps", "torch_threads", "eval_every", "eval_episodes", "log_every",
            "prediction_routes", "prediction_epochs",
        )
        for name in positive_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} 必须是正整数")
        self.hidden_sizes = validate_network_settings(
            self.hidden_sizes, self.activation, self.leaky_relu_slope
        )
        if not isinstance(self.learning_starts, int) or self.learning_starts < 0:
            raise ValueError("learning_starts 必须为非负整数")
        if self.replay_capacity < max(self.batch_size, self.learning_starts):
            raise ValueError("replay_capacity 不能小于 batch_size 或 learning_starts")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate 必须为有限正数")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma 必须在 [0, 1] 范围内")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("epsilon 必须满足 0 <= end <= start <= 1")
        if self.seed < 0 or self.eval_seed < self.seed + self.episodes:
            raise ValueError("训练种子必须非负，且训练/验证种子区间不能重叠")
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError("device 只能取 cpu、cuda 或 auto")
