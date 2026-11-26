"""集中管理场景参数和训练参数；修改默认值后重新训练即可。"""

from .observation import DEFAULT_OBSERVATION_FEATURES

from dataclasses import dataclass
import math

ACTIVATION_NAMES = ("relu", "leaky_relu", "tanh", "silu")


def validate_network_settings(hidden_sizes, activation, leaky_relu_slope) -> tuple[int, ...]:
    """构造 QNetwork 时校验网络结构。"""
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
    max_steps: int = 150
    uav_start: tuple[int, int] = (0, 0)
    uav_goal: tuple[int, int] = (49, 49)
    initial_speed: int = 1

    car_move_every: int = 1  # 每隔多少个环境步移动 1 格；1 为每步一格，2 为两步一格
    observation_delay: int = 0  # 车辆的延迟观测环境step
    car_straight_prob: float = 0.65
    car_turn_prob: float = 0.15  # 左、右转各自的基础概率
    car_reverse_prob: float = 0.05
    car_goal_bias: float = 1.0  # 用 exp(bias * 靠近终点的距离) 加权方向
    # fixed：起点右下角、终点左上角，按时到站并停留；random：每局随机起点（离 UAV 起点 8–15 格）
    # 和随机目的地（至少 20 格外），到达后换新目的地继续行驶，策略无法从训练中记住路线。
    car_route: str = "fixed"

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
    auto_return: bool = False  # 采集结束后只允许沿最短路径飞向终点的动作；返程不再交给策略学习
    # True：进度与接触奖励改为势函数塑形 γΦ(s')−Φ(s)（Φ = −进度权重·距离 + 接触奖励·采集进度，
    # 终止时 Φ=0，不改变最优策略，也不能靠反复进出刷分）；不再计非法移动扣分（动作掩码已屏蔽）；
    # 随机路线超时仍未采完按采集失败扣分，而不是只扣超时分。
    reward_v2: bool = False
    shaping_gamma: float = 0.99  # reward_v2 势函数塑形的折扣，应与 DQN 的 gamma 一致


@dataclass
class TrainConfig:
    episodes: int = 1500
    total_steps: int | None = None
    eval_every_steps: int | None = None  # 默认 rollout_steps 的 10 倍
    test_seed: int = 3_000_000
    test_episodes: int = 200  # 与 spec 的表3 测试（3000000 起 200 条路线）一致
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

    hidden_sizes: tuple[int, ...] = (64, 64, 64)  # 每个元素定义一个隐藏层的宽度
    activation: str | None = None  # 未指定时，PPO 使用 tanh，DQN 使用 leaky_relu
    leaky_relu_slope: float = 0.01
    seed: int = 42
    device: str = "cpu"
    torch_threads: int = 1
    eval_every: int = 100
    eval_episodes: int = 50  # 验证局数；太少会让最佳模型的挑选不稳定
    eval_seed: int = 1_000_000
    log_every: int = 25

    use_motion_prediction: bool = True
    prediction_routes: int = 1000  # 三组路线的总数，不受 DQN episodes 限制
    prediction_seed: int = 100_000
    prediction_split_seed: int = 42
    prediction_model_type: str = "lstm"
    prediction_point: str = "disk"
    rule_source: str = "linear"
    prediction_tcn_blocks: int = 4
    prediction_tcn_kernel_size: int = 3
    prediction_var_lags: int = 4  # VAR 使用最近多少个报告时刻的位置
    prediction_learning_rate: float = 0.001
    prediction_batch_size: int = 32
    prediction_device: str = "auto"
    prediction_hidden_size: int = 256
    prediction_head_size: int = 128
    prediction_num_layers: int = 1
    prediction_epochs: int = 20

    observation_features: tuple[str, ...] = DEFAULT_OBSERVATION_FEATURES
    prediction_steps: tuple[int, ...] = (4, 8, 12)
    # >0 时启用外推规则先验：Q 值上规则动作 0、其余动作 −rule_prior（log 概率形式），
    # 被拟合的 Q 不含先验，先验只进入动作选择和下一步目标，偏离规则持续有代价（参考残差 Q 学习）。
    # 需在 observation_features 中加入 rule_action。
    rule_prior: float = 0.0
    algorithm: str = "dqn"
    rollout_steps: int = 1024
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    target_kl: float = 0.01
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    train_pi_iters: int | None = None  # 未指定时为 80
    train_v_iters: int | None = None  # 未指定时为 80
    update_iters: int | None = None  # 兼容旧配置；独立字段优先
    # False：规则动作只用于先验，不进网络输入（仍存进经验，供先验读取）；需 rule_prior > 0。
    rule_action_input: bool = True

    @property
    def needs_true_trajectory(self):
        return self.rule_source == "oracle" or (self.use_motion_prediction and self.prediction_model_type == "oracle")

    @property
    def network_uses_true_future(self):
        return (self.rule_source == "oracle" and "rule_action" in self.observation_features
                or self.use_motion_prediction and self.prediction_model_type == "oracle")

    def __post_init__(self):
        if not self.rule_action_input and (self.rule_prior <= 0 or self.algorithm != "dqn"):
            raise ValueError("rule_action_input=False 时规则动作只给 DQN 的规则先验用，需要 algorithm=dqn 且 rule_prior > 0")
        if self.rule_source not in ("linear", "predictor", "oracle"):
            raise ValueError("Unknown rule_source")
        if self.rule_source == "predictor" and not self.use_motion_prediction:
            raise ValueError("rule_source=predictor requires motion prediction")
        if self.prediction_point not in ("mode", "disk"):
            raise ValueError("Unknown prediction_point")
