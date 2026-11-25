"""DQN 的 Q 网络；环境和训练主循环不放在这里。"""

from torch import nn

from .config import validate_network_settings


def build_mlp(observation_size, action_count, hidden_sizes, activation, leaky_relu_slope):
    hidden_sizes = validate_network_settings(
        hidden_sizes, activation, leaky_relu_slope
    )
    activation_factory = {
        "relu": nn.ReLU,
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=leaky_relu_slope),
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }[activation]
    layers = []
    input_size = observation_size
    for width in hidden_sizes:
        layers.extend((nn.Linear(input_size, width), activation_factory()))
        input_size = width

    # Q 值、动作 logits 和状态价值都使用线性输出。
    layers.append(nn.Linear(input_size, action_count))
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """输入当前状态及可选学习预测特征，输出各离散动作的 Q 值。"""

    def __init__(
        self,
        observation_size: int,
        action_count: int,
        hidden_sizes: tuple[int, ...] = (64, 64, 64),
        activation: str = "leaky_relu",
        leaky_relu_slope: float = 0.01,
    ):
        super().__init__()
        self.layers = build_mlp(observation_size, action_count, hidden_sizes,
                                activation or "leaky_relu", leaky_relu_slope)

    def forward(self, observation):
        return self.layers(observation)
