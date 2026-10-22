"""验证网络配置确实控制隐藏层、非线性和线性 Q 输出。"""

import unittest

import torch
from torch import nn

from uav_demo.config import TrainConfig
from uav_demo.environment import GridWorldEnv
from uav_demo.model import QNetwork


class NetworkTests(unittest.TestCase):
    def test_default_has_three_hidden_layers_and_leaky_relu(self):
        config = TrainConfig()
        inputs = GridWorldEnv().observation_size
        network = QNetwork(inputs, 17, config.hidden_sizes, config.activation, config.leaky_relu_slope)
        linear_layers = [layer for layer in network.layers if isinstance(layer, nn.Linear)]
        activations = [layer for layer in network.layers if isinstance(layer, nn.LeakyReLU)]
        self.assertEqual([(layer.in_features, layer.out_features) for layer in linear_layers],
                         [(inputs, 128), (128, 128), (128, 128), (128, 17)])
        self.assertEqual(len(activations), 3)
        self.assertTrue(all(layer.negative_slope == 0.01 for layer in activations))
        self.assertEqual(tuple(network(torch.zeros(4, inputs)).shape), (4, 17))

    def test_variable_depth_widths_and_supported_activations_have_linear_output(self):
        activation_types = {
            "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU,
            "tanh": nn.Tanh, "silu": nn.SiLU,
        }
        for widths in ((6,), (12, 7), (9, 8, 5, 3)):
            for name, activation_type in activation_types.items():
                with self.subTest(widths=widths, activation=name):
                    network = QNetwork(2, 3, widths, name, 0.2)
                    self.assertEqual(sum(isinstance(layer, activation_type) for layer in network.layers), len(widths))
                    self.assertIsInstance(network.layers[-1], nn.Linear)
                    if name == "leaky_relu":
                        self.assertTrue(all(layer.negative_slope == 0.2 for layer in network.layers
                                            if isinstance(layer, nn.LeakyReLU)))
                    # 负 Q 值必须能原样输出，不能被 ReLU 或 Softmax 限制。
                    with torch.no_grad():
                        for parameter in network.parameters():
                            parameter.zero_()
                        network.layers[-1].bias.fill_(-2.0)
                    torch.testing.assert_close(network(torch.ones(2)), torch.full((3,), -2.0))

    def test_network_config_validation_and_json_width_list(self):
        self.assertEqual(TrainConfig(hidden_sizes=[32, 16]).hidden_sizes, (32, 16))
        cases = (
            {"hidden_sizes": ()}, {"hidden_sizes": (0,)}, {"hidden_sizes": (-1, 8)},
            {"hidden_sizes": (3.5,)}, {"hidden_sizes": (True,)}, {"hidden_sizes": 128},
            {"activation": "softmax"}, {"leaky_relu_slope": -0.1},
            {"leaky_relu_slope": 1.1}, {"leaky_relu_slope": float("nan")},
            {"leaky_relu_slope": float("inf")},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    TrainConfig(**changes)
                with self.assertRaises(ValueError):
                    QNetwork(2, 3, **changes)


if __name__ == "__main__":
    unittest.main()
