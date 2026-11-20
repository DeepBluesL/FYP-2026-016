import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Module):
    # 左侧补零，保证不会读取未来时刻
    def __init__(self, in_channels, out_channels, kernel_size, dilation, ):
        super().__init__()

        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


# 两层因果卷积+一条残差连接
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, ):
        super().__init__()

        self.layers = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.LeakyReLU(),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.LeakyReLU()
        )

        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return F.leaky_relu(self.layers(x) + self.residual(x))


class TCNPredictor(nn.Module):
    def __init__(self, hidden_size=256, head_size=128, num_blocks=4, kernel_size=3, max_steps=150,
                 ):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.max_steps = max_steps

        blocks = []
        in_channels = 3

        for index in range(num_blocks):
            blocks.append(TemporalBlock(
                in_channels=in_channels,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                dilation=2 ** index,
            ))
            in_channels = hidden_size

        self.encoder = nn.Sequential(*blocks)

        self.position_head = nn.Sequential(
            nn.Linear(hidden_size, head_size),
            nn.LeakyReLU(),
            nn.Linear(head_size, 2),
        )

        self.receptive_field = (1 + 2 * (kernel_size - 1) * (2 ** num_blocks - 1))

    def forward(self, features):
        # [B, T, 3] → [B, 3, T]
        hidden = self.encoder(features.transpose(1, 2))

        # [B, 256, T] → [B, T, 256]
        positions = self.position_head(hidden.transpose(1, 2))

        # 保持 run_epoch() 中 predicted, _ = model(features) 的接口。
        return positions, None

    @torch.no_grad()
    def predict_positions(self, history, grid_size, steps, move_every=1, ):
        self.eval()

        device = next(self.parameters()).device
        scale = grid_size - 1

        features = torch.tensor(
            [[x / scale, y / scale, time / self.max_steps]
             for time, (x, y) in history
             ], dtype=torch.float32, device=device, ).unsqueeze(0)

        # 更早的输入不会影响最后一个输出，可以在推理时省去
        features = features[:, -self.receptive_field:, :]

        forecasts = [np.asarray(history[-1][1], dtype=np.float32)]
        report_time = history[-1][0]

        for offset in range(1, max(steps) + 1):
            predicted, _ = self(features)

            next_position = predicted[:, -1:, :].clamp(0.0, 1.0)

            forecasts.append(next_position[0, 0].cpu().numpy() * scale)

            if offset < max(steps):
                next_time = next_position.new_full(
                    (1, 1, 1), (report_time + offset) / self.max_steps,
                )

                next_input = torch.cat((next_position, next_time), dim=-1, )

                features = torch.cat((features, next_input), dim=1, )
                features = features[:, -self.receptive_field:, :]

        return np.stack([forecasts[h] for h in steps])

    def export_state(self):
        return {
            "model_type": "tcn",
            "hidden_size": self.hidden_size,
            "head_size": self.head_size,
            "num_blocks": self.num_blocks,
            "kernel_size": self.kernel_size,
            "max_steps": self.max_steps,
            "state_dict": {
                name: value.detach().cpu().clone()
                for name, value in self.state_dict().items()
            },
        }
