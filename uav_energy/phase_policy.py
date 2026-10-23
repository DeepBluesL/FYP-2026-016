"""采集和返航各用一组策略/价值网络，全部参数仍由 PPO 学习。

before/after 等属性名保留，以兼容已经保存的模型。
"""

import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from .mixed import MoveSwitchDistribution


class PhaseLinear(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.before = nn.Linear(dim, out)
        self.after = nn.Linear(dim, out)

    def forward(self, x):
        head_size = (x.shape[1] - 1) // 2
        complete = x[:, -1:]
        before_features = x[:, :head_size]
        after_features = x[:, head_size : 2 * head_size]
        return (1 - complete) * self.before(before_features) + complete * self.after(
            after_features
        )


class PhaseDistribution(MoveSwitchDistribution):
    def proba_distribution_net(self, latent_dim, log_std_init=0.0):
        return PhaseLinear((latent_dim - 1) // 2, 5), nn.Parameter(
            torch.full((2,), float(log_std_init))
        )


class PhaseExtractor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.latent_dim_pi = self.latent_dim_vf = 257

        # 四个独立 MLP：采集策略、返航策略、采集价值、返航价值。
        def net():
            return nn.Sequential(
                nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh()
            )

        self.before_pi, self.after_pi, self.before_vf, self.after_vf = (
            net(),
            net(),
            net(),
            net(),
        )

    def combine(self, x, before, after):
        # 输入堆叠了四帧，每帧 17 维；最新帧第 7 项表示是否采满 K。
        frame_size = 17
        complete_index = 7
        index = -frame_size + complete_index
        complete = x[:, index : index + 1]
        before_features = (1 - complete) * before(x)
        after_features = complete * after(x)
        return torch.cat([before_features, after_features, complete], dim=1)

    def forward_actor(self, x):
        return self.combine(x, self.before_pi, self.after_pi)

    def forward_critic(self, x):
        return self.combine(x, self.before_vf, self.after_vf)

    def forward(self, x):
        return self.forward_actor(x), self.forward_critic(x)


class PhasePolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor = PhaseExtractor(self.features_dim)

    def _build(self, lr_schedule):
        self.action_dist = PhaseDistribution(5)
        super()._build(lr_schedule)


def initialize_from(model, parent):
    """把父 PPO 的权重复制到两个阶段，不使用规则动作或模仿数据。"""
    current, old = model.policy, parent.policy
    for net in (current.mlp_extractor.before_pi, current.mlp_extractor.after_pi):
        net.load_state_dict(old.mlp_extractor.policy_net.state_dict())
    for net in (current.mlp_extractor.before_vf, current.mlp_extractor.after_vf):
        net.load_state_dict(old.mlp_extractor.value_net.state_dict())
    for head in (current.action_net.before, current.action_net.after):
        head.load_state_dict(old.action_net.state_dict())
    with torch.no_grad():
        current.log_std.copy_(old.log_std)
        # 价值层输入依次为 128 维采集特征、128 维返航特征和阶段标志。
        current.value_net.weight[:, :128].copy_(old.value_net.weight)
        current.value_net.weight[:, 128:256].copy_(old.value_net.weight)
        current.value_net.weight[:, 256:].zero_()
        current.value_net.bias.copy_(old.value_net.bias)
    model.num_timesteps = parent.num_timesteps
