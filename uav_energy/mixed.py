"""混合动作分布：二维移动用高斯分布，感知、查询、到站用伯努利分布。

PPO 的更新由 Stable-Baselines3 完成，这里只定义动作如何采样和计算概率。
参考 OpenAI Spinning Up 的策略分工：
https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/core.py
"""

import torch
from torch import nn
from torch.distributions import Bernoulli, Normal
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class MoveSwitchDistribution(DiagGaussianDistribution):
    def proba_distribution_net(self, latent_dim, log_std_init=0.0):
        # 网络输出：移动均值 x/y，以及三个开关的 logits。
        action_net = nn.Linear(latent_dim, 5)
        log_std = nn.Parameter(torch.full((2,), float(log_std_init)))
        return action_net, log_std

    def proba_distribution(self, mean_actions, log_std):
        move_mean = mean_actions[:, :2].tanh()
        move_std = log_std.exp().clamp(0.05, 1.0)
        switch_probability = 0.02 + 0.96 * mean_actions[:, 2:].sigmoid()
        self.distribution = Normal(move_mean, move_std)
        self.gates = Bernoulli(probs=switch_probability)
        return self

    def log_prob(self, actions):
        move_log_prob = self.distribution.log_prob(actions[:, :2]).sum(-1)
        # 环境用 -1/+1 表示关/开；伯努利分布需要 0/1。
        switches = (actions[:, 2:] + 1) / 2
        switch_log_prob = self.gates.log_prob(switches).sum(-1)
        return move_log_prob + switch_log_prob

    def entropy(self):
        return self.distribution.entropy().sum(-1) + self.gates.entropy().sum(-1)

    def sample(self):
        move = self.distribution.rsample()
        switches = 2 * self.gates.sample() - 1
        return torch.cat([move, switches], dim=-1)

    def mode(self):
        move = self.distribution.mean
        switches = 2 * (self.gates.probs >= 0.5).float() - 1
        return torch.cat([move, switches], dim=-1)


class MoveSwitchPolicy(ActorCriticPolicy):
    def _build(self, lr_schedule):
        self.action_dist = MoveSwitchDistribution(5)
        super()._build(lr_schedule)
