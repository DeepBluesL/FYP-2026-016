import torch
from torch import nn
from torch.distributions import Categorical


from .model import build_mlp


class PPOActorCritic(nn.Module):
    def __init__(self, observation_size, action_count,
                 hidden_sizes=(128, 128, 128), activation="tanh", leaky_relu_slope=0.01):
        super().__init__()
        self.actor = build_mlp(observation_size, action_count, hidden_sizes, activation, leaky_relu_slope)
        self.critic = build_mlp(observation_size, 1, hidden_sizes, activation, leaky_relu_slope)

    def distribution(self, features: torch.Tensor, mask: torch.Tensor) -> Categorical:
        # 让 Actor 给动作打分，再根据当前状态的 mask 构造概率分布
        logits = self.actor(features)
        return Categorical(logits=logits.masked_fill(~mask, -torch.inf))

    def value(self, features: torch.Tensor) -> torch.Tensor:
        return self.critic(features).squeeze(-1)

    @torch.no_grad()
    def step(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[int, float, float]:
        # 对一条已编码的观测采样，返回动作、V(s) 和该动作的 log 概率
        policy = self.distribution(features, mask)
        action = policy.sample()
        return action.item(), self.value(features).item(), policy.log_prob(action).item()
