"""Independent small critics; no imports from historical RL implementations."""

import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, observation_dim, action_dim=7, width=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim + action_dim, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, observation, action):
        return self.net(torch.cat((observation, action), dim=-1)).squeeze(-1)


class TwinCritic(nn.Module):
    def __init__(self, observation_dim, width=256):
        super().__init__()
        self.q1 = Critic(observation_dim, width=width)
        self.q2 = Critic(observation_dim, width=width)

    def forward(self, observation, action):
        return self.q1(observation, action), self.q2(observation, action)


def calibrated_penalty(q_sampled, q_behavior, return_to_go):
    """Eq. 5.1 of Cal-QL, finite proposal expectation, without logsumexp CQL.

    The published expectation form is used directly. Proposal actions are
    normalized perturbations of the logged reference; never robot commands.
    Unknown MC values are masked by the caller, never assigned zero.
    """
    return torch.maximum(q_sampled, return_to_go[:, None]).mean(1) - q_behavior
