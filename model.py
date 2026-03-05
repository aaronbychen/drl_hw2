"""
Filename: model.py
Authors: Aaron Chen (517971), Sherlock Zhang (518915)
"""

from hydra.utils import instantiate
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, activation):
        super(QNetwork, self).__init__()
        self.q_head = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            instantiate(activation),
            nn.Linear(hidden_size, hidden_size),
            instantiate(activation),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state):
        Qs = self.q_head(state)
        return Qs


class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, activation):
        super(DuelingQNetwork, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            instantiate(activation),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            instantiate(activation),
            nn.Linear(hidden_size, 1)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            instantiate(activation),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state):
        """
        Get the Q value of the current state and action using dueling network
        """
        ############################
        # YOUR IMPLEMENTATION HERE #
        features = self.feature_layer(state)

        value = self.value_head(features) # (B, 1)
        advantage = self.advantage_head(features) # (B, A)

        advantage_mean = advantage.mean(dim=1, keepdim=True) # (B, 1)
        Qs = value + (advantage - advantage_mean) # broadcast to (B, A)
        ############################
        return Qs
