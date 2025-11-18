# src/RL/actor_critic.py

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)  # (B,)
        return logits, value

    def act(self, state):
        """
        state: torch.tensor (state_dim,) or (1, state_dim)
        return: action(int), log_prob(tensor), value(tensor)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.squeeze(0), value.squeeze(0)

    def act_greedy(self, state):
        """
        평가용
        확률 샘플링 대신 argmax 기반으로 행동 선택
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        logits, value = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        action = torch.argmax(probs, dim=-1)
        
        return action.item(), value.squeeze(0)