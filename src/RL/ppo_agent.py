# src/RL/ppo_agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .actor_critic import ActorCritic


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        lr: float = 3e-4,
        update_epochs: int = 10,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.device = device

        self.net = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def rollout(self, env, horizon: int = 2048):
        states, actions = [], []
        rewards, dones = [], []
        log_probs, values = [], []

        state = env.reset()

        for _ in range(horizon):
            s_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            action, log_prob, value = self.net.act(s_tensor)

            next_state, reward, done, info = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob.detach().cpu().numpy())
            values.append(value.detach().cpu().numpy())

            state = next_state
            if done:
                state = env.reset()

        # 마지막 상태 value for bootstrap
        s_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        _, last_value = self.net.forward(s_tensor.unsqueeze(0))
        last_value = last_value.detach().cpu().numpy()[0]

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            np.array(log_probs, dtype=np.float32),
            np.array(values, dtype=np.float32),
            last_value,
        )

    def compute_gae(self, rewards, dones, values, last_value):
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else values[t + 1]
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def ppo_update(self, states, actions, log_probs_old, returns, advantages):
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        dataset_size = states.size(0)

        for _ in range(self.update_epochs):
            idxs = np.arange(dataset_size)
            np.random.shuffle(idxs)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = idxs[start:end]

                s_b = states[batch_idx]
                a_b = actions[batch_idx]
                logp_old_b = log_probs_old[batch_idx]
                ret_b = returns[batch_idx]
                adv_b = advantages[batch_idx]

                logits, values = self.net(s_b)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(a_b)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - logp_old_b)
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                surr1 = ratio * adv_b
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                ) * adv_b

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (ret_b - values).pow(2).mean()
                loss = policy_loss + value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
