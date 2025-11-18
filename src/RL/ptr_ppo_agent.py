# src/RL/ptr_ppo_agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .actor_critic import ActorCritic
from .trajectory_buffer import TrajectoryReplayBuffer


class PTRPPOAgent:
    """
    PTR-PPO 스타일의 에이전트:
      - 기본 PPO(clip) 구조 유지
      - trajectory-level replay buffer 사용
      - priority 기반 샘플링
      - truncated importance weight 적용
    """

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
        # PTR-PPO 관련 설정
        buffer_capacity: int = 16,
        replay_k: int = 1,
        alpha: float = 0.4,     # priority exponent
        iw_clip: float = 1.5,   # importance weight clipping
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.device = device

        self.net = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        # PTR-PPO 관련 멤버
        self.replay_k = replay_k
        self.iw_clip = iw_clip
        self.replay_buffer = TrajectoryReplayBuffer(
            capacity=buffer_capacity,
            alpha=alpha,
        )


    # Rollout & GAE 계산

    def rollout(self, env, horizon: int = 2048):
        """
        환경에서 horizon 길이만큼 trajectory를 수집.

        Returns
        -------
        states, actions, rewards, dones, log_probs_old, values, last_value
        (모두 numpy 배열 또는 float)
        """
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

        # 마지막 상태의 value for bootstrap
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
        """
        Generalized Advantage Estimation (GAE-Lambda)
        """
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


    # PTR-PPO 업데이트 로직

    def update_with_replay(self, on_policy_traj: dict):
        """
        on-policy trajectory + replay buffer에서 샘플한 trajectory들을
        합쳐서 하나의 큰 batch로 만드는 PTR-PPO 업데이트.
        """

        # 1) on-policy trajectory 우선 사용
        traj_list = [on_policy_traj]

        # 2) trajectory-level replay buffer에서 몇 개 샘플
        replay_trajs = []
        replay_indices = []
        if len(self.replay_buffer) > 0 and self.replay_k > 0:
            replay_trajs, replay_indices, _ = self.replay_buffer.sample(self.replay_k)
            traj_list.extend(replay_trajs)

        # 3) 모든 trajectory를 concat해서 big batch 구성
        states = np.concatenate([t["states"] for t in traj_list], axis=0)
        actions = np.concatenate([t["actions"] for t in traj_list], axis=0)
        log_probs_old = np.concatenate([t["log_probs"] for t in traj_list], axis=0)
        returns = np.concatenate([t["returns"] for t in traj_list], axis=0)
        advantages = np.concatenate([t["advantages"] for t in traj_list], axis=0)

        # 4) PPO + truncated importance weight 코어 업데이트
        self._ppo_core_update(
            states,
            actions,
            log_probs_old,
            returns,
            advantages,
            iw_clip=self.iw_clip,
        )

        # 5) 샘플된 replay trajectory들의 priority 업데이트 (선택적)
        #    여기서는 간단하게 mean(|advantages|)로 priority 갱신
        if len(replay_trajs) > 0:
            # on-policy 길이
            on_len = len(on_policy_traj["states"])
            offset = on_len
            for buf_idx, traj in zip(replay_indices, replay_trajs):
                T = len(traj["states"])
                adv_seg = advantages[offset:offset + T]
                new_prio = float(np.mean(np.abs(adv_seg)))
                self.replay_buffer.update_priority(buf_idx, new_prio)
                offset += T

    def _ppo_core_update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        log_probs_old: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
        iw_clip: float = 3.0,
    ):
        """
        PPO-Clip + truncated importance weight를 이용한 코어 업데이트.
        """

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device)
        logp_old_t = torch.tensor(log_probs_old, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        dataset_size = states_t.size(0)

        for _ in range(self.update_epochs):
            idxs = np.arange(dataset_size)
            np.random.shuffle(idxs)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = idxs[start:end]

                s_b = states_t[batch_idx]
                a_b = actions_t[batch_idx]
                logp_old_b = logp_old_t[batch_idx]
                ret_b = returns_t[batch_idx]
                adv_b = adv_t[batch_idx]

                logits, values = self.net(s_b)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(a_b)
                entropy = dist.entropy().mean()

                # importance ratio
                ratio = torch.exp(logp - logp_old_b)

                # truncated importance weight (PTR-PPO 스타일)
                iw = torch.clamp(ratio, 0.0, iw_clip)

                # advantage normalization
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                # PPO-Clip surrogate
                surr1 = ratio * adv_b
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_eps,
                    1.0 + self.clip_eps,
                ) * adv_b

                # PTR: surrogate에 importance weight를 곱해줌
                surr = torch.min(surr1, surr2) * iw

                policy_loss = -surr.mean()
                value_loss = 0.5 * (ret_b - values).pow(2).mean()
                loss = policy_loss + value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
