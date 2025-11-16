# src/train_capacity_ppo.py
import pandas as pd
import numpy as np
import torch

from config import GPU_DEMAND_CSV_PATH, EnvConfig, PPOConfig, get_model_path
from envs.gpu_capacity_env import GpuCapacityEnv
from rl.ppo_agent import PPOAgent


def main():
    env_cfg = EnvConfig()
    ppo_cfg = PPOConfig()

    df = pd.read_csv(GPU_DEMAND_CSV_PATH)
    gpu_demand = df["gpu_demand"].to_numpy()

    env = GpuCapacityEnv(
        gpu_demand=gpu_demand,
        window_size=env_cfg.window_size,
        capacity_levels=env_cfg.capacity_levels,
        shortage_penalty=env_cfg.shortage_penalty,
        idle_penalty=env_cfg.idle_penalty,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=ppo_cfg.gamma,
        lam=ppo_cfg.lam,
        clip_eps=ppo_cfg.clip_eps,
        lr=ppo_cfg.lr,
        update_epochs=ppo_cfg.update_epochs,
        batch_size=ppo_cfg.batch_size,
        device=ppo_cfg.device,
    )

    for it in range(ppo_cfg.num_iterations):
        (
            states,
            actions,
            rewards,
            dones,
            log_probs_old,
            values,
            last_value,
        ) = agent.rollout(env, horizon=ppo_cfg.horizon)

        advantages, returns = agent.compute_gae(
            rewards, dones, values, last_value
        )

        agent.ppo_update(
            states,
            actions,
            log_probs_old,
            returns,
            advantages,
        )

        print(f"[Iter {it:03d}] avg_reward = {rewards.mean():.3f}")

    save_path = get_model_path()
    torch.save(agent.net.state_dict(), save_path)
    print(f"[INFO] Saved model to {save_path}")


if __name__ == "__main__":
    main()
