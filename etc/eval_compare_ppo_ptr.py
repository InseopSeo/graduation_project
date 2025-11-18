# src/eval_compare_ppo_ptr.py

import numpy as np
import pandas as pd
import torch

from config import GPU_DEMAND_CSV_PATH, EnvConfig, get_model_path
from envs.gpu_capacity_env import GpuCapacityEnv
from RL.actor_critic import ActorCritic


def eval_policy(model_path: str, episodes: int = 5, label: str = ""):

    # 주어진 모델 capacity 환경에서 평가 및 요약 통계 반환

    # 1) 데이터 & 환경 준비
    df = pd.read_csv(GPU_DEMAND_CSV_PATH)
    gpu_demand = df["gpu_demand"].to_numpy()

    env_cfg = EnvConfig()
    env = GpuCapacityEnv(
        gpu_demand=gpu_demand,
        window_size=env_cfg.window_size,
        capacity_levels=env_cfg.capacity_levels,
        shortage_penalty=env_cfg.shortage_penalty,
        idle_penalty=env_cfg.idle_penalty,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 2) 모델 로드
    device = "cpu"
    net = ActorCritic(state_dim, action_dim).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    def act_greedy(state):
        s = torch.tensor(state, dtype=torch.float32, device=device)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        with torch.no_grad():
            logits, value = net(s)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1)
        return action.item()

    all_rewards = []
    all_shortage = []
    all_idle = []

    for ep in range(episodes):
        state = env.reset()
        done = False

        ep_rewards = []
        ep_shortage = 0.0
        ep_idle = 0.0

        while not done:
            action = act_greedy(state)
            next_state, reward, done, info = env.step(action)

            ep_rewards.append(reward)
            ep_shortage += info["shortage"]
            ep_idle += info["idle"]

            state = next_state

        avg_r = float(np.mean(ep_rewards))
        all_rewards.append(avg_r)
        all_shortage.append(ep_shortage)
        all_idle.append(ep_idle)

        print(
            f"[{label} Ep {ep}] "
            f"avg_reward={avg_r:.3f}, "
            f"total_shortage={ep_shortage:.1f}, total_idle={ep_idle:.1f}"
        )

    summary = {
        "label": label,
        "mean_avg_reward": float(np.mean(all_rewards)),
        "mean_total_shortage": float(np.mean(all_shortage)),
        "mean_total_idle": float(np.mean(all_idle)),
    }

    print(f"\n[{label}] Summary over {episodes} episodes")
    print(f"  mean_avg_reward   = {summary['mean_avg_reward']:.3f}")
    print(f"  mean_total_shortage = {summary['mean_total_shortage']:.1f}")
    print(f"  mean_total_idle   = {summary['mean_total_idle']:.1f}\n")

    return summary


def main():
    # PPO / PTR-PPO 모델 경로
    ppo_model_path = get_model_path("ppo_capacity.pt")
    ptr_model_path = get_model_path("ptr_ppo_capacity.pt")

    episodes = 50  # 에피소드 수

    print("========== Evaluate PPO ==========")
    ppo_summary = eval_policy(str(ppo_model_path), episodes=episodes, label="PPO")

    print("========== Evaluate PTR-PPO ==========")
    ptr_summary = eval_policy(str(ptr_model_path), episodes=episodes, label="PTR-PPO")

    # 최종 비교 출력
    print("========== PPO vs PTR-PPO Comparison ==========")
    for key in ["mean_avg_reward", "mean_total_shortage", "mean_total_idle"]:
        print(
            f"{key} : "
            f"PPO={ppo_summary[key]:.3f}  |  PTR-PPO={ptr_summary[key]:.3f}"
        )


if __name__ == "__main__":
    main()
