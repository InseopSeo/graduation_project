# src/train_capacity_ptr_ppo.py

import csv
import numpy as np
import pandas as pd
import torch

from config import GPU_DEMAND_CSV_PATH, PROJECT_ROOT, EnvConfig, PPOConfig, get_model_path
from envs.gpu_capacity_env import GpuCapacityEnv
from RL.ptr_ppo_agent import PTRPPOAgent


def main():
    env_cfg = EnvConfig()
    ppo_cfg = PPOConfig()

    # 1) 데이터 로드
    df = pd.read_csv(GPU_DEMAND_CSV_PATH)
    gpu_demand = df["gpu_demand"].to_numpy()

    # 2) 환경 생성
    env = GpuCapacityEnv(
        gpu_demand=gpu_demand,
        window_size=env_cfg.window_size,
        capacity_levels=env_cfg.capacity_levels,
        shortage_penalty=env_cfg.shortage_penalty,
        idle_penalty=env_cfg.idle_penalty,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 3) PTR-PPO 에이전트 생성
    agent = PTRPPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=ppo_cfg.gamma,
        lam=ppo_cfg.lam,
        clip_eps=ppo_cfg.clip_eps,
        lr=ppo_cfg.lr,
        update_epochs=ppo_cfg.update_epochs,
        batch_size=ppo_cfg.batch_size,
        device=ppo_cfg.device,
        # replay buffer parameters
        buffer_capacity=128,
        replay_k=4,
        alpha=0.6,
        iw_clip=3.0,
    )

    # 4) 로그 파일 준비
    log_path = PROJECT_ROOT / "logs" / "ptr_ppo_train_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "avg_reward", "mean_shortage", "mean_idle"])

        # 5) 학습 루프 (학습 + 로그)
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

            on_policy_traj = {
                "states": states,
                "actions": actions,
                "log_probs": log_probs_old,
                "returns": returns,
                "advantages": advantages,
            }

            # 새 trajectory를 버퍼에 추가 (초기 priority = mean(|A|))
            init_prio = float(np.mean(np.abs(advantages)))
            agent.replay_buffer.add(on_policy_traj, init_prio)

            # PTR-PPO 업데이트
            agent.update_with_replay(on_policy_traj)

            # 로그용 통계 (현재는 avg_reward만 진행, shortage/idle은 0.0으로 placeholder)
            avg_reward = rewards.mean()
            mean_shortage = 0.0  # rollout에서 shortage/idle 모으고 싶으면 추후 수정
            mean_idle = 0.0

            print(f"[PTR-PPO Iter {it:03d}] avg_reward = {avg_reward:.3f}")
            writer.writerow([it, avg_reward, mean_shortage, mean_idle])

    # 6) 학습 종료 & 최종 모델 저장
    save_path = get_model_path("ptr_ppo_capacity.pt")
    torch.save(agent.net.state_dict(), save_path)
    print(f"[INFO] Saved PTR-PPO model to {save_path}")


if __name__ == "__main__":
    main()