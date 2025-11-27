# src/train_hybrid_ptr_ppo.py

import csv
import numpy as np
import pandas as pd
import torch

from config import (
    GPU_DEMAND_CSV_PATH,
    EnvConfig,
    PPOConfig,
    PROJECT_ROOT,
    get_model_path,
)
from envs.gpu_capacity_env import GpuCapacityEnv
from RL.ptr_ppo_agent import PTRPPOAgent
from forecasting.models import ForecastModelConfig, DemandLSTM


def load_forecast_model(device: str = "cpu") -> torch.nn.Module:
    """
    train_forecast.py에서 학습한 LSTM 모델 로드.
    - config의 hidden_size, num_layers, forecast_horizon은
      train_forecast.py에서 쓴 값과 동일해야 함.
    """
    model_cfg = ForecastModelConfig(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        forecast_horizon=1,
    )
    model = DemandLSTM(model_cfg).to(device)

    ckpt_path = PROJECT_ROOT / "models" / "forecast_lstm.pt"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[INFO] Loaded forecast model from {ckpt_path}")
    return model


def main():
    env_cfg = EnvConfig()
    ppo_cfg = PPOConfig()

    device = ppo_cfg.device
    print(f"[INFO] Using device: {device}")

    # 1) demand 시계열 로드
    df = pd.read_csv(GPU_DEMAND_CSV_PATH)
    gpu_demand = df["gpu_demand"].to_numpy(dtype=np.float32)

    # 2) 예측 모델 로드
    forecast_model = load_forecast_model(device=device)

    # 3) 하이브리드 환경 생성 (예측 사용)
    env = GpuCapacityEnv(
        gpu_demand=gpu_demand,
        window_size=env_cfg.window_size,          # 이 값이 train_forecast의 input_window와 동일하면 베스트
        capacity_levels=env_cfg.capacity_levels,
        shortage_penalty=env_cfg.shortage_penalty,
        idle_penalty=env_cfg.idle_penalty,
        forecast_model=forecast_model,
        use_forecast=True,
    )

    state_dim = env.observation_space.shape[0]   # window_size + 1
    action_dim = env.action_space.n

    # 4) PTR-PPO 에이전트 생성 (replay 포함)
    agent = PTRPPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=ppo_cfg.gamma,
        lam=ppo_cfg.lam,
        clip_eps=ppo_cfg.clip_eps,
        lr=ppo_cfg.lr,
        update_epochs=ppo_cfg.update_epochs,
        batch_size=ppo_cfg.batch_size,
        device=device,
        buffer_capacity=16,
        replay_k=2,
        alpha=0.4,
        iw_clip=1.3,
    )

    # 5) 로그 파일 준비
    log_path = PROJECT_ROOT / "logs" / "ptr_hybrid_ppo_train_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "avg_reward", "mean_shortage", "mean_idle"])

        # 6) 학습 루프 (on-policy + replay)
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

            # PTR-PPO 업데이트 (replay 섞어서 학습)
            agent.update_with_replay(on_policy_traj)

            # 간단한 로그 통계
            avg_reward = rewards.mean()
            mean_shortage = 0.0  # 필요 시 env.step info 수집하여 갱신
            mean_idle = 0.0

            print(f"[Hybrid PTR-PPO Iter {it:03d}] avg_reward = {avg_reward:.3f}")
            writer.writerow([it, avg_reward, mean_shortage, mean_idle])

    # 7) 모델 저장
    save_path = get_model_path("ptr_ppo_hybrid_capacity.pt")
    torch.save(agent.net.state_dict(), save_path)
    print(f"[INFO] Saved Hybrid PTR-PPO model to {save_path}")


if __name__ == "__main__":
    main()
