# src/train_hybrid_ppo.py

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
from RL.ppo_agent import PPOAgent
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

    # 4) PPO 에이전트 생성
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=ppo_cfg.gamma,
        lam=ppo_cfg.lam,
        clip_eps=ppo_cfg.clip_eps,
        lr=ppo_cfg.lr,
        update_epochs=ppo_cfg.update_epochs,
        batch_size=ppo_cfg.batch_size,
        device=device,
    )

    # 5) 학습 루프
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

        avg_reward = rewards.mean()
        print(f"[Hybrid PPO Iter {it:03d}] avg_reward = {avg_reward:.3f}")

    # 6) 모델 저장
    save_path = get_model_path("ppo_hybrid_capacity.pt")
    torch.save(agent.net.state_dict(), save_path)
    print(f"[INFO] Saved Hybrid PPO model to {save_path}")


if __name__ == "__main__":
    main()
