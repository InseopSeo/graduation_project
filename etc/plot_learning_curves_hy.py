# src/plot_learning_curves_hy.py
# 학습 곡선 비교 플롯 생성
#  LSTM 적용 버전 PTR-PPO vs PPO

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import PROJECT_ROOT


def main():
    log_dir = PROJECT_ROOT / "logs"
    ppo_log_path = log_dir / "hybrid_ppo_train_log.csv"
    ptr_log_path = log_dir / "hybrid_ptr_ppo_train_log.csv"

    if not ppo_log_path.exists():
        raise FileNotFoundError(f"PPO log not found: {ppo_log_path}")
    if not ptr_log_path.exists():
        raise FileNotFoundError(f"PTR-PPO log not found: {ptr_log_path}")

    df_ppo = pd.read_csv(ppo_log_path)
    df_ptr = pd.read_csv(ptr_log_path)


    # 1) Avg Reward 곡선

    plt.figure()
    plt.plot(df_ppo["iter"], df_ppo["avg_reward"], label="PPO")
    plt.plot(df_ptr["iter"], df_ptr["avg_reward"], label="PTR-PPO")
    plt.xlabel("Iteration")
    plt.ylabel("Average Reward (rollout)")
    plt.title("PPO vs PTR-PPO - Learning Curve (Avg Reward)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 2) Shortage / Idle 곡선 (추후에 쓸 용도)
    # 지금은 logging에서 0.0으로 두었으면 의미 없음. rollout에서 shortage/idle도 기록하게 바꾼 후에 사용하면 됨.

    if "mean_shortage" in df_ppo.columns and "mean_shortage" in df_ptr.columns:
        if (df_ppo["mean_shortage"].abs().sum() > 0) or (df_ptr["mean_shortage"].abs().sum() > 0):
            plt.figure()
            plt.plot(df_ppo["iter"], df_ppo["mean_shortage"], label="PPO")
            plt.plot(df_ptr["iter"], df_ptr["mean_shortage"], label="PTR-PPO")
            plt.xlabel("Iteration")
            plt.ylabel("Mean Shortage per Step")
            plt.title("PPO vs PTR-PPO - Mean Shortage")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    if "mean_idle" in df_ppo.columns and "mean_idle" in df_ptr.columns:
        if (df_ppo["mean_idle"].abs().sum() > 0) or (df_ptr["mean_idle"].abs().sum() > 0):
            plt.figure()
            plt.plot(df_ppo["iter"], df_ppo["mean_idle"], label="PPO")
            plt.plot(df_ptr["iter"], df_ptr["mean_idle"], label="PTR-PPO")
            plt.xlabel("Iteration")
            plt.ylabel("Mean Idle per Step")
            plt.title("PPO vs PTR-PPO - Mean Idle")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

# ------------------------------------------------------------


if __name__ == "__main__":
    main()
