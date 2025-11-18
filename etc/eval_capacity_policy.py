import numpy as np
import pandas as pd
import torch

from config import GPU_DEMAND_CSV_PATH, EnvConfig, get_model_path
from envs.gpu_capacity_env import GpuCapacityEnv
from RL.actor_critic import ActorCritic


def eval_policy(model_path: str, episodes: int = 5):
    # 1) 데이터 로드 & 환경 생성
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

    # 3) 평가 루프
    all_rewards = []
    all_shortage = []
    all_idle = []

    for ep in range(episodes):
        state = env.reset()
        done = False

        ep_rewards = []
        ep_shortage = 0.0
        ep_idle = 0.0

        step = 0
        while not done:
            s_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, value = net.act_greedy(s_tensor)

            next_state, reward, done, info = env.step(action)

            ep_rewards.append(reward)
            ep_shortage += info["shortage"]
            ep_idle += info["idle"]

            state = next_state
            step += 1

        all_rewards.append(np.mean(ep_rewards))
        all_shortage.append(ep_shortage)
        all_idle.append(ep_idle)

        print(f"[Eval Ep {ep}] "
              f"avg_reward={np.mean(ep_rewards):.3f}, "
              f"total_shortage={ep_shortage:.1f}, total_idle={ep_idle:.1f}")

    print("========== Summary ==========")
    print(f"Mean avg_reward over episodes: {np.mean(all_rewards):.3f}")
    print(f"Mean total_shortage: {np.mean(all_shortage):.1f}")
    print(f"Mean total_idle   : {np.mean(all_idle):.1f}")


if __name__ == "__main__":
    
    from config import get_model_path
    model_path = get_model_path("ppo_capacity.pt")   # 또는 "ptr_ppo_capacity.pt"
    eval_policy(model_path, episodes=5)
