import numpy as np
import gym
from gym import spaces


class GpuCapacityEnv(gym.Env):
    
    # 시계열 GPU 수요(gpu_demand)를 기반으로, 각 step마다 capacity level을 선택하는 PPO용 환경.
    

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        gpu_demand: np.ndarray,
        window_size: int = 12,
        capacity_levels=None,
        shortage_penalty: float = 2.0,
        idle_penalty: float = 1.0,
    ):
        super().__init__()

        self.gpu_demand = gpu_demand.astype(np.float32)
        self.T = len(gpu_demand)
        self.window_size = window_size
        self.shortage_penalty = shortage_penalty
        self.idle_penalty = idle_penalty

        self.max_demand = float(np.max(self.gpu_demand))

        # capacity level (GPU 개수 후보)
        if capacity_levels is None:
            # 4단계 capacity 설정
            levels = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
            self.capacity_levels = (levels * self.max_demand).astype(np.int32)
        else:
            self.capacity_levels = np.array(capacity_levels, dtype=np.int32)

        # state: 최근 window_size step의 gpu_demand (0~1 정규화)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.window_size,),
            dtype=np.float32,
        )

        # action: capacity_levels index
        self.action_space = spaces.Discrete(len(self.capacity_levels))

        self.current_step = None

    def _get_state(self):
        start = self.current_step - self.window_size
        if start < 0:
            pad_len = -start
            pad = np.zeros(pad_len, dtype=np.float32)
            hist = self.gpu_demand[: self.current_step]
            seq = np.concatenate([pad, hist])
        else:
            seq = self.gpu_demand[start:self.current_step]

        if len(seq) != self.window_size:
            seq = np.pad(seq, (self.window_size - len(seq), 0), mode="constant")

        # 정규화
        if self.max_demand > 0:
            seq = seq / self.max_demand
        return seq.astype(np.float32)

    def reset(self):
        # window_size 이후 시점에서 시작
        self.current_step = self.window_size
        return self._get_state()

    def step(self, action):
        demand = float(self.gpu_demand[self.current_step])
        capacity = float(self.capacity_levels[action])

        shortage = max(demand - capacity, 0.0)
        idle = max(capacity - demand, 0.0)

        reward = -(
            self.shortage_penalty * shortage
            + self.idle_penalty * idle
        )

        self.current_step += 1
        done = self.current_step >= self.T

        if done:
            next_state = np.zeros(self.window_size, dtype=np.float32)
        else:
            next_state = self._get_state()

        info = {
            "demand": demand,
            "capacity": capacity,
            "shortage": shortage,
            "idle": idle,
        }

        return next_state, reward, done, info

    def render(self, mode="human"):
        if self.current_step is None or self.current_step >= self.T:
            print("[Env] Not running.")
            return
        print(
            f"[Env] step={self.current_step}, "
            f"demand={self.gpu_demand[self.current_step]:.1f}, "
            f"capacity_levels={self.capacity_levels}"
        )
