import numpy as np
import gym
from gym import spaces
import torch


class GpuCapacityEnv(gym.Env):
    
    # 시계열 GPU 수요(gpu_demand)를 기반으로, 각 step마다 capacity level을 선택하는 PPO용 환경.
    

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        gpu_demand: np.ndarray,
        window_size: int = 10,
        capacity_levels=None,
        shortage_penalty: float = 2.0,
        idle_penalty: float = 1.0,
        
        # 추가: 예측 모델 관련
        forecast_model: torch.nn.Module | None = None,
        use_forecast: bool = False,
    ):
        super().__init__()

        self.gpu_demand = gpu_demand.astype(np.float32)
        self.window_size = window_size
        self.shortage_penalty = shortage_penalty
        self.idle_penalty = idle_penalty
        self.max_demand = float(np.max(self.gpu_demand))

        if capacity_levels is None:
            levels = np.linspace(0.1, 1.0, 10, dtype=np.float32)
            self.capacity_levels = (levels * self.max_demand).astype(np.int32)
        else:
            self.capacity_levels = np.array(capacity_levels, dtype=np.int32)

        

        # 예측 모델 세팅
        self.forecast_model = forecast_model
        self.use_forecast = use_forecast and (forecast_model is not None)

        if self.use_forecast:
            self.forecast_model.eval()
            self.forecast_device = next(forecast_model.parameters()).device
            # 예측 horizon=1을 쓴다고 가정
            self.forecast_horizon = 1
        else:
            self.forecast_device = "cpu"
            self.forecast_horizon = 0

        self.T = 0
        self._build_spaces()


    def _build_spaces(self):
        # 원래 window_size만 state였는데, 예측값 하나를 추가하면 차원이 window_size + 1이 됨
        obs_dim = self.window_size + (1 if self.use_forecast else 0)

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Discrete(len(self.capacity_levels))

    def _get_state(self) -> np.ndarray:
        """
        현재 시점 t에서 상태를 구성:
        - 기본: 최근 window_size 길이의 demand (정규화)
        - LSTM + PPO : + 예측된 다음 demand (정규화)
        """
        start = max(0, self.T - self.window_size + 1)
        end = self.T + 1
        window = self.gpu_demand[start:end]

        # 초기 구간 채우기
        if len(window) < self.window_size:
            pad = np.full(self.window_size - len(window), window[0], dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)

        # 간단히 max_demand로 나눠서 0~1 스케일
        window_norm = window / (self.max_demand + 1e-8)   # shape (window_size,)

        if not self.use_forecast:
            return window_norm.astype(np.float32)

        # 예측값 추가
        with torch.no_grad():
            x = torch.tensor(window_norm, dtype=torch.float32, device=self.forecast_device)
            x = x.view(1, -1, 1)  # (B=1, L=window_size, 1)
            y_hat = self.forecast_model(x)  # (1, forecast_horizon)
            pred_next = y_hat[0, 0].item()  # 하나만 쓴다고 가정

        # pred_next는 이미 정규화된 스케일에서 나온 값이므로 그대로 사용
        pred_vec = np.array([pred_next], dtype=np.float32)

        state = np.concatenate([window_norm, pred_vec], axis=0)  # (window_size+1,)
        return state.astype(np.float32)
    

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
