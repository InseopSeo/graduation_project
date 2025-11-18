# src/RL/trajectory_buffer.py

import numpy as np


class TrajectoryReplayBuffer:
    """
    PTR-PPO 스타일의 trajectory-level replay buffer.

    각 trajectory는 dict 형태로 저장:
      {
        "states":      np.ndarray (T, state_dim),
        "actions":     np.ndarray (T,),
        "log_probs":   np.ndarray (T,),
        "returns":     np.ndarray (T,),
        "advantages":  np.ndarray (T,),
      }

    priority는 보통 mean(|advantages|) 또는 max(|advantages|) 등으로 설정.
    """

    def __init__(self, capacity: int = 128, alpha: float = 0.6):
        """
        Parameters
        ----------
        capacity : int
            저장 가능한 trajectory 개수.
        alpha : float
            priority exponent (0이면 uniform, 1.0에 가까울수록 priority 차이를 강하게 반영).
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: list[dict] = []
        self.priorities: list[float] = []

    def __len__(self):
        return len(self.buffer)

    def add(self, trajectory: dict, priority: float):
        """
        새로운 trajectory를 버퍼에 추가. capacity를 넘으면 FIFO로 가장 오래된 항목을 교체.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(trajectory)
            self.priorities.append(float(priority))
        else:
            # 가장 오래된 trajectory를 교체
            self.buffer.pop(0)
            self.priorities.pop(0)
            self.buffer.append(trajectory)
            self.priorities.append(float(priority))

    def update_priority(self, idx: int, priority: float):
        """
        버퍼 내 특정 trajectory의 priority를 갱신.
        """
        self.priorities[idx] = float(priority)

    def sample(self, num_samples: int):
        """
        priority 기반으로 trajectory를 샘플링.

        Returns
        -------
        trajectories : list[dict]
        indices : list[int]
        sample_probs : np.ndarray
            샘플링된 trajectory들의 선택 확률 (importance correction에 사용할 수 있음).
        """
        n = len(self.buffer)
        if n == 0:
            return [], [], np.array([])

        # priority: 확률 분포
        prios = np.array(self.priorities, dtype=np.float64)
        prios = np.clip(prios, 1e-8, None)  # 0 방지
        prios = prios ** self.alpha
        probs = prios / prios.sum()

        # replace=True로 허용 (buffer가 작은 경우)
        idxs = np.random.choice(n, size=num_samples, p=probs, replace=(n < num_samples))
        trajectories = [self.buffer[i] for i in idxs]
        sample_probs = probs[idxs]

        return trajectories, idxs.tolist(), sample_probs
