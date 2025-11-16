# src/config.py
from dataclasses import dataclass
from pathlib import Path



#  기본 경로 설정

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_TRACE_PATH = RAW_DATA_DIR / "disaggregated_DLRM_trace.csv"
GPU_DEMAND_CSV_PATH = PROCESSED_DATA_DIR / "gpu_demand_10min.csv"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)



#  전처리 / 시계열 설정

@dataclass
class PreprocessConfig:
    bin_size: int = 600  # seconds, 600 = 10분 단위


#  환경(Env) 설정

@dataclass
class EnvConfig:
    window_size: int = 12          # 최근 12 step = 2시간(10분 bin 기준)
    shortage_penalty: float = 2.0  # 부족 페널티 가중치
    idle_penalty: float = 1.0      # 낭비 페널티 가중치

    # capacity_levels를 직접 줄 수도 있고(None이면 auto)
    capacity_levels: list | None = None



#  PPO 하이퍼파라미터

@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    update_epochs: int = 10
    batch_size: int = 64
    horizon: int = 2048        # rollout 길이
    num_iterations: int = 50   # 학습 반복 수
    device: str = "cpu"
    seed: int = 42



#  모델 저장 경로

def get_model_path(name: str = "ppo_capacity.pt") -> Path:
    return MODEL_DIR / name
