# src/preprocess/build_gpu_demand.py

import os
import numpy as np
import pandas as pd
from pathlib import Path

# config.py에서 설정 가져오기
from src.config import RAW_TRACE_PATH, GPU_DEMAND_CSV_PATH, PreprocessConfig

# 600초 = 10분
def build_gpu_demand_series(df: pd.DataFrame, bin_size: int = 600):
    """

    Parameters
    ----------
    df : pd.DataFrame
        Alibaba GPU job trace
    bin_size : int
        시간 bin 크기(초). 기본값 600초 = 10분

    Returns
    -------
    time_bins : np.ndarray
        각 bin의 시작 시간 (초 단위)
    gpu_demand : np.ndarray
        해당 bin에서 동시에 실행 중인 GPU job 수
    """


    # 1) GPU job만 필터링 (gpu_request == 1)

    if "gpu_request" not in df.columns:
        raise ValueError("df에 'gpu_request' 컬럼이 없습니다.")

    gpu_df = df[df["gpu_request"] == 1].copy()


    # 2) 시간 컬럼 numeric 변환 + NaN 처리

    for col in ["creation_time", "deletion_time"]:
        gpu_df[col] = pd.to_numeric(gpu_df[col], errors="coerce")

    # creation_time/deletion_time NaN 처리:
    # - creation_time NaN → 0 (trace 시작 전)
    # - deletion_time NaN → max_time (trace 끝 이후까지)
    max_time = np.nanmax(gpu_df[["creation_time", "deletion_time"]].to_numpy())
    gpu_df["creation_time"] = gpu_df["creation_time"].fillna(0)
    gpu_df["deletion_time"] = gpu_df["deletion_time"].fillna(max_time)


    # 3) 전체 시간축을 bin_size 만큼으로 쪼갬

    num_bins = int(max_time // bin_size) + 1
    time_bins = np.arange(0, num_bins * bin_size, bin_size, dtype=np.int64)

    gpu_demand = np.zeros(num_bins, dtype=np.int32)


    # 4) 각 job이 실행 중인 시간 구간을 bin index로 변환해서 demand += 1

    for _, row in gpu_df.iterrows():
        start_bin = int(row["creation_time"] // bin_size)
        end_bin = int(row["deletion_time"] // bin_size)

        # 인덱스 안전하게 클리핑
        start_bin = max(0, min(start_bin, num_bins - 1))
        end_bin = max(0, min(end_bin, num_bins - 1))

        gpu_demand[start_bin:end_bin + 1] += 1

    return time_bins, gpu_demand


def main():
    # config.py에 정의한 전처리 설정
    cfg = PreprocessConfig()

    print(f"[INFO] Loading raw trace file: {RAW_TRACE_PATH}")
    df = pd.read_csv(RAW_TRACE_PATH)

    print("[INFO] Building GPU demand series...")
    time_bins, gpu_demand = build_gpu_demand_series(df, bin_size=cfg.bin_size)

    # 출력 DataFrame 구성
    df_out = pd.DataFrame({
        "time_bin": time_bins,
        "gpu_demand": gpu_demand
    })

    # 출력 경로 생성
    GPU_DEMAND_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    # CSV로 저장
    df_out.to_csv(GPU_DEMAND_CSV_PATH, index=False)
    print(f"[INFO] Saved processed GPU demand to: {GPU_DEMAND_CSV_PATH}")

    # 샘플 출력
    print(df_out.head())


if __name__ == "__main__":
    main()
