# src/preprocess/build_gpu_demand.py
import os
import pandas as pd
from .build_gpu_demand import build_gpu_demand_series
from config import RAW_TRACE_PATH, GPU_DEMAND_CSV_PATH, PreprocessConfig

def main():
    cfg = PreprocessConfig()

    print(f"[INFO] Load: {RAW_TRACE_PATH}")
    df = pd.read_csv(RAW_TRACE_PATH)

    time_bins, gpu_demand = build_gpu_demand_series(df, bin_size=cfg.bin_size)

    df_out = pd.DataFrame({
        "time_bin": time_bins,
        "gpu_demand": gpu_demand
    })
    GPU_DEMAND_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(GPU_DEMAND_CSV_PATH, index=False)
    print(f"[INFO] Saved gpu_demand to: {GPU_DEMAND_CSV_PATH}")
