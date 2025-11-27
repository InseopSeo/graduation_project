# src/forecasting/train_forecast.py

from __future__ import annotations

import os
import math
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from forecasting.dataset import (
    ForecastDatasetConfig,
    WindowedDemandDataset,
    split_train_val
)

from forecasting.models import (
    ForecastModelConfig,
    DemandLSTM,
    DemandGRU,
)

from config import GPU_DEMAND_CSV_PATH, PROJECT_ROOT


# -----------------------------
# 학습 설정값
# -----------------------------
@dataclass
class TrainForecastConfig:
    input_window: int = 24             # 입력 시퀀스 길이
    forecast_horizon: int = 1          # 예측 시점 수
    batch_size: int = 64
    num_epochs: int = 20
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_type: str = "LSTM"           # "LSTM" 또는 "GRU"
    train_ratio: float = 0.8
    shuffle: bool = True
    num_workers: int = 0               # In Windows, set to 0


def compute_metrics(y_true, y_pred):
    """
    RMSE, MAE 계산 함수.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    return rmse, mae


# -----------------------------
# 학습 루프
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)  # (B, L, 1)
        y = y.to(device)  # (B, horizon)

        optimizer.zero_grad()
        y_hat = model(x)  # (B, horizon)

        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    preds = []
    trues = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)

        loss = criterion(y_hat, y)
        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

        preds.append(y_hat.cpu().numpy())
        trues.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    rmse, mae = compute_metrics(trues, preds)

    return total_loss / total_samples, rmse, mae


# -----------------------------
# 메인 학습 프로세스
# -----------------------------
def main():
    cfg = TrainForecastConfig()

    print(f"[INFO] Device: {cfg.device}")
    print(f"[INFO] Loading series from: {GPU_DEMAND_CSV_PATH}")

    # 1) demand CSV 로드
    df = pd.read_csv(GPU_DEMAND_CSV_PATH)
    series = df["gpu_demand"].to_numpy(dtype=np.float32)

    # 2) Dataset 생성
    ds_cfg = ForecastDatasetConfig(
        input_window=cfg.input_window,
        forecast_horizon=cfg.forecast_horizon,
        stride=1,
        normalize=True,
    )

    ds_train, ds_val = split_train_val(series, train_ratio=cfg.train_ratio, cfg=ds_cfg)

    loader_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers
    )

    loader_val = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )

    # 3) 모델 생성
    model_cfg = ForecastModelConfig(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        forecast_horizon=cfg.forecast_horizon,
    )

    if cfg.model_type.upper() == "LSTM":
        model = DemandLSTM(model_cfg)
    else:
        model = DemandGRU(model_cfg)

    model = model.to(cfg.device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # 4) 학습
    print(f"[INFO] Training {cfg.model_type} for {cfg.num_epochs} epochs...")

    best_val_rmse = math.inf
    best_state = None

    for epoch in range(cfg.num_epochs):
        train_loss = train_one_epoch(model, loader_train, criterion, optimizer, cfg.device)
        val_loss, val_rmse, val_mae = evaluate(model, loader_val, criterion, cfg.device)

        print(
            f"[Epoch {epoch+1:03d}/{cfg.num_epochs}] "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_RMSE={val_rmse:.3f}, "
            f"val_MAE={val_mae:.3f}"
        )

        # best model 저장
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = model.state_dict()

    # 5) 모델 저장
    save_dir = PROJECT_ROOT / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"forecast_{cfg.model_type.lower()}.pt"

    torch.save(best_state, save_path)
    print(f"[INFO] Saved best model to {save_path}")
    print(f"[INFO] Best RMSE: {best_val_rmse:.3f}")


if __name__ == "__main__":
    main()
