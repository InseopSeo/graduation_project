# src/forecasting/dataset.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ForecastDatasetConfig:

    # 수요 예측용 데이터셋 설정 값.

    input_window: int = 10         # 입력 시퀀스 길이 (예: 24 step)
    forecast_horizon: int = 5      # 예측 시점 수 (1이면 one-step)
    stride: int = 1                # sliding window stride
    normalize: bool = True         # 평균 0, 표준편차 1 정규화 사용 여부


class WindowedDemandDataset(Dataset):
    """
    1D demand 시계열로부터 (input_window -> forecast_horizon) 샘플을 만드는 Dataset.

    예)
      series: [d0, d1, d2, ..., dN]
      input_window = 4, horizon = 2 이면:
        X[0] = [d0, d1, d2, d3],   y[0] = [d4, d5]
        X[1] = [d1, d2, d3, d4],   y[1] = [d5, d6]
        ...
    """

    def __init__(
        self,
        series: np.ndarray,
        input_window: int,
        forecast_horizon: int = 1,
        stride: int = 1,
        normalize: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        series : np.ndarray
            1D demand 시계열 (shape: [T] 또는 [T,]).
        input_window : int
            입력 시퀀스 길이.
        forecast_horizon : int
            예측할 시점 수 (1이면 one-step ahead).
        stride : int
            sliding window 이동 간격.
        normalize : bool
            True이면 전체 시계열 기준 (x - mean) / std 로 정규화.
        """
        super().__init__()

        series = np.asarray(series, dtype=np.float32).reshape(-1)
        self.original_series = series.copy()

        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        self.normalize = normalize

        if normalize:
            self.mean = float(series.mean())
            self.std = float(series.std() + 1e-8)
            series = (series - self.mean) / self.std
        else:
            self.mean = 0.0
            self.std = 1.0

        self.series = series

        # 가능한 window 개수 계산
        T = len(series)
        max_start = T - input_window - forecast_horizon
        if max_start < 0:
            raise ValueError(
                f"시계열 길이 {T}가 너무 짧습니다. "
                f"input_window={input_window}, forecast_horizon={forecast_horizon}"
            )

        indices = list(range(0, max_start + 1, stride))
        self.start_indices = np.array(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        x : torch.Tensor, shape (input_window, 1)
        y : torch.Tensor, shape (forecast_horizon,)
            또는 (forecast_horizon, 1)로 reshape해서 써도 됨.
        """
        start = int(self.start_indices[idx])
        end_x = start + self.input_window
        end_y = end_x + self.forecast_horizon

        x = self.series[start:end_x]           # (input_window,)
        y = self.series[end_x:end_y]           # (forecast_horizon,)

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)  # (L, 1)
        y = torch.tensor(y, dtype=torch.float32)                # (H,)

        return x, y


def split_train_val(
    series: np.ndarray,
    train_ratio: float = 0.8,
    cfg: Optional[ForecastDatasetConfig] = None,
) -> Tuple[WindowedDemandDataset, WindowedDemandDataset]:
    """
    전체 demand 시계열을 앞부분 train / 뒷부분 val로 나누고,
    각각 WindowedDemandDataset으로 감싸서 리턴.

    Parameters
    ----------
    series : np.ndarray
        전체 시계열
    train_ratio : float
        train 비율 (0~1).
    cfg : ForecastDatasetConfig
        input_window, forecast_horizon, stride, normalize 설정.

    Returns
    -------
    train_ds, val_ds : WindowedDemandDataset
    """
    if cfg is None:
        cfg = ForecastDatasetConfig()

    series = np.asarray(series, dtype=np.float32).reshape(-1)
    T = len(series)
    split_idx = int(T * train_ratio)

    train_series = series[:split_idx]
    val_series = series[split_idx:]

    train_ds = WindowedDemandDataset(
        train_series,
        input_window=cfg.input_window,
        forecast_horizon=cfg.forecast_horizon,
        stride=cfg.stride,
        normalize=cfg.normalize,
    )

    # val도 동일한 정규화 기준을 쓰고 싶다면, train mean/std를 넘기는 방식으로 확장 가능
    val_ds = WindowedDemandDataset(
        val_series,
        input_window=cfg.input_window,
        forecast_horizon=cfg.forecast_horizon,
        stride=cfg.stride,
        normalize=cfg.normalize,
    )

    return train_ds, val_ds
