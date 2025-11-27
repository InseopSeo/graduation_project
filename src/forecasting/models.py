# src/forecasting/models.py

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ForecastModelConfig:
    input_size: int = 1
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    forecast_horizon: int = 1 # 예측할 step 수


class DemandLSTM(nn.Module):
    """
    시계열 수요 예측용 LSTM 모델.
    입력: (B, L, input_size)
    출력: (B, forecast_horizon)
    """

    def __init__(self, cfg: ForecastModelConfig):
        super().__init__()
        self.cfg = cfg

        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        # 마지막 hidden state → forecast_horizon 길이 예측
        self.fc = nn.Linear(cfg.hidden_size, cfg.forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            shape (batch_size, seq_len, input_size)

        Returns
        -------
        y_hat : torch.Tensor
            shape (batch_size, forecast_horizon)
        """
        # lstm_out: (B, L, H), (h_n, c_n)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # h_n: (num_layers, B, H) → 마지막 layer의 hidden만 사용
        h_last = h_n[-1]  # (B, H)

        y_hat = self.fc(h_last)  # (B, forecast_horizon)
        return y_hat


class DemandGRU(nn.Module):
    """
    시계열 수요 예측용 GRU 모델.
    입력/출력 형태는 DemandLSTM과 동일.
    """

    def __init__(self, cfg: ForecastModelConfig):
        super().__init__()
        self.cfg = cfg

        self.gru = nn.GRU(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(cfg.hidden_size, cfg.forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, input_size) -> y_hat: (B, forecast_horizon)
        """
        gru_out, h_n = self.gru(x)   # h_n: (num_layers, B, H)
        h_last = h_n[-1]            # (B, H)
        y_hat = self.fc(h_last)
        return y_hat
