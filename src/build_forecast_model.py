import torch

from config import PROJECT_ROOT
from forecasting.train_forecast import (TrainForecastConfig, train_forecast)
from forecasting.models import (ForecastModelConfig, DemandLSTM)


def load_forecast_model(device: str = "cpu") -> torch.nn.Module:
    """
    train_forecast.py에서 학습한 LSTM 모델 로드.
    - config의 hidden_size, num_layers, forecast_horizon은
      train_forecast.py에서 쓴 값과 동일해야 함.
    """

    ckpt_path = PROJECT_ROOT / "models" / "forecast_lstm.pt"

    if not ckpt_path.exists():
        print("[INFO] forecast_lstm.pt not found. Training forecast model first...")
        fcfg = TrainForecastConfig(
            device=device,
            model_type="LSTM",
        )
        ckpt_path = train_forecast(fcfg)
    else:
        print(f"[INFO] Found existing forecast model: {ckpt_path}")

    
    model_cfg = ForecastModelConfig(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        forecast_horizon=1,
    )

    model = DemandLSTM(model_cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"[INFO] Loaded forecast model from {ckpt_path}")
    return model

def main():
    load_forecast_model()

if __name__ == "__main__":
    main()