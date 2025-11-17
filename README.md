# graduation_project

Alibaba GPU trace 기반으로 GPU 수요 시계열을 만들고 PPO로 capacity control 정책을 학습하는 프로젝트입니다.

## 환경 준비

```bash
pip install -r requirements.txt
```

## 데이터 전처리

```bash
python -m src.preprocess.build_gpu_demand
```

raw trace(`data/raw/disaggregated_DLRM_trace.csv`)를 읽어서 10분 bin 단위의 GPU 수요(`data/processed/gpu_demand_@min.csv`)를 생성합니다.

## PPO 학습 실행

```bash
python src/train_capacity_ppo.py
```

`src/config.py`에서 환경/학습 하이퍼파라미터와 경로를 통일적으로 관리합니다.
