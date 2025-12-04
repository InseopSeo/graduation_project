# graduation_project

본 프로젝트는 **Alibaba GPU Trace** 데이터를 활용하여, 클라우드 환경에서의 GPU 오토스케일링(Capacity Control) 효율을 올리기 위해 강화학습 모델을 구현한 졸업 프로젝트입니다.

기존의 PPO의 문제점인 낮은 샘플 효율성을 개선하기 위해 PTR-PPO를 도입하고 미래의 수요를 예측하여 대비하기 위해 LSTM까지 도입했습니다.

즉, LSTM과 PTR-PPO를 파이프라인으로 연결한 모델입니다.

## 환경 준비

Python Ver. 
3.10.11

```bash
pip install -r requirements.txt
```

## 데이터 전처리

```bash
python -m src.preprocess.build_gpu_demand
```

raw trace(`data/raw/disaggregated_DLRM_trace.csv`)를 읽어서 X초(또는 분) 단위의 GPU 수요(`data/processed/gpu_demand_Xsec(min).csv`)를 생성합니다.



## LSTM 모델 생성

데이터 전처리를 한 뒤에 LSTM 모델을 생성합니다.

```bash
python src/build_forecast_model.py
```

기본적으로 src/train_hybrid_ppo.py 파일과 src/train_hybrid_ptr_ppo.py 파일에 학습 기능이 있지만, 이 코드를 먼저 실행하여 명시적으로 LSTM 모델을 먼저 학습합니다.

## PPO 학습 실행

### PPO + LSTM X
```bash
python src/train_capacity_ppo.py 
```

### PTR-PPO + LSTM X
```bash
python src/train_capacity_ptr_ppo.py 
```

### PPO + LSTM O
```bash
python src/train_hybrid_ppo.py 
```

### PTR-PPO + LSTM O
```bash
python src/train_hybrid_ptr_ppo.py 
```

### 모델끼리 비교하는 명령어
```bash
python src/plot.py
```
  
`src/config.py`에서 환경/학습 하이퍼파라미터와 경로를 통일적으로 관리합니다.
