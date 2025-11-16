import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드
df = pd.read_csv("disaggregated_DLRM_trace.csv")

# GPU 요청만 필터링
gpu_df = df[df["gpu_request"] == 1].copy()

# 결측치 처리
gpu_df["creation_time"].fillna(0, inplace=True)
gpu_df["deletion_time"].fillna(gpu_df["deletion_time"].max(), inplace=True)

# 전체 시간 구간 설정 (초 단위), (3600초 = 1시간)
time_index = np.arange(0, gpu_df["deletion_time"].max(), step=600)

# 각 시간대별 GPU 사용량 계산
usage_counts = []
for t in time_index:
    active = ((gpu_df["creation_time"] <= t) & (gpu_df["deletion_time"] > t)).sum()
    usage_counts.append(active)

# 결과 시각화
plt.plot(time_index, usage_counts)
plt.xlabel("Time (s)")
plt.ylabel("Active GPU Instances")
plt.title("GPU Usage Over Time")
plt.show()
