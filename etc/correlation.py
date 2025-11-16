import pandas as pd
from scipy.stats import pointbiserialr

df = pd.read_csv("disaggregated_DLRM_trace.csv")

# numeric columns
num_cols = [c for c in df.columns if df[c].dtype in ["int64", "float64"]]

# point biserial correlation
corrs = {}
for c in num_cols:
    if c != "gpu_request":
        try:
            corrs[c] = pointbiserialr(df["gpu_request"], df[c])[0]
        except:
            corrs[c] = None

# sort by absolute correlation
sorted_corr = sorted(corrs.items(), key=lambda x: abs(x[1]) if x[1] is not None else 0, reverse=True)

# show top correlated features
print("Top correlated features with gpu_request:")
for f, v in sorted_corr[:]:
    print(f"{f}: {v:.3f}")
