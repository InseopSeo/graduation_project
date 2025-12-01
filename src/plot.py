import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from matplotlib.lines import Line2D
from pathlib import Path


try:
    from config import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(".").resolve()

def main():
    log_dir = PROJECT_ROOT / "logs"
    
    # 파일 경로 정의
    files = {
        "PPO (Base)": log_dir / "ppo_train_log.csv",
        "PTR-PPO (Base)": log_dir / "ptr_ppo_train_log.csv",
        "PPO (+LSTM)": log_dir / "hybrid_ppo_train_log.csv",
        "PTR-PPO (+LSTM)": log_dir / "hybrid_ptr_ppo_train_log.csv"
    }

    data = {}
    for label, path in files.items():
        if path.exists():
            data[label] = pd.read_csv(path)
        else:
            print(f"[Warning] File not found: {path}")

    if not data:
        print("표시할 데이터가 없습니다.")
        return


    fig, ax = plt.subplots(figsize=(12, 7))
    

    plt.subplots_adjust(left=0.3)  

    lines = []
    labels = []
    
    # 색상 지정
    colors = {
        "PPO (Base)": "blue", 
        "PTR-PPO (Base)": "orange", 
        "PPO (+LSTM)": "green", 
        "PTR-PPO (+LSTM)": "red"
    }


    for label, df in data.items():
        if "iter" in df.columns and "avg_reward" in df.columns:
            c = colors.get(label, "black")
            line, = ax.plot(df["iter"], df["avg_reward"], label=label, color=c, linewidth=1.5)
            lines.append(line)
            labels.append(label)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Reward")
    ax.set_title("Learning Curve Comparison")
    ax.grid(True)


    # 위치: [left, bottom, width, height]
    # 그래프 왼쪽(0.05), 높이는 중간쯤(0.5)에 배치
    rax = plt.axes([0.05, 0.5, 0.15, 0.2])
    
    check = CheckButtons(rax, labels, [True] * len(labels))

    # 클릭 이벤트
    def func(label):
        index = labels.index(label)
        lines[index].set_visible(not lines[index].get_visible())
        plt.draw()

    check.on_clicked(func)


    # 범례가 시작될 Y좌표 (체크박스 바로 아래)
    start_y = 0.45 
    gap = 0.04  # 줄 간격

    # "Color Legend" 타이틀 표시
    fig.text(0.05, start_y, "Color Legend:", fontsize=10, fontweight='bold', transform=fig.transFigure)
    start_y -= gap

    for label in labels:
        c = colors.get(label, "black")
        
        # 1) 색상 선 그리기 (Line2D)
        # x좌표: 0.05 ~ 0.08 (짧은 선)
        # y좌표: start_y
        line_legend = Line2D([0.05, 0.08], [start_y, start_y], 
                             transform=fig.transFigure, color=c, linewidth=2)
        fig.add_artist(line_legend)
        
        # 2) 모델 이름 텍스트 쓰기
        fig.text(0.09, start_y, label, fontsize=9, verticalalignment='center', transform=fig.transFigure)
        
        # 다음 줄로 이동
        start_y -= gap

    plt.show()

if __name__ == "__main__":
    main()