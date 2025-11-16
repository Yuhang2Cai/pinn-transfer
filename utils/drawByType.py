import os
import pandas as pd
import matplotlib.pyplot as plt

# 数据集主目录
base_dir = "../data"

# 工况列表
conditions = ["Normal", "Block", "Leak", "Worn"]

# 三种数据集线型
linestyle_map = {
    "train": "-",
    "val": "--",
    "test": ":"
}

# 时间列可能的名称
time_candidates = ["time", "Time", "t", "T", "Cycle", "Cycles"]

# 输出文件夹
output_dir = "../condition_plots"
os.makedirs(output_dir, exist_ok=True)

for cond in conditions:
    plt.figure(figsize=(12, 5))

    for split in ["train", "val", "test"]:
        folder_path = os.path.join(base_dir, split)
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

        # 找工况对应的文件（如 NormalTrain.csv）
        target_file = None
        for f in csv_files:
            if cond.lower() in f.lower():  # 用工况名匹配文件名
                target_file = f
                break

        if target_file is None:
            print(f"⚠ 未找到 {cond}-{split} 文件")
            continue

        csv_path = os.path.join(folder_path, target_file)

        # 读取 CSV
        df = pd.read_csv(csv_path)

        # 找时间列
        time_col = None
        for c in time_candidates:
            if c in df.columns:
                time_col = c
                break

        if time_col is None:
            print(f"⚠ {csv_path} 中没有找到时间列")
            continue

        if "pOut" not in df.columns:
            print(f"⚠ {csv_path} 中没有 pOut 列")
            continue

        # 绘制曲线
        plt.plot(
            df[time_col],
            df["pOut"],
            linestyle=linestyle_map[split],
            label=f"{split}"
        )

    plt.title(f"{cond} Condition - pOut Trend")
    plt.xlabel("Time")
    plt.ylabel("pOut")
    plt.grid(True)
    plt.legend()

    save_path = os.path.join(output_dir, f"{cond}_pOut.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✔ 已生成图： {save_path}")
