import os
import pandas as pd
import matplotlib.pyplot as plt

# 数据主目录
base_dir = "../data"

# 时间列可能的名字（自动识别）
time_candidates = ["time", "Time", "t", "T", "Cycle", "Cycles"]

# 遍历 train / val / test
for split in ["train", "val", "test"]:
    folder_path = os.path.join(base_dir, split)
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    for csv in csv_files:
        csv_path = os.path.join(folder_path, csv)

        # 加载数据
        df = pd.read_csv(csv_path)

        # 自动查找时间列
        time_col = None
        for c in time_candidates:
            if c in df.columns:
                time_col = c
                break

        if time_col is None:
            print(f"⚠ 找不到时间列（time/t/Cycle）: {csv_path}")
            continue

        if "pOut" not in df.columns:
            print(f"⚠ 找不到 pOut 列: {csv_path}")
            continue

        # 绘图
        plt.figure(figsize=(10, 4))
        plt.plot(df[time_col], df["pOut"], linewidth=1.0)

        plt.title(f"{split} - {csv} (pOut Trend)")
        plt.xlabel("Time")
        plt.ylabel("pOut")
        plt.grid(True)

        # 保存到 images/{split}/xxx.png
        save_dir = os.path.join("../images", split)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, csv.replace(".csv", ".png"))
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✔ 已保存: {save_path}")
