import os
import pandas as pd

base_dir = "D:\project\pinnTrasfer\data"   # 你的 data 目录
conditions = ["Normal", "Leak", "Block", "Worn"]
splits = ["train", "val", "test"]

for split in splits:
    dfs = []

    for cond in conditions:
        # 文件名：NormalTrain.csv / LeakTrain.csv / ...
        fname = f"{cond}{split.capitalize()}.csv"
        fpath = os.path.join(base_dir, split, fname)

        print("loading:", fpath)
        df = pd.read_csv(fpath)

        # 如果想保留工况信息，可以加一列；不需要就注释掉
        # df["condition"] = cond

        dfs.append(df)

    # 按顺序 Normal -> Leak -> Block -> Worn 纵向拼接
    merged = pd.concat(dfs, axis=0, ignore_index=True)

    # 输出文件名：Train.csv / Val.csv / Test.csv
    out_name = f"{split.capitalize()}.csv"
    out_path = os.path.join(base_dir, out_name)

    merged.to_csv(out_path, index=False)
    print("saved:", out_path)

print("done.")
