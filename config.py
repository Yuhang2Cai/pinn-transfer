# config.py
import os
import torch
import matplotlib
import matplotlib.pyplot as plt

# 使用非交互式后端，方便服务器上画图
matplotlib.use('Agg')

# 画图默认配置
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.unicode_minus'] = False

# PyTorch 全局设置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_default_dtype(torch.float64)
torch.cuda.empty_cache()
torch.set_printoptions(precision=10)

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
BATCH_SIZE   = 128
NUM_EPOCH    = 200
NUM_LAYERS   = 6
NUM_NEURONS  = 256
NUM_ROUNDS   = 2
SEQ_LEN      = 1
LR           = 0.001

# 时间缩放
TIME_SCALE   = 10000.0

# 数据相关
CSV_PATH     = "combined_all_t_p.csv"   # 你的数据文件路径
INPUT_DIM    = 3                        # 现在输入维度是 3（去掉了 iMotor）
OUTPUT_DIM   = 1
TARGET_COL   = "pOut"
DROP_COLS    = ["pOut", "iMotor"]       # 从 X 里 drop 掉的列
