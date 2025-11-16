# utils.py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
class TensorDataset(Dataset):
    """自定义数据集类，用于包装张量数据"""
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def standardize_tensor(data, mode, mean=0, std=1):
    """
    标准化张量数据
    data: (..., features)
    mode: 'fit' or 'transform'
    """
    data_2D = data.contiguous().view((-1, data.shape[-1]))
    if mode == 'fit':
        mean = torch.mean(data_2D, dim=0)
        std = torch.std(data_2D, dim=0)
    data_norm_2D = (data_2D - mean) / (std + 1e-8)
    data_norm = data_norm_2D.contiguous().view((-1, data.shape[-2], data.shape[-1]))
    return data_norm, mean, std


def inverse_standardize_tensor(data_norm, mean, std):
    """
    反标准化张量数据
    """
    data_norm_2D = data_norm.contiguous().view((-1, data_norm.shape[-1]))
    data_2D = data_norm_2D * std + mean
    data = data_2D.contiguous().view(
        (-1, data_norm.shape[-2], data_norm.shape[-1])
    )
    return data


def calculate_metrics_in_batches(predictions, targets, batch_size=1024):
    """
    分批计算 RMSE / MAE / MAPE，节省内存
    """
    total_samples = predictions.size(0)
    rmse_sum = 0.0
    mae_sum = 0.0
    mape_sum = 0.0
    count = 0

    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        pred_batch = predictions[i:end_idx]
        target_batch = targets[i:end_idx]

        batch_rmse = torch.sqrt(torch.mean((pred_batch - target_batch) ** 2))
        batch_mae = torch.mean(torch.abs(pred_batch - target_batch))

        non_zero_indices = target_batch != 0
        if torch.any(non_zero_indices):
            batch_mape = torch.mean(
                torch.abs(
                    (pred_batch[non_zero_indices] - target_batch[non_zero_indices])
                    / target_batch[non_zero_indices]
                )
            ) * 100
        else:
            batch_mape = torch.tensor(0.0, device=pred_batch.device)

        batch_size_actual = end_idx - i
        rmse_sum += batch_rmse * batch_size_actual
        mae_sum += batch_mae * batch_size_actual
        mape_sum += batch_mape * batch_size_actual
        count += batch_size_actual

    rmse = rmse_sum / count
    mae = mae_sum / count
    mape = mape_sum / count
    return rmse, mae, mape


def calculate_r2_in_batches(predictions, targets, batch_size=1024):
    """
    分批计算 R²
    """
    if predictions.dim() > 1:
        predictions = predictions.squeeze()
    if targets.dim() > 1:
        targets = targets.squeeze()

    total_samples = predictions.size(0)

    # mean of target
    target_mean_sum = 0.0
    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        target_batch = targets[i:end_idx]
        target_mean_sum += torch.sum(target_batch)
    target_mean = target_mean_sum / total_samples

    sum_squared_error = 0.0
    sum_squared_total = 0.0

    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        pred_batch = predictions[i:end_idx]
        target_batch = targets[i:end_idx]

        sum_squared_error += torch.sum((target_batch - pred_batch) ** 2)
        sum_squared_total += torch.sum((target_batch - target_mean) ** 2)

    epsilon = 1e-10
    if sum_squared_total < epsilon:
        print(f"Warning: sum_squared_total is very small: {sum_squared_total}")
        return torch.tensor(0.0)

    print(f"SSE: {sum_squared_error.item()}, SST: {sum_squared_total.item()}")
    r2 = 1 - (sum_squared_error / sum_squared_total)
    return r2
def load_condition_split_csv(root_dir, split, condition,
                             target_col="pOut",
                             drop_cols=None,
                             device="cpu"):
    """
    root_dir: 数据根目录，比如 "data"
    split: "train" / "val" / "test"
    condition: "Normal" / "Leak" / "Block" / "Worn"
    文件名假设形如: NormalTrain.csv / LeakVal.csv ...
    """
    if drop_cols is None:
        drop_cols = [target_col,"pOut"]  # 按你之前的写法

    filename = f"{condition}{split.capitalize()}.csv"  # e.g. NormalTrain.csv
    path = f"{root_dir}/{split}/{filename}"

    df = pd.read_csv(path)
    X = df.drop(drop_cols, axis=1)
    y = df[target_col]

    X_tensor = torch.tensor(X.values, dtype=torch.float64).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.float64).unsqueeze(1).to(device)

    return X_tensor, y_tensor


class EarlyStopping:
    def __init__(self, patience=20, delta=0.0, save_path="best_model.pth"):
        """
        patience:   容忍多少个 epoch 验证损失不下降
        delta:      最小下降幅度（小于这个视为没有改善）
        save_path:  自动保存最好模型
        """
        self.patience = patience
        self.delta = delta
        self.save_path = save_path

        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):

        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)

        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            print(f"⚠ 早停计数 {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                print("⛔ Early Stopping triggered !")
                self.early_stop = True

        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.save_path)
        print(f"✔ 验证损失改善，保存模型: {self.save_path}  (val_loss: {val_loss:.6f})")
