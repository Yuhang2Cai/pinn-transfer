import os
import itertools
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import time

from config import (
    DEVICE, BATCH_SIZE, NUM_EPOCH, TIME_SCALE,
    INPUT_DIM, OUTPUT_DIM
)
from utils import (
    TensorDataset, standardize_tensor,
    calculate_metrics_in_batches, calculate_r2_in_batches,
    load_condition_split_csv
)
from models import TriplexPINN
from losses import My_loss
import matplotlib.pyplot as plt


# ========= 早停 ==============
class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, save_path="best_tmp.pth"):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path

        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)


# ========= 训练一个模型（给超参搜索调用） ==============
def train_one_model(
    condition_name,
    num_layers,
    num_neurons,
    lr,
    init_log_lambda,
    batch_size,
    max_epochs,
    data_root="data",
    save_dir="hp_results"
):
    """
    在单个工况上训练一次模型, 返回验证集指标。
    """

    os.makedirs(save_dir, exist_ok=True)

    print(f"\n===== 超参数试验 =====")
    print(f"condition   : {condition_name}")
    print(f"num_layers  : {num_layers}")
    print(f"num_neurons : {num_neurons}")
    print(f"lr          : {lr}")
    print(f"init_log_l  : {init_log_lambda}")
    print(f"batch_size  : {batch_size}")
    print(f"max_epochs  : {max_epochs}")
    print("=================================\n")

    # 1. 加载同一工况的 train/val/test
    X_train, y_train = load_condition_split_csv(
        root_dir=data_root, split="train", condition=condition_name, device=DEVICE
    )
    X_val, y_val = load_condition_split_csv(
        root_dir=data_root, split="val", condition=condition_name, device=DEVICE
    )
    X_test, y_test = load_condition_split_csv(
        root_dir=data_root, split="test", condition=condition_name, device=DEVICE
    )

    inputs_train, inputs_val, inputs_test = X_train, X_val, X_test
    targets_train, targets_val, targets_test = y_train, y_val, y_test

    # 2. 标准化（只用 train 拟合）
    num_train = inputs_train.shape[0]
    _, mean_inputs_train, std_inputs_train = standardize_tensor(
        torch.reshape(inputs_train, (num_train, 1, INPUT_DIM)), mode='fit'
    )
    _, mean_targets_train, std_targets_train = standardize_tensor(
        targets_train, mode='fit'
    )

    # 3. DataLoader
    train_set = TensorDataset(inputs_train, targets_train)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )

    # 4. 模型 & 损失 & 优化器
    layers = [num_neurons] * num_layers
    model = TriplexPINN(
        seq_len=1,
        inputs_dim=INPUT_DIM,
        outputs_dim=OUTPUT_DIM,
        layers=layers,
        scaler_inputs=(mean_inputs_train, std_inputs_train),
        scaler_targets=(mean_targets_train, std_targets_train)
    ).to(DEVICE)

    criterion = My_loss(init_log_lambda=init_log_lambda).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1)

    # 早停
    tag = f"{condition_name}_L{num_layers}_N{num_neurons}_lr{lr:.0e}_logl{init_log_lambda}_ep{max_epochs}"
    best_model_path = os.path.join(save_dir, f"best_{tag}.pth")
    early_stopping = EarlyStopping(
        patience=10,
        delta=1e-5,
        save_path=best_model_path
    )

    train_losses = []
    val_losses = []
    lambda_history = []

    # 5. 训练循环
    for epoch in range(max_epochs):
        model.train()
        epoch_train_loss = 0.0

        with torch.backends.cudnn.flags(enabled=False):
            for batch_x, batch_y in train_loader:
                p_pred, P_t_pred = model(inputs=batch_x)

                loss = criterion(
                    targets_P=batch_y,
                    outputs_P=p_pred,
                    dpdt=P_t_pred,
                    mdot_A=batch_x[:, 2],
                    V=2 * np.exp(-4),
                    bulk_modulus_model='const',
                    air_dissolution_model='off',
                    rho_L_atm=851.6,
                    beta_L_atm=1.46696e+03,
                    beta_gain=0.2,
                    air_fraction=0.005,
                    rho_g_atm=1.225,
                    polytropic_index=1.0,
                    p_atm=0.101325,
                    p_crit=3,
                    p_min=1
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        inputs_val.requires_grad_(True)
        P_pred_val, P_t_pred_val = model(inputs=inputs_val)
        val_loss = criterion(
            targets_P=targets_val,
            outputs_P=P_pred_val,
            dpdt=P_t_pred_val,
            mdot_A=inputs_val[:, 2],
            V=2 * np.exp(-4),
            bulk_modulus_model='const',
            air_dissolution_model='off',
            rho_L_atm=851.6,
            beta_L_atm=1.46696e+03,
            beta_gain=0.2,
            air_fraction=0.005,
            rho_g_atm=1.225,
            polytropic_index=1.0,
            p_atm=0.101325,
            p_crit=3,
            p_min=1
        )

        val_loss_value = val_loss.item()
        val_losses.append(val_loss_value)

        # 记录当前 λ
        lambda_history.append(criterion.physics_weight.item())

        # 早停检查
        early_stopping(val_loss_value, model)

        print(f"[{tag}] Epoch {epoch+1}/{max_epochs}, "
              f"train_loss={avg_train_loss:.5f}, val_loss={val_loss_value:.5f}, "
              f"lambda={criterion.physics_weight.item():.4f}")

        if early_stopping.early_stop:
            print(f"🛑 {tag} 提前停止在 epoch {epoch+1}")
            break

        scheduler.step()

    # 加载该组合的最优模型
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        print(f"✔ 使用 {tag} 的最佳模型参数进行评估")
    else:
        print(f"⚠ 未找到 {best_model_path}，使用最后一轮模型评估")

    # 6. 评估指标（train/val/test）
    model.eval()
    P_pred_train, _ = model(inputs=inputs_train)
    P_pred_val, _ = model(inputs=inputs_val)
    P_pred_test, _ = model(inputs=inputs_test)

    rmse_train, mae_train, mape_train = calculate_metrics_in_batches(P_pred_train, targets_train)
    rmse_val, mae_val, mape_val = calculate_metrics_in_batches(P_pred_val, targets_val)
    rmse_test, mae_test, mape_test = calculate_metrics_in_batches(P_pred_test, targets_test)

    r2_train = calculate_r2_in_batches(P_pred_train, targets_train)
    r2_val   = calculate_r2_in_batches(P_pred_val, targets_val)
    r2_test  = calculate_r2_in_batches(P_pred_test, targets_test)

    metrics = {
        "rmse_train": rmse_train.item(),
        "mae_train": mae_train.item(),
        "mape_train": mape_train.item(),
        "r2_train": r2_train.item(),
        "rmse_val": rmse_val.item(),
        "mae_val": mae_val.item(),
        "mape_val": mape_val.item(),
        "r2_val": r2_val.item(),
        "rmse_test": rmse_test.item(),
        "mae_test": mae_test.item(),
        "mape_test": mape_test.item(),
        "r2_test": r2_test.item(),
    }

    # 可选：保存损失曲线和 lambda 历史
    history_df = pd.DataFrame({
        "epoch": np.arange(1, len(train_losses) + 1),
        "train_loss": train_losses,
        "val_loss": val_losses[:len(train_losses)],
        "lambda": lambda_history[:len(train_losses)]
    })
    history_csv = os.path.join(save_dir, f"history_{tag}.csv")
    history_df.to_csv(history_csv, index=False)

    print(f"✔ 训练记录已保存: {history_csv}")

    return metrics


# ========= 超参数搜索主程序 ==============
def hyperparameter_search():
    condition_name = "Normal"   # 先在 Normal 上调参

    # 搜索空间（你可以按需求修改）
    num_layers_list = [4, 6]
    num_neurons_list = [128, 256]
    lr_list = [1e-3, 5e-4]
    init_log_lambda_list = [0.0, 0.5]  # 0→λ=1, 0.5→λ≈1.65
    batch_size_list = [64,128,256]            # 可以加 [64, 128]
    max_epochs_list = [100, 200]             # 用 config 中的
    total_combinations = (
            len(num_layers_list) *
            len(num_neurons_list) *
            len(lr_list) *
            len(init_log_lambda_list) *
            len(batch_size_list)*
            len(max_epochs_list)
    )

    print(f"Total hyperparameter combinations: {total_combinations}")
    results = []
    first_run_time = None
    for idx, (num_layers, num_neurons, lr, init_log_lambda, batch_size, max_epochs) in enumerate(
            itertools.product(
                num_layers_list,
                num_neurons_list,
                lr_list,
                init_log_lambda_list,
                batch_size_list,
                max_epochs_list
            ),
            start=1
    ):
        print(f"\n>>> 开始第 {idx}/{total_combinations} 个组合的训练...")

        start_time = time.time()
        metrics = train_one_model(
            condition_name=condition_name,
            num_layers=num_layers,
            num_neurons=num_neurons,
            lr=lr,
            init_log_lambda=init_log_lambda,
            batch_size=batch_size,
            max_epochs=max_epochs,
            data_root="data",
            save_dir="hp_results"
        )
        elapsed = time.time() - start_time
        # 第一次训练结束后，估算总搜索时间
        if first_run_time is None:
            first_run_time = elapsed
            est_total_seconds = first_run_time * total_combinations

            est_total_minutes = est_total_seconds / 60
            est_total_hours = est_total_seconds / 3600

            print("\n⏱ 第一个组合训练耗时："
                  f"{elapsed:.1f} 秒（约 {elapsed / 60:.2f} 分钟）")
            print("📌 预估整个超参数搜索总耗时："
                  f"{est_total_seconds:.1f} 秒 ≈ {est_total_minutes:.2f} 分钟 ≈ {est_total_hours:.2f} 小时\n")

        else:
            print(f"⏱ 当前组合实际耗时：{elapsed:.1f} 秒（约 {elapsed / 60:.2f} 分钟）")

        row = {
            "condition": condition_name,
            "num_layers": num_layers,
            "num_neurons": num_neurons,
            "lr": lr,
            "init_log_lambda": init_log_lambda,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
        }
        row.update(metrics)
        results.append(row)

    # 汇总结果
    df = pd.DataFrame(results)
    os.makedirs("hp_results", exist_ok=True)
    result_csv = os.path.join("hp_results", "hyperparam_results_Normal.csv")
    df.to_csv(result_csv, index=False)
    print(f"\n✅ 超参数搜索结果已保存到: {result_csv}")

    # 找一个验证集 RMSE 最小的组合
    best_row = df.loc[df["rmse_val"].idxmin()]
    print("\n⭐ 最优超参数组合（按验证集 RMSE）:")
    print(best_row)


if __name__ == "__main__":
    hyperparameter_search()
