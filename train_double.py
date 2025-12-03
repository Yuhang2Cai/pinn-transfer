# train_tl.py  /  train_double.py
import os
import time
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    DEVICE, BATCH_SIZE, NUM_EPOCH, NUM_LAYERS, NUM_NEURONS,
    LR, INPUT_DIM, OUTPUT_DIM, TIME_SCALE
)
from utils import (
    TensorDataset, standardize_tensor,
    calculate_metrics_in_batches, calculate_r2_in_batches,
    load_condition_split_csv
)
from models import TriplexPINN
from losses import My_loss


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 保证一些算子可复现（会有一定性能损失）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def train_one_stage(condition_name,
                    prev_state_dict=None,
                    physics_weight=2.0,
                    data_root="data",
                    record_epoch_r2=False):
    """
    训练某一个工况（Normal / Leak / Block / Worn）。
    - 如果 prev_state_dict 不为 None，就从给定参数初始化（用于 Normal -> Fault 的迁移）。
    - 如果 record_epoch_r2=True，则在每个 epoch 记录当前模型在 test 集上的 R2，
      最后自动保存 CSV 和 曲线图。
    - 不使用早停，固定训练 NUM_EPOCH 轮，但会保存 val_loss 最优的 checkpoint。
    """
    prev_state_dict = None
    print(f"\n============================")
    print(f"开始训练工况: {condition_name}")
    print(f"是否迁移初始化: {'是' if prev_state_dict is not None else '否'}")
    print(f"============================\n")

    # 1. 加载该工况的 train/val/test
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

    # 2. 标准化（只用本工况的 train 去 fit）
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
        batch_size=BATCH_SIZE,
        shuffle=False,  # 和你原来的设置保持一致
        num_workers=0,
        drop_last=True
    )

    # 4. 初始化模型
    layers = [NUM_NEURONS] * NUM_LAYERS
    model = TriplexPINN(
        seq_len=1,
        inputs_dim= INPUT_DIM,
        outputs_dim=OUTPUT_DIM,
        layers=layers,
        scaler_inputs=(mean_inputs_train, std_inputs_train),
        scaler_targets=(mean_targets_train, std_targets_train)
    ).to(DEVICE)

    # 如果有前一阶段参数，则迁移（例如 Normal -> Leak）
    if prev_state_dict is not None:
        model.load_state_dict(prev_state_dict)

    # 5. 损失函数 & 优化器
    criterion = My_loss()
    if condition_name == "Leak":
        lr = LR * 0.5  # 你原来的特例
    else:
        lr = LR

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1)

    train_losses = []
    val_losses = []
    lambda_history = []      # 记录每个 epoch 的物理权重 λ
    r2_epoch_history = []    # ⭐ 记录每个 epoch 的 test R2

    # checkpoint 路径（保存 val_loss 最优模型）
    best_path = f"tl_results/best_{condition_name}.pth"
    os.makedirs("tl_results", exist_ok=True)
    best_val_loss = float('inf')
    delta = 1e-5  # 和原来 EarlyStopping 里的 delta 类似

    # ⭐ 如果需要记录 R2，先算一次 epoch=0 的 test R2（迁移前 / 训练前）
    if record_epoch_r2:
        model.eval()
        # 关键：需要梯度来算 P_t，所以不能用 no_grad，要给输入开 requires_grad
        inputs_test.requires_grad_(True)
        P_pred_test0, _ = model(inputs=inputs_test)
        r2_0 = calculate_r2_in_batches(P_pred_test0, targets_test).item()
        r2_epoch_history.append(r2_0)
        print(f"[{condition_name}] Epoch 0 (before fine-tune), test R2={r2_0:.4f}")
    else:
        # 确保 test 不影响后面训练
        inputs_test.requires_grad_(False)

    # 7. 训练循环（不早停，跑满 NUM_EPOCH）
    for epoch in range(NUM_EPOCH):
        model.train()
        epoch_train_loss = 0.0

        with torch.backends.cudnn.flags(enabled=False):
            for period, (batch_x, batch_y) in enumerate(train_loader):
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
        cur_val = val_loss.item()
        val_losses.append(cur_val)
        lambda_history.append(criterion.physics_weight.item())

        # ⭐ 如果当前 val_loss 更好，则更新 best checkpoint
        if cur_val < best_val_loss - delta:
            best_val_loss = cur_val
            torch.save(model.state_dict(), best_path)

        # ⭐ 每个 epoch 计算一次 test R2
        if record_epoch_r2:
            model.eval()
            # 已经在前面设置过 requires_grad_(True)，持续有效
            P_pred_test, _ = model(inputs=inputs_test)
            r2_e = calculate_r2_in_batches(P_pred_test, targets_test).item()
            r2_epoch_history.append(r2_e)

        log_msg = (f"[{condition_name}] Epoch {epoch + 1}/{NUM_EPOCH}, "
                   f"train_loss={avg_train_loss:.5f}, val_loss={cur_val:.5f}, "
                   f"lambda={criterion.physics_weight.item():.4f}")
        if record_epoch_r2:
            log_msg += f", test_R2={r2_epoch_history[-1]:.4f}"
        print(log_msg)

        scheduler.step()

    # 8. 加载该工况的最优模型参数（按 val_loss 最小）
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        print(f"✔ 使用 {condition_name} 的最佳模型参数进行评估 (val_loss={best_val_loss:.5f})")
    else:
        print(f"⚠ 未找到 {best_path}，使用最后一轮模型评估")

    # ⭐ 保存每 epoch 的 test R2 曲线（只对 record_epoch_r2=True 的情况）
    if record_epoch_r2 and len(r2_epoch_history) > 0:
        r2_df = pd.DataFrame({
            "epoch": np.arange(len(r2_epoch_history)),  # 从 0 开始：0=初始化
            "R2_test": r2_epoch_history
        })
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        r2_csv_path = os.path.join("tl_results", f"r2_epoch_{condition_name}_{timestamp}.csv")
        r2_png_path = os.path.join("tl_results", f"r2_epoch_{condition_name}_{timestamp}.png")
        r2_df.to_csv(r2_csv_path, index=False)
        print(f"✔ {condition_name} 的 R2-epoch 曲线数据已保存到: {r2_csv_path}")

        # 画 R2 曲线图
        plt.figure(figsize=(8, 4))
        plt.plot(r2_df["epoch"], r2_df["R2_test"], linewidth=1.5)
        plt.xlabel("Fine-tuning epoch")
        plt.ylabel("Test R2")
        plt.title(f"Test R2 vs Epoch (Normal → {condition_name})")
        plt.grid(True)
        plt.savefig(r2_png_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✔ {condition_name} 的 R2 曲线图已保存到: {r2_png_path}")

    # 9. 用“最佳模型”在 train/val/test 上分别算最终指标
    model.eval()
    inputs_train.requires_grad_(True)
    inputs_val.requires_grad_(True)
    inputs_test.requires_grad_(True)

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
        "train": {"RMSE": rmse_train.item(), "MAE": mae_train.item(),
                  "MAPE": mape_train.item(), "R2": r2_train.item()},
        "val":   {"RMSE": rmse_val.item(),   "MAE": mae_val.item(),
                  "MAPE": mape_val.item(),   "R2": r2_val.item()},
        "test":  {"RMSE": rmse_test.item(),  "MAE": mae_test.item(),
                  "MAPE": mape_test.item(),  "R2": r2_test.item()},
    }

    print(f"\n[{condition_name}] 最终指标：")
    for split in ["train", "val", "test"]:
        m = metrics[split]
        print(f"  {split}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, "
              f"MAPE={m['MAPE']:.2f}, R2={m['R2']:.4f}")

    # 返回当前模型的参数（用于后续迁移）和指标
    return model.state_dict(), metrics, (train_losses, val_losses)


def main_tl():
    """
    先训练 Normal 工况（无迁移），得到 base 模型参数；
    然后分别执行多次迁移：
        Normal -> Leak
        Normal -> Block
        Normal -> Worn

    对每个故障工况，在训练过程中记录每个 epoch 的 test R2，
    并自动保存 r2_epoch_xxx.csv 和 r2_epoch_xxx.png。
    """
    physics_weight = 2.0
    os.makedirs("tl_results", exist_ok=True)

    all_stage_metrics = {}
    all_stage_losses = {}

    start_time = time.time()

    # 1) 先训练 Normal（只训一次，作为所有迁移的源模型）
    print("\n================ 训练 Normal（基模型） ================")
    normal_state_dict, normal_metrics, normal_losses = train_one_stage(
        condition_name="Normal",
        prev_state_dict=None,
        physics_weight=physics_weight,
        data_root="data",
        record_epoch_r2=False  # Normal 不记录 epoch R2
    )
    all_stage_metrics["Normal"] = normal_metrics
    all_stage_losses["Normal"] = normal_losses

    # 2) 对每个故障工况：从 Normal 的参数出发进行迁移学习
    # 如果暂时没有 Block 数据，可以改成 ["Leak", "Worn"]
    target_conditions = ["Leak", "Worn","Block"]  # 没有 Block 的话就这样写

    for cond in target_conditions:
        print(f"\n================ Normal → {cond} 迁移训练 ================")
        # 每个故障工况都从同一个 normal_state_dict 初始化
        state_dict, metrics, losses = train_one_stage(
            condition_name=cond,
            prev_state_dict=normal_state_dict,
            physics_weight=physics_weight,
            data_root="data",
            record_epoch_r2=True   # ⭐ 故障工况：记录 epoch 的 test R2
        )
        all_stage_metrics[cond] = metrics
        all_stage_losses[cond] = losses

    end_time = time.time()
    total_training_time = end_time - start_time

    # 3) 把所有阶段的最终指标存到 CSV
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    rows = []
    for cond, ms in all_stage_metrics.items():
        for split in ["train", "val", "test"]:
            rows.append({
                "condition": cond,
                "split": split,
                "RMSE": ms[split]["RMSE"],
                "MAE": ms[split]["MAE"],
                "MAPE": ms[split]["MAPE"],
                "R2": ms[split]["R2"],
            })
    df = pd.DataFrame(rows)
    metrics_filename = f"tl_stage_metrics_{timestamp}.csv"
    metrics_path = os.path.join("tl_results", metrics_filename)
    df.to_csv(metrics_path, index=False)

    training_info = {
        "timestamp": timestamp,
        "total_training_time_seconds": total_training_time,
        "conditions_trained": ", ".join(["Normal"] + target_conditions)
    }
    training_info_df = pd.DataFrame([training_info])
    training_info_filename = f"training_time_{timestamp}.csv"
    training_info_path = os.path.join("tl_results", training_info_filename)
    training_info_df.to_csv(training_info_path, index=False)

    print(f"\nTotal training time: {total_training_time:.2f} seconds")
    print(f"TL-PINN 各阶段最终指标已保存到 {metrics_path}")
    print(f"Training time info saved to {training_info_path}")


if __name__ == "__main__":
    main_tl()
