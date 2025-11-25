# train_tl.py
import os
import time

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
    load_condition_split_csv, EarlyStopping
)
from models import TriplexPINN
from losses import My_loss


def train_one_stage(condition_name,
                    prev_state_dict=None,
                    physics_weight=2.0,
                    data_root="data"):
    """
    训练某一个工况（Normal / Leak / Block / Worn），
    如果 prev_state_dict 不为 None，就从前一阶段参数继续训练（迁移学习）。
    """

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
        shuffle=False,
        num_workers=0,
        drop_last=True
    )

    # 4. 初始化模型
    layers = [NUM_NEURONS] * NUM_LAYERS
    model = TriplexPINN(
        seq_len=1,
        inputs_dim=INPUT_DIM,
        outputs_dim=OUTPUT_DIM,
        layers=layers,
        scaler_inputs=(mean_inputs_train, std_inputs_train),
        scaler_targets=(mean_targets_train, std_targets_train)
    ).to(DEVICE)

    # 如果有前一阶段参数，则迁移
    if prev_state_dict is not None:
        model.load_state_dict(prev_state_dict)

    # 5. 损失函数 & 优化器
    criterion = My_loss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=LR
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1)

    train_losses = []
    val_losses = []
    lambda_history = []  # 🔥 记录每个 epoch 的物理权重 λ
    # ⭐⭐ 6. 创建 EarlyStopping（每个工况一个独立 best_xxx.pth） ⭐⭐
    best_path = f"tl_results/best_{condition_name}.pth"
    os.makedirs("tl_results", exist_ok=True)
    early_stopping = EarlyStopping(
        patience=15,
        delta=1e-5,
        save_path=best_path
    )
    # 6. 训练循环（和你原来的 train() 几乎一样）
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
        val_losses.append(val_loss.item())
        # 记录当前 epoch 的 λ
        lambda_history.append(criterion.physics_weight.item())
        # 🔥 调用早停
        early_stopping(val_loss.item(), model)

        print(f"[{condition_name}] Epoch {epoch + 1}/{NUM_EPOCH}, "
              f"train_loss={avg_train_loss:.5f}, val_loss={val_loss.item():.5f}, "
              f"lambda={criterion.physics_weight.item():.4f}")

        if early_stopping.early_stop:
            print(f"🛑 {condition_name} 提前停止在 epoch {epoch + 1}")
            break
        scheduler.step()
        #
        # print(f"[{condition_name}] Epoch {epoch+1}/{NUM_EPOCH}, "
        #       f"train_loss={avg_train_loss:.5f}, val_loss={val_loss.item():.5f}")
    # ⭐ 在计算指标前，先加载该工况的最优模型参数
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        print(f"✔ 使用 {condition_name} 的最佳模型参数进行评估")
    else:
        print(f"⚠ 未找到 {best_path}，使用最后一轮模型评估")
        # 🔥 保存该工况训练过程中 λ 的变化曲线（csv + png）
        if len(lambda_history) > 0:
            lambda_df = pd.DataFrame({
                "epoch": np.arange(1, len(lambda_history) + 1),
                "lambda": lambda_history
            })
            lambda_csv_path = os.path.join("tl_results", f"lambda_{condition_name}.csv")
            lambda_df.to_csv(lambda_csv_path, index=False)
            print(f"✔ {condition_name} 的 λ 变化已保存到: {lambda_csv_path}")

            # 画 λ 曲线图
            plt.figure(figsize=(8, 4))
            plt.plot(lambda_df["epoch"], lambda_df["lambda"], marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Lambda (physics_weight)")
            plt.title(f"Lambda evolution - {condition_name}")
            plt.grid(True)

            lambda_png_path = os.path.join("tl_results", f"lambda_{condition_name}.png")
            plt.savefig(lambda_png_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"✔ {condition_name} 的 λ 曲线图已保存到: {lambda_png_path}")
    # 7. 在 train/val/test 上分别算指标
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
        "train": {"RMSE": rmse_train.item(), "MAE": mae_train.item(),
                  "MAPE": mape_train.item(), "R2": r2_train.item()},
        "val":   {"RMSE": rmse_val.item(),   "MAE": mae_val.item(),
                  "MAPE": mape_val.item(),   "R2": r2_val.item()},
        "test":  {"RMSE": rmse_test.item(),  "MAE": mae_test.item(),
                  "MAPE": mape_test.item(),  "R2": r2_test.item()},
    }

    print(f"\n[{condition_name}] 指标：")
    for split in ["train", "val", "test"]:
        m = metrics[split]
        print(f"  {split}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, "
              f"MAPE={m['MAPE']:.2f}, R2={m['R2']:.4f}")

    # 返回当前模型的参数（用于迁移）和指标
    return model.state_dict(), metrics, (train_losses, val_losses)


def main_tl():
    # 迁移顺序：Normal -> Leak -> Block -> Worn
    condition_order = ["Normal", "Leak", "Worn"]#, "Block"
    physics_weight = 2.0  # 你原来的 My_loss(weight)

    prev_state_dict = None
    all_stage_metrics = {}
    all_stage_losses = {}
    start_time = time.time()
    for idx, cond in enumerate(condition_order):
        # 第一个工况 prev_state_dict=None -> 随机初始化
        # 后面的工况 prev_state_dict!=None -> 迁移学习
        prev_state_dict, metrics, losses = train_one_stage(
            condition_name=cond,
            prev_state_dict=prev_state_dict,
            physics_weight=physics_weight,
            data_root="data"
        )

        all_stage_metrics[cond] = metrics
        all_stage_losses[cond] = losses  # (train_losses, val_losses)
        # Record end time and calculate total duration
    end_time = time.time()
    total_training_time = end_time - start_time
    # 可以把 all_stage_metrics 存成一个 json / csv，方便论文画图/做表
    os.makedirs("tl_results", exist_ok=True)
    # Generate timestamp for unique filenames
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
    # Include timestamp and training time in the filename
    metrics_filename = f"tl_stage_metrics_{timestamp}.csv"
    metrics_path = os.path.join("tl_results", metrics_filename)
    df.to_csv(metrics_path, index=False)

    # Save training time to a separate file
    training_info = {
        "timestamp": timestamp,
        "total_training_time_seconds": total_training_time,
        "conditions_trained": ", ".join(condition_order)
    }
    training_info_df = pd.DataFrame([training_info])
    training_info_filename = f"training_time_{timestamp}.csv"
    training_info_path = os.path.join("tl_results", training_info_filename)
    training_info_df.to_csv(training_info_path, index=False)

    print(f"\nTotal training time: {total_training_time:.2f} seconds")
    print(f"TL-PINN 各阶段指标已保存到 {metrics_path}")
    print(f"Training time info saved to {training_info_path}")

    # === 新增：用“最终模型”统一评估四个工况的 test 集，并单独保存 =================
    print("\n================ 使用最终模型评估四个工况的测试集 ================")
    final_test_metrics = evaluate_final_model_on_all_tests(
        final_state_dict=prev_state_dict,  # 最后一个阶段返回的 best state_dict
        condition_order=condition_order,
        data_root="data"
    )

    rows_final = []
    for cond, m in final_test_metrics.items():
        rows_final.append({
            "condition": cond,
            "split": "test_final_model",
            "RMSE": m["RMSE"],
            "MAE": m["MAE"],
            "MAPE": m["MAPE"],
            "R2": m["R2"],
        })
    df_final = pd.DataFrame(rows_final)
    final_filename = f"tl_final_model_test_metrics_{timestamp}.csv"
    final_path = os.path.join("tl_results", final_filename)
    df_final.to_csv(final_path, index=False)

    print(f"最终模型在四个测试集上的指标已保存到 {final_path}")
    # ==================================================================
# === 新增：用“最终模型权重”统一评估四个工况的 test 集 ==================
def evaluate_final_model_on_all_tests(final_state_dict,
                                      condition_order,
                                      data_root="data"):
    """
    使用最后阶段得到的最终模型参数（final_state_dict），
    对每一个工况的 test 集进行统一指标评估。
    """
    layers = [NUM_NEURONS] * NUM_LAYERS
    results = {}

    for cond in condition_order:
        print(f"\n[Final model] 开始评估工况 {cond} 的 test 集")

        # 1) 读取该工况的 train / test，用 train 算标准化（和 train_one_stage 保持一致）
        X_train, y_train = load_condition_split_csv(
            root_dir=data_root, split="train", condition=cond, device=DEVICE
        )
        X_test, y_test = load_condition_split_csv(
            root_dir=data_root, split="test", condition=cond, device=DEVICE
        )

        num_train = X_train.shape[0]
        _, mean_inputs_train, std_inputs_train = standardize_tensor(
            torch.reshape(X_train, (num_train, 1, INPUT_DIM)), mode='fit'
        )
        _, mean_targets_train, std_targets_train = standardize_tensor(
            y_train, mode='fit'
        )

        # 2) 构建模型并加载“最终模型”的权重
        model = TriplexPINN(
            seq_len=1,
            inputs_dim=INPUT_DIM,
            outputs_dim=OUTPUT_DIM,
            layers=layers,
            scaler_inputs=(mean_inputs_train, std_inputs_train),
            scaler_targets=(mean_targets_train, std_targets_train)
        ).to(DEVICE)
        model.load_state_dict(final_state_dict)
        model.eval()


        P_pred_test, _ = model(inputs=X_test)

        rmse_test, mae_test, mape_test = calculate_metrics_in_batches(
            P_pred_test, y_test
        )
        r2_test = calculate_r2_in_batches(P_pred_test, y_test)

        print(f"[Final model | {cond} - test] "
              f"RMSE={rmse_test:.4f}, MAE={mae_test:.4f}, "
              f"MAPE={mape_test:.2f}, R2={r2_test:.4f}")

        results[cond] = {
            "RMSE": rmse_test.item(),
            "MAE": mae_test.item(),
            "MAPE": mape_test.item(),
            "R2": r2_test.item()
        }

    return results
if __name__ == "__main__":
    main_tl()
