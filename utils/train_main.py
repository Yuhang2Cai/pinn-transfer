# train_main.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt

from config import (
    DEVICE, BATCH_SIZE, NUM_EPOCH, NUM_LAYERS, NUM_NEURONS,
    NUM_ROUNDS, LR, CSV_PATH, INPUT_DIM, OUTPUT_DIM,
    TARGET_COL, DROP_COLS
)
from utils import (
    TensorDataset, standardize_tensor, calculate_metrics_in_batches,
    calculate_r2_in_batches
)
from models import TriplexPINN
from losses import My_loss


def train(num_epoch, batch_size, train_loader, num_slices_train,
          inputs_val, targets_val, model, optimizer, scheduler, criterion):

    num_period = int(num_slices_train / batch_size)
    train_losses = []
    val_losses = []

    for epoch in range(num_epoch):
        model.train()
        epoch_train_loss = 0.0

        with torch.backends.cudnn.flags(enabled=False):
            for period, (inputs_train_batch, targets_train_batch) in enumerate(train_loader):
                p_pred, P_t_pred = model(inputs=inputs_train_batch)

                _ = dict()
                _['loss_train'] = torch.zeros(num_period)
                _['var_P'] = torch.zeros(num_period)
                _['loss_physics'] = torch.zeros(num_period)

                loss = criterion(
                    targets_P=targets_train_batch,
                    outputs_P=p_pred,
                    dpdt=P_t_pred,
                    mdot_A=inputs_train_batch[:, 2],
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

                if (epoch + 1) % 1 == 0 and (period + 1) % 10 == 0:
                    print(
                        f"Epoch: {epoch+1}, Period: {period+1}, "
                        f"Loss: {loss.item():.5f}, "
                        f"Loss_M: {criterion.loss_M.item():.5f}, "
                        f"Loss_physics: {criterion.loss_physics.item():.5f}"
                    )

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

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

        scheduler.step()

    return model, train_losses, val_losses


def main(weight):
    torch.manual_seed(42)
    np.random.seed(42)

    print("1. 加载和预处理数据...")
    data = pd.read_csv(CSV_PATH)

    X = data.drop(DROP_COLS, axis=1)
    Y = data[TARGET_COL]

    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)

    X_train = X.iloc[:train_size]
    y_train = Y.iloc[:train_size]
    X_val = X.iloc[train_size:train_size + val_size]
    y_val = Y.iloc[train_size:train_size + val_size]
    X_test = X.iloc[train_size + val_size:]
    y_test = Y.iloc[train_size + val_size:]

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float64)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float64).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float64)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float64).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float64)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float64).unsqueeze(1)

    inputs = {
        'train': X_train_tensor.to(DEVICE),
        'val':   X_val_tensor.to(DEVICE),
        'test':  X_test_tensor.to(DEVICE)
    }
    targets = {
        'train': y_train_tensor.to(DEVICE),
        'val':   y_val_tensor.to(DEVICE),
        'test':  y_test_tensor.to(DEVICE)
    }

    inputs_train = inputs['train']
    inputs_val   = inputs['val']
    inputs_test  = inputs['test']
    targets_train = targets['train']
    targets_val   = targets['val']
    targets_test  = targets['test']

    layers = NUM_LAYERS * [NUM_NEURONS]

    num = inputs_train.shape[0]
    _, mean_inputs_train, std_inputs_train = standardize_tensor(
        torch.reshape(inputs_train, (num, 1, INPUT_DIM)), mode='fit'
    )
    _, mean_targets_train, std_targets_train = standardize_tensor(
        targets_train, mode='fit'
    )

    train_set = TensorDataset(inputs_train, targets_train)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )

    metric_rounds = {
        'train': np.zeros(NUM_ROUNDS),
        'val':   np.zeros(NUM_ROUNDS),
        'test':  np.zeros(NUM_ROUNDS)
    }

    print(f"2. 开始 {NUM_ROUNDS} 轮训练和评估...")

    all_train_losses = []
    all_val_losses = []

    for round_id in range(NUM_ROUNDS):
        print(f"\n=== 第 {round_id+1}/{NUM_ROUNDS} 轮 ===")

        model = TriplexPINN(
            seq_len=1,
            inputs_dim=INPUT_DIM,
            outputs_dim=OUTPUT_DIM,
            layers=layers,
            scaler_inputs=(mean_inputs_train, std_inputs_train),
            scaler_targets=(mean_targets_train, std_targets_train)
        ).to(DEVICE)

        criterion = My_loss()
        params = [p for p in model.parameters()]
        optimizer = optim.Adam(params, lr=LR)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1)

        print("训练模型中...")
        model, train_losses, val_losses = train(
            num_epoch=NUM_EPOCH,
            batch_size=BATCH_SIZE,
            train_loader=train_loader,
            num_slices_train=inputs_train.shape[0],
            inputs_val=inputs_val,
            targets_val=targets_val,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion
        )

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        print("评估模型中...")
        model.eval()

        P_pred_train, _ = model(inputs=inputs_train)
        RMSE_train, MAE_train, MAPE_train = calculate_metrics_in_batches(P_pred_train, targets_train)
        print(f"RMSE_train:{RMSE_train}")
        print(f"MAE_train:{MAE_train}")
        print(f"MAPE_train:{MAPE_train}")

        P_pred_val, _ = model(inputs=inputs_val)
        RMSE_val, MAE_val, MAPE_val = calculate_metrics_in_batches(P_pred_val, targets_val)
        print(f"RMSE_val:{RMSE_val}")
        print(f"MAE_val:{MAE_val}")
        print(f"MAPE_val:{MAPE_val}")

        P_pred_test, _ = model(inputs=inputs_test)
        RMSE_test, MAE_test, MAPE_test = calculate_metrics_in_batches(P_pred_test, targets_test)
        print(f"RMSE_test:{RMSE_test}")
        print(f"MAE_test:{MAE_test}")
        print(f"MAPE_test:{MAPE_test}")

        R2_train = calculate_r2_in_batches(P_pred_train, targets_train)
        R2_val   = calculate_r2_in_batches(P_pred_val, targets_val)
        R2_test  = calculate_r2_in_batches(P_pred_test, targets_test)
        print(f"R-squared (train): {R2_train.item():.4f}")
        print(f"R-squared (val):   {R2_val.item():.4f}")
        print(f"R-squared (test):  {R2_test.item():.4f}")

        metric_rounds['train'][round_id] = RMSE_train.item()
        metric_rounds['val'][round_id]   = RMSE_val.item()
        metric_rounds['test'][round_id]  = RMSE_test.item()

    print("\n3. 计算多轮平均结果...")
    metric_mean = {
        'train': np.mean(metric_rounds['train']),
        'val':   np.mean(metric_rounds['val']),
        'test':  np.mean(metric_rounds['test'])
    }
    metric_std = {
        'train': np.std(metric_rounds['train']),
        'val':   np.std(metric_rounds['val']),
        'test':  np.std(metric_rounds['test'])
    }

    print(f"\n平均指标 ({NUM_ROUNDS} 轮):")
    print(f"训练集 RMSE: {metric_mean['train']:.4f} ± {metric_std['train']:.4f}")
    print(f"验证集 RMSE: {metric_mean['val']:.4f} ± {metric_std['val']:.4f}")
    print(f"测试集 RMSE: {metric_mean['test']:.4f} ± {metric_std['test']:.4f}")

    plt.figure(figsize=(12, 6))
    avg_train_losses = np.mean(all_train_losses, axis=0)
    avg_val_losses   = np.mean(all_val_losses, axis=0)
    plt.plot(range(1, NUM_EPOCH + 1), avg_train_losses, label='train loss')
    plt.plot(range(1, NUM_EPOCH + 1), avg_val_losses,   label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f'lossfunction(avg {NUM_ROUNDS} times)')
    plt.legend()
    plt.grid(True)

    batch_size = BATCH_SIZE
    epochs = NUM_EPOCH
    time_scale = getattr(model, "time_scale", 1.0)
    physics_weight = getattr(criterion, "physics_weight", 1.0)
    num_layers = len(model.surrogateNN.layers) // 3
    neurons_per_layer = NUM_NEURONS

    result_dir = (
        f"metrics_"
        f"bs{batch_size}_"
        f"ep{epochs}_"
        f"tscale{time_scale:.0e}_"
        f"pweight{physics_weight:.1f}_"
        f"layers{num_layers}_"
        f"neurons{neurons_per_layer}_"
        f"lr{LR:.1e}"
    )
    os.makedirs(result_dir, exist_ok=True)
    metrics_png_path = os.path.join(result_dir, f'{result_dir}.png')
    plt.savefig(metrics_png_path, dpi=300, bbox_inches='tight')

    print("\n4. 保存结果...")
    model.eval()
    P_pred_test, _ = model(inputs=inputs_test)
    results = {
        'P_true':  targets_test.detach().cpu().numpy().squeeze(),
        'P_pred':  P_pred_test.detach().cpu().numpy().squeeze(),
        'Cycles':  inputs_test[:, 0].detach().cpu().numpy().squeeze(),
        'Epochs':  np.arange(0, NUM_EPOCH)
    }
    metrics_pth_path = os.path.join(result_dir, f'{result_dir}.pth')
    torch.save(results, metrics_pth_path)

    print("\n5. 计算并保存评估指标...")
    P_pred_train, _ = model(inputs=inputs_train)
    rmse_train, mae_train, mape_train = calculate_metrics_in_batches(P_pred_train, targets_train)
    r2_train = calculate_r2_in_batches(P_pred_train, targets_train)

    P_pred_val, _ = model(inputs=inputs_val)
    rmse_val, mae_val, mape_val = calculate_metrics_in_batches(P_pred_val, targets_val)
    r2_val = calculate_r2_in_batches(P_pred_val, targets_val)

    rmse_test, mae_test, mape_test = calculate_metrics_in_batches(P_pred_test, targets_test)
    r2_test = calculate_r2_in_batches(P_pred_test, targets_test)

    metrics_data = {
        "train_set": {
            "RMSE": rmse_train.item(),
            "MAE":  mae_train.item(),
            "MAPE": mape_train.item(),
            "R2":   r2_train.item()
        },
        "val_set": {
            "RMSE": rmse_val.item(),
            "MAE":  mae_val.item(),
            "MAPE": mape_val.item(),
            "R2":   r2_val.item()
        },
        "test_set": {
            "RMSE": rmse_test.item(),
            "MAE":  mae_test.item(),
            "MAPE": mape_test.item(),
            "R2":   r2_test.item()
        }
    }

    metrics_txt_path = os.path.join(result_dir, "metrics.txt")
    with open(metrics_txt_path, "w") as f:
        for dataset, metrics in metrics_data.items():
            f.write(f"{dataset} metrics:\n")
            for metric_name, value in metrics.items():
                f.write(f"  {metric_name}: {value:.6f}\n")
            f.write("\n")

    metrics_df = pd.DataFrame({
        "dataset": ["train_set", "val_set", "test_set"],
        "RMSE": [metrics_data["train_set"]["RMSE"],
                 metrics_data["val_set"]["RMSE"],
                 metrics_data["test_set"]["RMSE"]],
        "MAE":  [metrics_data["train_set"]["MAE"],
                 metrics_data["val_set"]["MAE"],
                 metrics_data["test_set"]["MAE"]],
        "MAPE": [metrics_data["train_set"]["MAPE"],
                 metrics_data["val_set"]["MAPE"],
                 metrics_data["test_set"]["MAPE"]],
        "R2":   [metrics_data["train_set"]["R2"],
                 metrics_data["val_set"]["R2"],
                 metrics_data["test_set"]["R2"]]
    })
    metrics_csv_path = os.path.join(result_dir, f'{result_dir}.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)

    print(f"Evaluation metrics saved to {result_dir} folder")
    print(f"结果已保存到 {metrics_pth_path}")
    return model, results


if __name__ == "__main__":
    weightList = [1.9, 2.2]
    for w in weightList:
        main(w)
