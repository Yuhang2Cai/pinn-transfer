# models.py
import torch
from torch import nn
from utils import standardize_tensor, inverse_standardize_tensor
from config import TIME_SCALE

class Sin(nn.Module):
    def forward(self, input):
        return torch.sin(input)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Neural_Net(nn.Module):
    def __init__(self, seq_len, inputs_dim, outputs_dim, layers, activation='Sin'):
        super(Neural_Net, self).__init__()
        self.seq_len = seq_len
        self.inputs_dim = inputs_dim
        self.outputs_dim = outputs_dim

        self.layers = nn.ModuleList()

        # 第一层
        layer = nn.Linear(inputs_dim, layers[0]).double()
        nn.init.xavier_normal_(layer.weight)
        self.layers.append(layer)

        if activation == 'Tanh':
            self.layers.append(nn.Tanh())
        elif activation == 'Sin':
            self.layers.append(Swish())
        self.layers.append(nn.Dropout(p=0.2))

        # 中间层
        for l in range(len(layers) - 1):
            layer = nn.Linear(layers[l], layers[l + 1]).double()
            nn.init.xavier_normal_(layer.weight)
            self.layers.append(layer)

            if activation == 'Tanh':
                self.layers.append(nn.Tanh())
            elif activation == 'Sin':
                self.layers.append(Swish())
            self.layers.append(nn.Dropout(p=0.2))

        # 输出层
        layer = nn.Linear(layers[-1], outputs_dim).double()
        nn.init.xavier_normal_(layer.weight)
        self.layers.append(layer)

        self.NN = nn.Sequential(*self.layers)

    def forward(self, x):
        x = x.double() if x.dtype != torch.float64 else x
        self.x = x.contiguous().view(-1, self.inputs_dim)
        NN_out_2D = self.NN(self.x)
        self.p_pred = NN_out_2D.view(-1, self.seq_len, self.outputs_dim)
        return self.p_pred


class TriplexPINN(nn.Module):
    def __init__(self, seq_len, inputs_dim, outputs_dim, layers, scaler_inputs, scaler_targets):
        super(TriplexPINN, self).__init__()
        self.seq_len, self.inputs_dim, self.outputs_dim = seq_len, inputs_dim, outputs_dim
        self.scaler_inputs, self.scaler_targets = scaler_inputs, scaler_targets
        self.time_scale = TIME_SCALE

        self.surrogateNN = Neural_Net(
            seq_len=self.seq_len,
            inputs_dim=self.inputs_dim,
            outputs_dim=self.outputs_dim,
            layers=layers
        )

    def forward(self, inputs):
        s = inputs[:, 1:]        # 非时间变量
        t = inputs[:, 0:1]       # 时间变量

        # 标准化非时间变量（注意跳过时间维）
        s_norm, _, _ = standardize_tensor(
            s, mode='transform',
            mean=self.scaler_inputs[0][1:],
            std=self.scaler_inputs[1][1:]
        )

        # 时间缩放
        t_scaled = t / self.time_scale
        t_scaled = t_scaled.unsqueeze(0)
        t_scaled.requires_grad_(True)

        # 拼接时间和其他特征
        P_norm = self.surrogateNN(x=torch.cat((s_norm, t_scaled), dim=2))
        P = inverse_standardize_tensor(P_norm, mean=self.scaler_targets[0], std=self.scaler_targets[1])

        # 自动微分求时间导数
        grad_outputs = torch.ones_like(P)
        P_t = torch.autograd.grad(
            P, t_scaled,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0] * (1.0 / self.time_scale)

        return P, P_t[0]
