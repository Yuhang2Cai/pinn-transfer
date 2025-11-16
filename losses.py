# losses.py
import random

import torch
from torch import nn

class My_loss(nn.Module):
    def __init__(self, init_log_lambda=0.0):
        super().__init__()
        self.log_lambda = nn.Parameter(torch.tensor(init_log_lambda, dtype=torch.float64))


        self.loss_M = torch.tensor(0.0, requires_grad=False)
        self.loss_physics = torch.tensor(0.0, requires_grad=False)

    @property
    def physics_weight(self):
        return torch.exp(self.log_lambda)
    def mixture_density_derivative(self, p, bulk_modulus_model, air_dissolution_model,
                                   rho_L_atm, beta_L_atm, beta_gain, air_fraction,
                                   rho_g_atm, polytropic_index, p_atm, p_crit, p_min):

        rho_L_atm = torch.tensor(rho_L_atm, dtype=p.dtype, device=p.device)
        beta_L_atm = torch.tensor(beta_L_atm, dtype=p.dtype, device=p.device)
        beta_gain = torch.tensor(beta_gain, dtype=p.dtype, device=p.device)
        air_fraction = torch.tensor(air_fraction, dtype=p.dtype, device=p.device)
        rho_g_atm = torch.tensor(rho_g_atm, dtype=p.dtype, device=p.device)
        polytropic_index = torch.tensor(polytropic_index, dtype=p.dtype, device=p.device)
        p_atm = torch.tensor(p_atm, dtype=p.dtype, device=p.device)
        p_crit = torch.tensor(p_crit, dtype=p.dtype, device=p.device)
        p_min = torch.tensor(p_min, dtype=p.dtype, device=p.device)

        p_used = p if p >= p_min else p_min

        if air_dissolution_model == 'off':
            theta = 1.0
            dtheta_dp = 0.0
        else:
            if p_used <= p_atm:
                theta = 1.0
            elif p_used >= p_crit:
                theta = 0.0
            else:
                L = p_crit - p_atm
                x = (p_used - p_atm) / L
                theta = 1 - 3 * x ** 2 + 2 * x ** 3

            if p_used <= p_atm or p_used >= p_crit:
                dtheta_dp = 0.0
            else:
                L = p_crit - p_atm
                dtheta_dp = 6 * (p_used - p_atm) * (p_used - p_crit) / (L ** 3)

        if air_fraction == 0:
            p_denom = 0.0
        else:
            p_denom = (air_fraction / (1 - air_fraction)) * (p_atm / p_used) ** (1 / polytropic_index) * theta

        if air_fraction == 0:
            p_ratio = 0.0
        else:
            if air_dissolution_model == 'off':
                p_ratio = p_denom / (p_used * polytropic_index)
            else:
                term1 = (air_fraction / (1 - air_fraction)) * (p_atm / p_used) ** (1 / polytropic_index)
                term2 = (theta / (p_used * polytropic_index)) - dtheta_dp
                p_ratio = term1 * term2

        if air_fraction == 0:
            if bulk_modulus_model == 'const':
                exp_term = torch.exp((p_used - p_atm) / beta_L_atm) / beta_L_atm
            else:
                base = 1 + beta_gain * (p_used - p_atm) / beta_L_atm
                exponent = (-1 + 1 / beta_gain)
                exp_term = (base ** exponent) / beta_L_atm
        else:
            if bulk_modulus_model == 'const':
                with torch.no_grad():
                    exp_term = torch.exp(-(p_used - p_atm) / beta_L_atm) / beta_L_atm
            else:
                base = 1 + beta_gain * (p_used - p_atm) / beta_L_atm
                exponent = (-1 - 1 / beta_gain)
                exp_term = (base ** exponent) / beta_L_atm

        rho_mix_init = rho_L_atm + rho_g_atm * (air_fraction / (1 - air_fraction))

        if air_fraction == 0:
            drho_mix_dp = rho_L_atm * exp_term
        else:
            if bulk_modulus_model == 'const':
                denominator = beta_L_atm * exp_term + p_denom
            else:
                beta_term = 1 + beta_gain * (p_used - p_atm) / beta_L_atm
                denominator = beta_L_atm * exp_term * beta_term + p_denom
            numerator = exp_term + p_ratio
            drho_mix_dp = rho_mix_init * numerator / (denominator ** 2)

        return drho_mix_dp

    def forward(self, targets_P, dpdt, mdot_A, V, outputs_P, bulk_modulus_model, air_dissolution_model,
                rho_L_atm, beta_L_atm, beta_gain, air_fraction,
                rho_g_atm, polytropic_index, p_atm, p_crit, p_min):

        num = targets_P.shape[0]
        batch_loss_M = torch.tensor(0.0, requires_grad=True)
        batch_loss_physics = torch.tensor(0.0, requires_grad=True)

        for i in range(num):
            if abs(dpdt[i][0]) < 1e-12:
                print("梯度异常小")
                continue

            physics_loss = V * self.mixture_density_derivative(
                outputs_P[i] * 0.1, bulk_modulus_model,
                air_dissolution_model, rho_L_atm, beta_L_atm, beta_gain,
                air_fraction, rho_g_atm, polytropic_index, p_atm, p_crit,
                p_min
            ) * dpdt[i][0] - mdot_A[i]

            mae_loss = torch.abs(outputs_P[i] - targets_P[i])

            batch_loss_physics = batch_loss_physics + torch.abs(physics_loss)
            batch_loss_M = batch_loss_M + mae_loss

        self.loss_physics = batch_loss_physics / num
        num = random.Random().randint(1, 10)
        # if num % 5 == 0:
        #     print("loss_M: {:.5f}, loss_physics: {:.5f}".format(self.loss_M.item(), self.loss_physics.item()))
        # self.loss_M = batch_loss_M / num

        total_loss = self.loss_M + self.physics_weight * self.loss_physics
        return total_loss
