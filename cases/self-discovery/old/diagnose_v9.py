#!/usr/bin/env python3
"""诊断V9: 快速检查各α下网络在ξ=1的预测值"""
import sys
sys.path.append('/share/project/zpy/PINN_WE/cases/self-discovery/src')

import torch
import torch.optim as optim
from model import GuderleyPINN
from physics import (compute_total_loss, rankine_hugoniot_bc,
                     guderley_ode_residual)
from scan_alpha_v9 import create_dense_sampling

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gamma, n, mach = 1.4, 3, 10.0

V_bc, C_bc, _ = rankine_hugoniot_bc(gamma, mach)
print(f"RH理论值: V_bc={V_bc:.6f}, C_bc={C_bc:.6f}\n")

xi_int = create_dense_sampling(300, device)
xi_bd = torch.tensor([[1.0]]).to(device)
xi_ctr = torch.tensor([[1e-3]]).to(device)

alphas = [0.60, 0.68, 0.717, 0.75, 0.80, 0.90, 1.00]

print(f"{'alpha':<7} {'V(1)':<10} {'C(1)':<10} "
      f"{'V_err':<10} {'C_err':<10} {'BC_loss':<10}")
print("-" * 57)

for a in alphas:
    model = GuderleyPINN(alpha_init=a).to(device)
    model.train()
    model.raw_alpha.requires_grad = False

    opt = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4)

    # Adam 8000 epochs
    for ep in range(1, 8001):
        opt.zero_grad()
        tl, ld = compute_total_loss(
            model, xi_int, xi_bd, xi_ctr,
            gamma, n, mach,
            weight_pde=1.0,
            weight_bc_shock=10.0,
            weight_bc_center=5.0)
        tl.backward()
        opt.step()
