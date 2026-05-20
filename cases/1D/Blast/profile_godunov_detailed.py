#!/usr/bin/env python3
"""
详细性能分析：分解 loss_godunov_euler 的每个步骤
"""
import sys
sys.path.insert(0, '/share/project/zpy/PINN_WE/PINNsrc')

import torch
import time
from riemann_solver import EulerSolver

device = torch.device('cuda:0')
dtype = torch.float64
n_iterations = 100

# 模拟训练数据
n_points = 10000
x_int = torch.rand(n_points, 2, dtype=dtype, device=device)
x_int[:, 0] = 0.0  # t=0
x_int[:, 1] = torch.rand(n_points, dtype=dtype, device=device)  # x in [0,1]

# 模拟网络输出
rho = torch.rand(n_points, 1, dtype=dtype, device=device) * 0.5 + 0.5
p = torch.rand(n_points, 1, dtype=dtype, device=device) * 0.5 + 0.5
u = torch.rand(n_points, 1, dtype=dtype, device=device) * 0.2 - 0.1

gamma = 1.4

print("=" * 80)
print("详细性能分析：loss_godunov_euler 的每个步骤")
print("=" * 80)
print(f"数据规模: {n_points} 个点")
print(f"迭代次数: {n_iterations}")
print()

def profile_step(name, func):
    """性能分析辅助函数"""
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iterations):
        result = func()
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / n_iterations * 1000  # ms
    print(f"{name:<40} {elapsed:>10.4f} ms")
    return result

# 步骤1：计算通量
def step1_compute_flux():
    E = p / (gamma - 1.0) + 0.5 * rho * u**2
    F1 = rho * u
    F2 = rho * u**2 + p
    F3 = u * (E + p)
    return F1, F2, F3

F1, F2, F3 = profile_step("1. 计算通量 (F1, F2, F3)", step1_compute_flux)

# 步骤2：排序
def step2_sort():
    sort_idx = torch.argsort(x_int[:, 1])
    return sort_idx

sort_idx = profile_step("2. 排序 (torch.argsort)", step2_sort)

# 步骤3：重排数据
def step3_reorder():
    rho_sorted = rho[sort_idx]
    p_sorted = p[sort_idx]
    u_sorted = u[sort_idx]
    F1_sorted = F1[sort_idx]
    F2_sorted = F2[sort_idx]
    F3_sorted = F3[sort_idx]
    return rho_sorted, p_sorted, u_sorted, F1_sorted, F2_sorted, F3_sorted

rho_sorted, p_sorted, u_sorted, F1_sorted, F2_sorted, F3_sorted = profile_step(
    "3. 重排数据 (6个变量)", step3_reorder
)

# 步骤4：构造相邻点对
def step4_construct_pairs():
    rho_left = rho_sorted[:-1].flatten()
    rho_right = rho_sorted[1:].flatten()
    p_left = p_sorted[:-1].flatten()
    p_right = p_sorted[1:].flatten()
    u_left = u_sorted[:-1].flatten()
    u_right = u_sorted[1:].flatten()
    return rho_left, rho_right, p_left, p_right, u_left, u_right

rho_left, rho_right, p_left, p_right, u_left, u_right = profile_step(
    "4. 构造相邻点对 (12个切片)", step4_construct_pairs
)

# 步骤5：激波检测
def step5_shock_detection():
    delta_u = torch.abs(u_left - u_right)
    delta_p = torch.abs(p_left - p_right)
    threshold = 0.2
    eta = torch.clamp(delta_u - threshold, min=0) * torch.clamp(delta_p - threshold, min=0)
    return eta

eta = profile_step("5. 激波检测 (delta_u, delta_p, eta)", step5_shock_detection)

print()
print("=" * 80)
print("Godunov Flux 计算 (核心步骤)")
print("=" * 80)

# 步骤6a：Godunov flux (Torch)
def step6a_godunov_torch():
    with torch.no_grad():
        F1_g, F2_g, F3_g = EulerSolver.godunov_flux(
            rho_left, u_left, p_left,
            rho_right, u_right, p_right,
            gamma=gamma,
            backend='torch'
        )
    return F1_g.detach(), F2_g.detach(), F3_g.detach()

F1_g_torch, F2_g_torch, F3_g_torch = profile_step(
    "6a. Godunov flux (Torch backend)", step6a_godunov_torch
)

# 步骤6b：Godunov flux (Triton)
def step6b_godunov_triton():
    with torch.no_grad():
        F1_g, F2_g, F3_g = EulerSolver.godunov_flux(
            rho_left, u_left, p_left,
            rho_right, u_right, p_right,
            gamma=gamma,
            backend='triton'
        )
    return F1_g.detach(), F2_g.detach(), F3_g.detach()

F1_g_triton, F2_g_triton, F3_g_triton = profile_step(
    "6b. Godunov flux (Triton backend)", step6b_godunov_triton
)

print()
print("=" * 80)
print("后续步骤")
print("=" * 80)

# 步骤7：计算平均通量
F1_left = F1_sorted[:-1].flatten()
F1_right = F1_sorted[1:].flatten()
F2_left = F2_sorted[:-1].flatten()
F2_right = F2_sorted[1:].flatten()
F3_left = F3_sorted[:-1].flatten()
F3_right = F3_sorted[1:].flatten()

def step7_avg_flux():
    F1_avg = 0.5 * (F1_left + F1_right)
    F2_avg = 0.5 * (F2_left + F2_right)
    F3_avg = 0.5 * (F3_left + F3_right)
    return F1_avg, F2_avg, F3_avg

F1_avg, F2_avg, F3_avg = profile_step("7. 计算平均通量", step7_avg_flux)

# 步骤8：计算损失
def step8_compute_loss_torch():
    loss_F1 = ((F1_g_torch - F1_avg)**2).mean()
    loss_F2 = ((F2_g_torch - F2_avg)**2).mean()
    loss_F3 = ((F3_g_torch - F3_avg)**2).mean()
    loss = (loss_F1 + loss_F2 + loss_F3) * eta.mean()
    return loss

loss_torch = profile_step("8a. 计算损失 (Torch)", step8_compute_loss_torch)

def step8_compute_loss_triton():
    loss_F1 = ((F1_g_triton - F1_avg)**2).mean()
    loss_F2 = ((F2_g_triton - F2_avg)**2).mean()
    loss_F3 = ((F3_g_triton - F3_avg)**2).mean()
    loss = (loss_F1 + loss_F2 + loss_F3) * eta.mean()
    return loss

loss_triton = profile_step("8b. 计算损失 (Triton)", step8_compute_loss_triton)

print()
print("=" * 80)
print("总结")
print("=" * 80)
