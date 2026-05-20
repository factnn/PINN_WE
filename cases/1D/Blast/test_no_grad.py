#!/usr/bin/env python3
"""
测试 torch.no_grad() 对 Triton 性能的影响
"""
import sys
sys.path.insert(0, '/share/project/zpy/PINN_WE/PINNsrc')

import torch
import time
from riemann_solver import EulerSolver

device = torch.device('cuda:0')
dtype = torch.float64
n_iterations = 100

print("=" * 80)
print("测试 torch.no_grad() 对 Triton 性能的影响")
print("=" * 80)

# 准备数据
n_points = 10000
rho_left = torch.rand(n_points, dtype=dtype, device=device)
u_left = torch.rand(n_points, dtype=dtype, device=device)
p_left = torch.rand(n_points, dtype=dtype, device=device)
rho_right = torch.rand(n_points, dtype=dtype, device=device)
u_right = torch.rand(n_points, dtype=dtype, device=device)
p_right = torch.rand(n_points, dtype=dtype, device=device)

# 场景1：无 no_grad
print(f"\n场景1：无 torch.no_grad()")

# 预热
for _ in range(10):
    _ = EulerSolver.godunov_flux(rho_left, u_left, p_left,
                                  rho_right, u_right, p_right,
                                  backend='triton')
torch.cuda.synchronize()

# 测试
torch.cuda.synchronize()
start = time.time()
for _ in range(n_iterations):
    F1, F2, F3 = EulerSolver.godunov_flux(rho_left, u_left, p_left,
                                          rho_right, u_right, p_right,
                                          backend='triton')
torch.cuda.synchronize()
time_no_grad_off = (time.time() - start) / n_iterations * 1000

print(f"  Triton 耗时: {time_no_grad_off:.4f} ms")

# 场景2：有 no_grad
print(f"\n场景2：有 torch.no_grad()")

# 预热
for _ in range(10):
    with torch.no_grad():
        _ = EulerSolver.godunov_flux(rho_left, u_left, p_left,
                                      rho_right, u_right, p_right,
                                      backend='triton')
torch.cuda.synchronize()

# 测试
torch.cuda.synchronize()
start = time.time()
for _ in range(n_iterations):
    with torch.no_grad():
        F1, F2, F3 = EulerSolver.godunov_flux(rho_left, u_left, p_left,
                                              rho_right, u_right, p_right,
                                              backend='triton')
torch.cuda.synchronize()
time_no_grad_on = (time.time() - start) / n_iterations * 1000

print(f"  Triton 耗时: {time_no_grad_on:.4f} ms")

# 场景3：有 no_grad + detach
print(f"\n场景3：有 torch.no_grad() + detach()")

# 预热
for _ in range(10):
    with torch.no_grad():
        F1, F2, F3 = EulerSolver.godunov_flux(rho_left, u_left, p_left,
                                              rho_right, u_right, p_right,
                                              backend='triton')
    F1 = F1.detach()
    F2 = F2.detach()
    F3 = F3.detach()
torch.cuda.synchronize()

# 测试
torch.cuda.synchronize()
start = time.time()
for _ in range(n_iterations):
    with torch.no_grad():
        F1, F2, F3 = EulerSolver.godunov_flux(rho_left, u_left, p_left,
                                              rho_right, u_right, p_right,
                                              backend='triton')
    F1 = F1.detach()
    F2 = F2.detach()
    F3 = F3.detach()
torch.cuda.synchronize()
time_no_grad_detach = (time.time() - start) / n_iterations * 1000

print(f"  Triton 耗时: {time_no_grad_detach:.4f} ms")

print(f"\n{'='*80}")
print("总结")
print(f"{'='*80}")
print(f"场景1（无 no_grad）:         {time_no_grad_off:.4f} ms")
print(f"场景2（有 no_grad）:         {time_no_grad_on:.4f} ms")
print(f"场景3（no_grad + detach）:   {time_no_grad_detach:.4f} ms")
