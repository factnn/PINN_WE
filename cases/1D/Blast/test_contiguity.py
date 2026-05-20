#!/usr/bin/env python3
"""
测试数据连续性对 Triton 性能的影响
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
print("测试数据连续性对 Triton 性能的影响")
print("=" * 80)

# 场景1：连续数据（模拟单独测试）
n_points = 10000
rho_left_contig = torch.rand(n_points, dtype=dtype, device=device)
u_left_contig = torch.rand(n_points, dtype=dtype, device=device)
p_left_contig = torch.rand(n_points, dtype=dtype, device=device)
rho_right_contig = torch.rand(n_points, dtype=dtype, device=device)
u_right_contig = torch.rand(n_points, dtype=dtype, device=device)
p_right_contig = torch.rand(n_points, dtype=dtype, device=device)

print(f"\n场景1：连续数据")
print(f"  rho_left.is_contiguous() = {rho_left_contig.is_contiguous()}")

# 预热
for _ in range(10):
    _ = EulerSolver.godunov_flux(rho_left_contig, u_left_contig, p_left_contig,
                                  rho_right_contig, u_right_contig, p_right_contig,
                                  backend='triton')
torch.cuda.synchronize()

# 测试
torch.cuda.synchronize()
start = time.time()
for _ in range(n_iterations):
    F1, F2, F3 = EulerSolver.godunov_flux(rho_left_contig, u_left_contig, p_left_contig,
                                          rho_right_contig, u_right_contig, p_right_contig,
                                          backend='triton')
torch.cuda.synchronize()
time_contig = (time.time() - start) / n_iterations * 1000

print(f"  Triton 耗时: {time_contig:.4f} ms")

# 场景2：切片数据（模拟实际训练）
n_points_full = n_points + 1
rho_full = torch.rand(n_points_full, 1, dtype=dtype, device=device)
u_full = torch.rand(n_points_full, 1, dtype=dtype, device=device)
p_full = torch.rand(n_points_full, 1, dtype=dtype, device=device)

rho_left_slice = rho_full[:-1].flatten()
rho_right_slice = rho_full[1:].flatten()
u_left_slice = u_full[:-1].flatten()
u_right_slice = u_full[1:].flatten()
p_left_slice = p_full[:-1].flatten()
p_right_slice = p_full[1:].flatten()

print(f"\n场景2：切片数据（模拟实际训练）")
print(f"  rho_left.is_contiguous() = {rho_left_slice.is_contiguous()}")

# 预热
for _ in range(10):
    _ = EulerSolver.godunov_flux(rho_left_slice, u_left_slice, p_left_slice,
                                  rho_right_slice, u_right_slice, p_right_slice,
                                  backend='triton')
torch.cuda.synchronize()

# 测试
torch.cuda.synchronize()
start = time.time()
for _ in range(n_iterations):
    F1, F2, F3 = EulerSolver.godunov_flux(rho_left_slice, u_left_slice, p_left_slice,
                                          rho_right_slice, u_right_slice, p_right_slice,
                                          backend='triton')
torch.cuda.synchronize()
time_slice = (time.time() - start) / n_iterations * 1000

print(f"  Triton 耗时: {time_slice:.4f} ms")

# 场景3：切片数据 + contiguous()
rho_left_slice_contig = rho_full[:-1].flatten().contiguous()
rho_right_slice_contig = rho_full[1:].flatten().contiguous()
u_left_slice_contig = u_full[:-1].flatten().contiguous()
u_right_slice_contig = u_full[1:].flatten().contiguous()
p_left_slice_contig = p_full[:-1].flatten().contiguous()
p_right_slice_contig = p_full[1:].flatten().contiguous()

print(f"\n场景3：切片数据 + contiguous()")
print(f"  rho_left.is_contiguous() = {rho_left_slice_contig.is_contiguous()}")

# 预热
for _ in range(10):
    _ = EulerSolver.godunov_flux(rho_left_slice_contig, u_left_slice_contig, p_left_slice_contig,
                                  rho_right_slice_contig, u_right_slice_contig, p_right_slice_contig,
                                  backend='triton')
torch.cuda.synchronize()

# 测试
torch.cuda.synchronize()
start = time.time()
for _ in range(n_iterations):
    F1, F2, F3 = EulerSolver.godunov_flux(rho_left_slice_contig, u_left_slice_contig, p_left_slice_contig,
                                          rho_right_slice_contig, u_right_slice_contig, p_right_slice_contig,
                                          backend='triton')
torch.cuda.synchronize()
time_slice_contig = (time.time() - start) / n_iterations * 1000

print(f"  Triton 耗时: {time_slice_contig:.4f} ms")

print(f"\n{'='*80}")
print("总结")
print(f"{'='*80}")
print(f"场景1（连续数据）:           {time_contig:.4f} ms")
print(f"场景2（切片数据）:           {time_slice:.4f} ms  (慢 {time_slice/time_contig:.2f}x)")
print(f"场景3（切片+contiguous）:    {time_slice_contig:.4f} ms  (慢 {time_slice_contig/time_contig:.2f}x)")
