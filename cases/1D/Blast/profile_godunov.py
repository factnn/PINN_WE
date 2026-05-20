#!/usr/bin/env python3
"""
性能分析脚本：对比 Torch 和 Triton 的 Godunov flux 性能
"""
import sys
sys.path.insert(0, '/share/project/zpy/PINN_WE/PINNsrc')

import torch
import time
import numpy as np
from riemann_solver import EulerSolver

# 设置
device = torch.device('cuda:0')
dtype = torch.float64

# 测试不同的数据规模
sizes = [1000, 5000, 10000, 20000, 50000]
n_iterations = 100  # 每个规模测试100次

print("=" * 80)
print("Godunov Flux 性能分析")
print("=" * 80)
print(f"设备: {device}")
print(f"数据类型: {dtype}")
print(f"每个规模测试次数: {n_iterations}")
print()

results = []

for size in sizes:
    print(f"\n{'='*80}")
    print(f"数据规模: {size} 个点")
    print(f"{'='*80}")

    # 生成测试数据
    rho_left = torch.rand(size, dtype=dtype, device=device) * 0.5 + 0.5
    u_left = torch.rand(size, dtype=dtype, device=device) * 0.2 - 0.1
    p_left = torch.rand(size, dtype=dtype, device=device) * 0.5 + 0.5

    rho_right = torch.rand(size, dtype=dtype, device=device) * 0.3 + 0.1
    u_right = torch.rand(size, dtype=dtype, device=device) * 0.2 - 0.1
    p_right = torch.rand(size, dtype=dtype, device=device) * 0.3 + 0.1

    # 预热
    for _ in range(10):
        _ = EulerSolver.godunov_flux(rho_left, u_left, p_left,
                                      rho_right, u_right, p_right,
                                      backend='torch')
        _ = EulerSolver.godunov_flux(rho_left, u_left, p_left,
                                      rho_right, u_right, p_right,
                                      backend='triton')
    torch.cuda.synchronize()

    # 测试 Torch
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iterations):
        F1_torch, F2_torch, F3_torch = EulerSolver.godunov_flux(
            rho_left, u_left, p_left,
            rho_right, u_right, p_right,
            backend='torch'
        )
    torch.cuda.synchronize()
    time_torch = (time.time() - start) / n_iterations * 1000  # ms

    # 测试 Triton
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iterations):
        F1_triton, F2_triton, F3_triton = EulerSolver.godunov_flux(
            rho_left, u_left, p_left,
            rho_right, u_right, p_right,
            backend='triton'
        )
    torch.cuda.synchronize()
    time_triton = (time.time() - start) / n_iterations * 1000  # ms

    # 验证结果一致性
    max_diff_F1 = torch.max(torch.abs(F1_torch - F1_triton)).item()
    max_diff_F2 = torch.max(torch.abs(F2_torch - F2_triton)).item()
    max_diff_F3 = torch.max(torch.abs(F3_torch - F3_triton)).item()

    speedup = time_torch / time_triton

    print(f"Torch:   {time_torch:.4f} ms")
    print(f"Triton:  {time_triton:.4f} ms")
    print(f"加速比:  {speedup:.2f}x {'(Triton更快)' if speedup > 1 else '(Torch更快)'}")
    print(f"最大误差: F1={max_diff_F1:.2e}, F2={max_diff_F2:.2e}, F3={max_diff_F3:.2e}")

    results.append({
        'size': size,
        'time_torch': time_torch,
        'time_triton': time_triton,
        'speedup': speedup
    })

# 总结
print(f"\n{'='*80}")
print("总结")
print(f"{'='*80}")
print(f"{'规模':<10} {'Torch (ms)':<15} {'Triton (ms)':<15} {'加速比':<10}")
print("-" * 80)
for r in results:
    print(f"{r['size']:<10} {r['time_torch']:<15.4f} {r['time_triton']:<15.4f} {r['speedup']:<10.2f}x")

# 分析
print(f"\n{'='*80}")
print("分析")
print(f"{'='*80}")

avg_speedup = np.mean([r['speedup'] for r in results])
if avg_speedup < 1:
    print(f"⚠️  Triton 平均比 Torch 慢 {1/avg_speedup:.2f}x")
    print("\n可能的原因：")
    print("1. Kernel 启动开销：Triton 每次调用都有固定的启动开销")
    print("2. BLOCK_SIZE 不是最优：当前 BLOCK_SIZE=1024 可能不适合这个数据规模")
    print("3. 分支预测：HLLC 算法有很多 if-else 分支，可能影响 GPU 性能")
    print("4. 寄存器压力：HLLC 算法需要很多临时变量，可能导致寄存器溢出")
else:
    print(f"✓ Triton 平均比 Torch 快 {avg_speedup:.2f}x")
