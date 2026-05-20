#!/usr/bin/env python3
"""
测试不同数据规模下 Triton vs Torch 的性能
"""
import sys
sys.path.insert(0, '/share/project/zpy/PINN_WE/PINNsrc')

import torch
import time
import numpy as np
from riemann_solver import EulerSolver

device = torch.device('cuda:0')
dtype = torch.float64
n_iterations = 50

print("=" * 80)
print("测试数据规模对 Triton 性能的影响")
print("=" * 80)
print(f"BLOCK_SIZE = 1024")
print()

# 测试不同的数据规模
sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]

results = []

for size in sizes:
    print(f"\n{'='*80}")
    print(f"数据规模: {size:,} 个点")
    print(f"{'='*80}")

    # 生成测试数据
    rho_left = torch.rand(size, dtype=dtype, device=device) * 0.5 + 0.5
    u_left = torch.rand(size, dtype=dtype, device=device) * 0.2 - 0.1
    p_left = torch.rand(size, dtype=dtype, device=device) * 0.5 + 0.5

    rho_right = torch.rand(size, dtype=dtype, device=device) * 0.3 + 0.1
    u_right = torch.rand(size, dtype=dtype, device=device) * 0.2 - 0.1
    p_right = torch.rand(size, dtype=dtype, device=device) * 0.3 + 0.1

    # 预热
    for _ in range(5):
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

    speedup = time_torch / time_triton

    # 计算 GPU 利用率
    n_blocks = (size + 1023) // 1024
    n_threads = n_blocks * 1024
    utilization = size / n_threads * 100

    print(f"Torch:       {time_torch:>8.4f} ms")
    print(f"Triton:      {time_triton:>8.4f} ms")
    print(f"加速比:      {speedup:>8.2f}x {'✓ Triton更快' if speedup > 1 else '✗ Torch更快'}")
    print(f"GPU blocks:  {n_blocks:>8,}")
    print(f"利用率:      {utilization:>8.1f}%")

    results.append({
        'size': size,
        'time_torch': time_torch,
        'time_triton': time_triton,
        'speedup': speedup,
        'n_blocks': n_blocks,
        'utilization': utilization
    })

# 总结
print(f"\n{'='*80}")
print("总结")
print(f"{'='*80}")
print(f"{'规模':<12} {'Torch(ms)':<12} {'Triton(ms)':<12} {'加速比':<10} {'Blocks':<10} {'利用率':<10}")
print("-" * 80)
for r in results:
    print(f"{r['size']:<12,} {r['time_torch']:<12.4f} {r['time_triton']:<12.4f} "
          f"{r['speedup']:<10.2f}x {r['n_blocks']:<10,} {r['utilization']:<10.1f}%")

# 分析
print(f"\n{'='*80}")
print("分析")
print(f"{'='*80}")

# 找到 Triton 开始变快的临界点
for i, r in enumerate(results):
    if r['speedup'] > 1.0:
        print(f"✓ Triton 在数据规模 >= {r['size']:,} 时开始比 Torch 快")
        break
else:
    print(f"✗ 在所有测试规模下，Triton 都没有比 Torch 快")

print(f"\n建议：")
print(f"  - 当前训练使用 10,000 个点，Triton 加速比 = {results[2]['speedup']:.2f}x")
if results[2]['speedup'] < 1:
    print(f"  - ⚠️  数据量太小，建议使用 Torch backend")
    print(f"  - 或者增加采样点数量到 >= 50,000")
else:
    print(f"  - ✓ Triton 已经比 Torch 快，可以继续使用")
