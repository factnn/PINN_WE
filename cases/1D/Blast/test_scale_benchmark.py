"""
测试不同数据规模下 Torch vs Triton 的性能
验证：Triton 在大规模数据下是否有优势
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'PINNsrc'))

import torch
import time
from PINNs import PINNs_WE_Euler_1D
from utility import loss_godunov_euler

cuda = torch.device('cuda')
dtype = torch.float64

# 测试不同规模
scales = [1000, 5000, 10000, 50000, 100000, 500000]

print("=" * 70)
print("Torch vs Triton 性能对比（不同数据规模）")
print("=" * 70)
print(f"{'数据点数':<12} {'Torch (ms)':<15} {'Triton (ms)':<15} {'加速比':<10}")
print("-" * 70)

model = PINNs_WE_Euler_1D(Nl=6, Nn=60).to(cuda).double()

for n_points in scales:
    x_int = torch.randn(n_points, 2, requires_grad=True, dtype=dtype, device=cuda)
    
    # 测试 Torch
    torch_times = []
    for _ in range(20):
        torch.cuda.synchronize()
        tic = time.time()
        loss = loss_godunov_euler(model, x_int, backend='torch')
        loss.backward()
        torch.cuda.synchronize()
        torch_times.append(time.time() - tic)
        model.zero_grad()
        if x_int.grad is not None:
            x_int.grad.zero_()
    
    torch_avg = sum(torch_times) / len(torch_times) * 1000
    
    # 测试 Triton
    triton_times = []
    for _ in range(20):
        torch.cuda.synchronize()
        tic = time.time()
        loss = loss_godunov_euler(model, x_int, backend='triton')
        loss.backward()
        torch.cuda.synchronize()
        triton_times.append(time.time() - tic)
        model.zero_grad()
        if x_int.grad is not None:
            x_int.grad.zero_()
    
    triton_avg = sum(triton_times) / len(triton_times) * 1000
    speedup = torch_avg / triton_avg
    
    print(f"{n_points:<12} {torch_avg:<15.2f} {triton_avg:<15.2f} {speedup:<10.2f}x")

print("=" * 70)
print("结论：观察加速比在大规模数据下是否 > 1.0")
print("=" * 70)
