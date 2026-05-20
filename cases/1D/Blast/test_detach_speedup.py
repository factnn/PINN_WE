"""
测试 detach() 对 Godunov 损失的加速效果
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'PINNsrc'))

import torch
import time
from PINNs import PINNs_WE_Euler_1D
from utility import loss_godunov_euler

# 设置
cuda = torch.device('cuda')
dtype = torch.float64
n_points = 10000

# 创建模型和数据
model = PINNs_WE_Euler_1D(Nl=6, Nn=60).to(cuda).double()
x_int = torch.randn(n_points, 2, requires_grad=True, dtype=dtype, device=cuda)

print("=" * 60)
print("测试 Godunov 损失的梯度截断效果")
print("=" * 60)
print(f"数据点数: {n_points}")
print(f"测试次数: 100 次")
print()

# 测试 Torch 后端
for backend in ['torch', 'triton']:
    print(f"--- 测试 {backend.upper()} 后端 ---")
    
    times = []
    for i in range(100):
        torch.cuda.synchronize()
        tic = time.time()
        
        loss = loss_godunov_euler(model, x_int, backend=backend, lambda_godunov=10.0)
        loss.backward()
        
        torch.cuda.synchronize()
        toc = time.time()
        times.append(toc - tic)
        
        # 清空梯度
        model.zero_grad()
        if x_int.grad is not None:
            x_int.grad.zero_()
    
    avg_time = sum(times) / len(times) * 1000  # ms
    print(f"  平均耗时: {avg_time:.2f} ms/次")
    print()

print("=" * 60)
print("注意：现在 Godunov 通量已经 detach，不参与梯度计算")
print("预期：Triton 应该和 Torch 速度相近（都不需要对 Riemann Solver 求导）")
print("=" * 60)
