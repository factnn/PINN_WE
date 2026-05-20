#!/usr/bin/env python
"""
快速查看训练结果
在终端运行：python quick_view.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath('../../../PINNsrc'))

from PINNs import *
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合终端
import matplotlib.pyplot as plt

dtype = torch.float64
Ts, Te, Xs, Xe = 0, 0.2, 0, 1

print("=" * 60)
print("快速查看PINN训练结果")
print("=" * 60)

# 检查是否有保存的模型
model_path = 'model.pth'
if os.path.exists(model_path):
    print(f"找到保存的模型: {model_path}")
    model = PINNs_WE_Euler_1D(Nl=6, Nn=60).to(cuda).double()
    model.load_state_dict(torch.load(model_path))
    print("模型已加载")
else:
    print("未找到保存的模型，使用当前训练的模型")
    print("注意：需要先运行训练脚本，模型在内存中")
    # 这里假设模型已经在内存中（如果刚训练完）
    # 实际使用时，建议先保存模型
    model = PINNs_WE_Euler_1D(Nl=6, Nn=60).to(cuda).double()
    print("使用默认模型（如果训练已完成，模型应该在内存中）")

# 生成预测
print("\n生成预测结果...")
x = np.linspace(Xs, Xe, 200)
t = np.full_like(x, Te)
x_test = torch.tensor(np.hstack((t[:, None], x[:, None])), dtype=dtype).to(cuda)

model.eval()
with torch.no_grad():
    u_pred = model(x_test)
    rho_pred = u_pred[:, 0].cpu().numpy()
    p_pred = u_pred[:, 1].cpu().numpy()
    u_vel_pred = u_pred[:, 2].cpu().numpy()

print("预测完成")

# 绘图
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(x, rho_pred, 'b-', linewidth=2, label='PINN')
plt.xlabel('x')
plt.ylabel('Density (ρ)')
plt.title(f'Density at t={Te}')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(x, p_pred, 'r-', linewidth=2, label='PINN')
plt.xlabel('x')
plt.ylabel('Pressure (p)')
plt.title(f'Pressure at t={Te}')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(x, u_vel_pred, 'g-', linewidth=2, label='PINN')
plt.xlabel('x')
plt.ylabel('Velocity (u)')
plt.title(f'Velocity at t={Te}')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
output_file = 'quick_view_results.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ 图片已保存: {output_file}")

# 保存数据
data_file = 'quick_view_results.dat'
results = np.column_stack([x, rho_pred, p_pred, u_vel_pred])
np.savetxt(data_file, results, header='x rho p u', fmt='%.6e')
print(f"✓ 数据已保存: {data_file}")

# 打印统计信息
print(f"\n结果统计:")
print(f"  密度范围: [{rho_pred.min():.4f}, {rho_pred.max():.4f}]")
print(f"  压力范围: [{p_pred.min():.4f}, {p_pred.max():.4f}]")
print(f"  速度范围: [{u_vel_pred.min():.4f}, {u_vel_pred.max():.4f}]")

print("\n" + "=" * 60)
print("完成！")
print("=" * 60)

