"""
可视化训练结果
可以单独运行，用于查看已训练模型的结果
"""
import sys
import os
sys.path.insert(0, os.path.abspath('../../../PINNsrc'))

from PINNs import *
import torch
import numpy as np
import matplotlib.pyplot as plt

dtype = torch.float64

# ========== 参数设置（需要和训练时一致）==========
Ts = 0
Te = 0.2
Xs = 0
Xe = 1

# ========== 加载模型 ==========
print("加载模型...")
model = PINNs_WE_Euler_1D(Nl=6, Nn=60).to(cuda).double()

# 如果保存了模型，可以加载：
# model.load_state_dict(torch.load('model.pth'))
# 否则使用当前训练好的模型（需要先运行1.py）

print("模型已加载")

# ========== 生成预测结果 ==========
print("生成预测结果...")

# 在最终时刻预测
x = np.linspace(Xs, Xe, 200)
t = np.full_like(x, Te)
x_test = np.hstack((t[:, None], x[:, None]))
x_test_tensor = torch.tensor(x_test, dtype=dtype).to(cuda)

# 预测
model.eval()
with torch.no_grad():
    u_pred = model(x_test_tensor)
    rho_pred = u_pred[:, 0].cpu().numpy()
    p_pred = u_pred[:, 1].cpu().numpy()
    u_vel_pred = u_pred[:, 2].cpu().numpy()

print("预测完成")

# ========== 可视化 ==========
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(x, rho_pred, 'b-', linewidth=2, label='PINN Prediction')
plt.xlabel('x', fontsize=12)
plt.ylabel('Density (ρ)', fontsize=12)
plt.title(f'Density at t={Te}', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(x, p_pred, 'r-', linewidth=2, label='PINN Prediction')
plt.xlabel('x', fontsize=12)
plt.ylabel('Pressure (p)', fontsize=12)
plt.title(f'Pressure at t={Te}', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(x, u_vel_pred, 'g-', linewidth=2, label='PINN Prediction')
plt.xlabel('x', fontsize=12)
plt.ylabel('Velocity (u)', fontsize=12)
plt.title(f'Velocity at t={Te}', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('results.png', dpi=300, bbox_inches='tight')
print("结果已保存到: results.png")
plt.show()

# 保存数据
results = np.column_stack([x, rho_pred, p_pred, u_vel_pred])
np.savetxt('results.dat', results, header='x rho p u', fmt='%.6e')
print("数据已保存到: results.dat")

