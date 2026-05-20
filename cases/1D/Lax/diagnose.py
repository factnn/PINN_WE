"""
诊断脚本：检查Lax问题的网络初始输出和损失函数值
"""
import sys
sys.path.insert(0, '/share/project/zpy/PINN_WE')

import torch
import numpy as np
from cases.1D.Lax.one import DNN, IC, Unit_var, gradients

# 设置参数
dtype = torch.float64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始条件（归一化前）
crhoL = 0.445
cuL = 0.698
cpL = 3.528
crhoR = 0.5
cuR = 0
cpR = 0.571
Te = 1.4
Xs = 0
Xe = 1

# 归一化
crhoL, cuL, cpL, crhoR, cuR, cpR, Te, rhoref, uref, pref = Unit_var(
    crhoL, cuL, cpL, crhoR, cuR, cpR, Te
)

print("=" * 60)
print("归一化后的参数:")
print(f"  crhoL={crhoL:.6f}, cuL={cuL:.6f}, cpL={cpL:.6f}")
print(f"  crhoR={crhoR:.6f}, cuR={cuR:.6f}, cpR={cpR:.6f}")
print(f"  Te={Te:.6f}")
print(f"  rhoref={rhoref:.6f}, uref={uref:.6f}, pref={pref:.6f}")
print("=" * 60)

# 创建模型
model = DNN().to(device).double()
model.eval()

# 测试点：初始时刻的几个点
x_test = np.array([
    [0.0, 0.25],  # 左侧
    [0.0, 0.5],   # 间断点
    [0.0, 0.75],  # 右侧
    [Te, 0.25],   # 最终时刻左侧
    [Te, 0.5],    # 最终时刻间断点
    [Te, 0.75],   # 最终时刻右侧
])
x_test_tensor = torch.tensor(x_test, dtype=dtype).to(device)

# 网络初始输出
with torch.no_grad():
    y_pred = model(x_test_tensor)
    rho_pred = y_pred[:, 0].cpu().numpy()
    p_pred = y_pred[:, 1].cpu().numpy()
    u_pred = y_pred[:, 2].cpu().numpy()

print("\n网络初始输出（未训练）:")
print(f"{'位置':<15} {'密度(rho)':<15} {'压力(p)':<15} {'速度(u)':<15}")
print("-" * 60)
for i, (t, x) in enumerate(x_test):
    print(f"t={t:.2f}, x={x:.2f}  {rho_pred[i]:>14.6f}  {p_pred[i]:>14.6f}  {u_pred[i]:>14.6f}")

# 检查初始条件
print("\n期望的初始条件（归一化后）:")
print(f"  左侧 (x<=0.5): rho={crhoL:.6f}, u={cuL:.6f}, p={cpL:.6f}")
print(f"  右侧 (x>0.5):  rho={crhoR:.6f}, u={cuR:.6f}, p={cpR:.6f}")

# 计算初始条件的损失
x_ic_test = np.array([[0.0, 0.25], [0.0, 0.5], [0.0, 0.75]])
rho_ic_test, u_ic_test, p_ic_test = IC(x_ic_test)
x_ic_tensor = torch.tensor(x_ic_test, dtype=dtype).to(device)
rho_ic_tensor = torch.tensor(rho_ic_test, dtype=dtype).to(device)
u_ic_tensor = torch.tensor(u_ic_test, dtype=dtype).to(device)
p_ic_tensor = torch.tensor(p_ic_test, dtype=dtype).to(device)

actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
loss_ic = actual_model.loss_ic(x_ic_tensor, rho_ic_tensor, u_ic_tensor, p_ic_tensor)

print(f"\n初始条件损失 (loss_ic): {loss_ic.item():.6f}")

# 检查网络输出范围
print("\n网络输出范围:")
print(f"  rho: [{rho_pred.min():.6f}, {rho_pred.max():.6f}]")
print(f"  p:   [{p_pred.min():.6f}, {p_pred.max():.6f}]")
print(f"  u:   [{u_pred.min():.6f}, {u_pred.max():.6f}]")

# 检查是否有NaN或Inf
print("\n检查NaN/Inf:")
print(f"  rho中有NaN: {np.isnan(rho_pred).any()}, Inf: {np.isinf(rho_pred).any()}")
print(f"  p中有NaN:   {np.isnan(p_pred).any()}, Inf: {np.isinf(p_pred).any()}")
print(f"  u中有NaN:   {np.isnan(u_pred).any()}, Inf: {np.isinf(u_pred).any()}")

print("\n" + "=" * 60)
print("诊断完成！")
print("=" * 60)

