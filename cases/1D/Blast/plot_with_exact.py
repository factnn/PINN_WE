"""
带精确解对比的可视化
一眼就能看出算得好不好！
"""
import sys
import os
sys.path.insert(0, os.path.abspath('../../../PINNsrc'))

from PINNs import *
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

dtype = torch.float64
Ts, Te, Xs, Xe = 0, 0.2, 0, 1

print("=" * 60)
print("生成带精确解对比的结果图")
print("=" * 60)

# ========== 加载模型 ==========
model_path = 'model.pth'
if os.path.exists(model_path):
    print(f"加载模型: {model_path}")
    model = PINNs_WE_Euler_1D(Nl=6, Nn=60).to(cuda).double()
    model.load_state_dict(torch.load(model_path))
    print("✓ 模型已加载")
else:
    print("❌ 未找到模型文件！请先运行训练脚本。")
    print("   运行: python 1.py")
    exit(1)

# ========== 加载精确解 ==========
exact_file = 'sode.dat'
if os.path.exists(exact_file):
    print(f"加载精确解: {exact_file}")
    exact_data = np.loadtxt(exact_file)
    x_exact = exact_data[:, 0]
    rho_exact = exact_data[:, 1]
    u_exact = exact_data[:, 2]
    p_exact = exact_data[:, 3]
    print(f"✓ 精确解已加载 ({len(x_exact)} 个点)")
else:
    print(f"⚠ 未找到精确解文件 {exact_file}，将只显示预测结果")
    x_exact = None

# ========== 生成PINN预测 ==========
print("\n生成PINN预测...")
x = np.linspace(Xs, Xe, 200)
t = np.full_like(x, Te)
x_test = torch.tensor(np.hstack((t[:, None], x[:, None])), dtype=dtype).to(cuda)

model.eval()
with torch.no_grad():
    u_pred = model(x_test)
    rho_pred = u_pred[:, 0].cpu().numpy()
    p_pred = u_pred[:, 1].cpu().numpy()
    u_vel_pred = u_pred[:, 2].cpu().numpy()

print("✓ 预测完成")

# ========== 计算误差（如果有精确解）==========
if x_exact is not None:
    # 插值到相同网格
    from scipy.interpolate import interp1d
    rho_exact_interp = interp1d(x_exact, rho_exact, kind='linear', 
                                bounds_error=False, fill_value='extrapolate')(x)
    p_exact_interp = interp1d(x_exact, p_exact, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')(x)
    u_exact_interp = interp1d(x_exact, u_exact, kind='linear', 
                               bounds_error=False, fill_value='extrapolate')(x)
    
    # 计算相对误差
    l2_rho = np.sqrt(np.mean((rho_pred - rho_exact_interp)**2)) / np.sqrt(np.mean(rho_exact_interp**2))
    l2_p = np.sqrt(np.mean((p_pred - p_exact_interp)**2)) / np.sqrt(np.mean(p_exact_interp**2))
    l2_u = np.sqrt(np.mean((u_vel_pred - u_exact_interp)**2)) / np.sqrt(np.mean(u_exact_interp**2))
    
    print(f"\n相对L2误差:")
    print(f"  密度: {l2_rho:.4%}")
    print(f"  压力: {l2_p:.4%}")
    print(f"  速度: {l2_u:.4%}")
    
    # 判断好坏
    avg_error = (l2_rho + l2_p + l2_u) / 3
    if avg_error < 0.05:
        quality = "优秀 ✓✓✓"
    elif avg_error < 0.10:
        quality = "良好 ✓✓"
    elif avg_error < 0.20:
        quality = "一般 ✓"
    else:
        quality = "需要改进 ✗"
    
    print(f"\n总体评价: {quality} (平均误差: {avg_error:.2%})")

# ========== 绘图 ==========
print("\n生成对比图...")

fig = plt.figure(figsize=(18, 6))

# 密度
ax1 = plt.subplot(1, 3, 1)
ax1.plot(x, rho_pred, 'b-', linewidth=2.5, label='PINN预测', zorder=3)
if x_exact is not None:
    ax1.plot(x_exact, rho_exact, 'r--', linewidth=2, label='精确解', alpha=0.8, zorder=2)
    ax1.fill_between(x, rho_pred, rho_exact_interp, alpha=0.2, color='gray', label='误差区域')
ax1.set_xlabel('位置 x', fontsize=13, fontweight='bold')
ax1.set_ylabel('密度 ρ', fontsize=13, fontweight='bold')
ax1.set_title(f'密度对比 (t={Te})', fontsize=14, fontweight='bold')
if x_exact is not None:
    ax1.text(0.02, 0.98, f'相对误差: {l2_rho:.2%}', 
             transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.8))
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11, loc='best')

# 压力
ax2 = plt.subplot(1, 3, 2)
ax2.plot(x, p_pred, 'b-', linewidth=2.5, label='PINN预测', zorder=3)
if x_exact is not None:
    ax2.plot(x_exact, p_exact, 'r--', linewidth=2, label='精确解', alpha=0.8, zorder=2)
    ax2.fill_between(x, p_pred, p_exact_interp, alpha=0.2, color='gray', label='误差区域')
ax2.set_xlabel('位置 x', fontsize=13, fontweight='bold')
ax2.set_ylabel('压力 p', fontsize=13, fontweight='bold')
ax2.set_title(f'压力对比 (t={Te})', fontsize=14, fontweight='bold')
if x_exact is not None:
    ax2.text(0.02, 0.98, f'相对误差: {l2_p:.2%}', 
             transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.8))
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=11, loc='best')

# 速度
ax3 = plt.subplot(1, 3, 3)
ax3.plot(x, u_vel_pred, 'b-', linewidth=2.5, label='PINN预测', zorder=3)
if x_exact is not None:
    ax3.plot(x_exact, u_exact, 'r--', linewidth=2, label='精确解', alpha=0.8, zorder=2)
    ax3.fill_between(x, u_vel_pred, u_exact_interp, alpha=0.2, color='gray', label='误差区域')
ax3.set_xlabel('位置 x', fontsize=13, fontweight='bold')
ax3.set_ylabel('速度 u', fontsize=13, fontweight='bold')
ax3.set_title(f'速度对比 (t={Te})', fontsize=14, fontweight='bold')
if x_exact is not None:
    ax3.text(0.02, 0.98, f'相对误差: {l2_u:.2%}', 
             transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.8))
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(fontsize=11, loc='best')

plt.tight_layout()

# 保存图片
output_file = 'results_with_exact.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ 图片已保存: {output_file}")

# 保存数据
data_file = 'results_with_exact.dat'
if x_exact is not None:
    results = np.column_stack([x, rho_pred, p_pred, u_vel_pred, 
                               rho_exact_interp, p_exact_interp, u_exact_interp])
    np.savetxt(data_file, results, 
               header='x rho_pred p_pred u_pred rho_exact p_exact u_exact', 
               fmt='%.6e')
else:
    results = np.column_stack([x, rho_pred, p_pred, u_vel_pred])
    np.savetxt(data_file, results, header='x rho p u', fmt='%.6e')
print(f"✓ 数据已保存: {data_file}")

print("\n" + "=" * 60)
if x_exact is not None:
    print(f"总体评价: {quality}")
    print("=" * 60)
    print("\n看图说明:")
    print("  - 蓝色实线 = PINN预测结果")
    print("  - 红色虚线 = 精确解（标准答案）")
    print("  - 灰色区域 = 误差大小")
    print("  - 两条线越重合，算得越好！")
    print("  - 激波位置（x≈0.5附近）应该很陡，不能有振荡")
else:
    print("提示: 找到精确解文件后，可以显示对比图")
print("=" * 60)

