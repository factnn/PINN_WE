#!/usr/bin/env python3
"""
调试脚本：验证绘图数据是否正确
"""
import numpy as np
import matplotlib.pyplot as plt

# 读取预测数据
pred_data = np.loadtxt('output/Sod_1GPU_fourier_20260119_185507/predictions.dat')
x_pred = pred_data[:, 0]
rho_pred = pred_data[:, 1]
p_pred = pred_data[:, 2]
u_pred = pred_data[:, 3]

# 读取精确解
exact_data = np.loadtxt('sode.dat')
x_exact = exact_data[:, 0]
rho_exact = exact_data[:, 1]
u_exact = exact_data[:, 2]  # sode.dat格式: [x, rho, u, p]
p_exact = exact_data[:, 3]

print("=" * 60)
print("数据验证")
print("=" * 60)
print(f"\n预测数据 (x=0.5附近):")
mask_pred = (x_pred >= 0.48) & (x_pred <= 0.52)
print(f"  x={x_pred[mask_pred][0]:.4f}: rho={rho_pred[mask_pred][0]:.4f}, p={p_pred[mask_pred][0]:.4f}, u={u_pred[mask_pred][0]:.4f}")

print(f"\n精确解 (x=0.5附近):")
mask_exact = (x_exact >= 0.48) & (x_exact <= 0.52)
print(f"  x={x_exact[mask_exact][0]:.4f}: rho={rho_exact[mask_exact][0]:.4f}, p={p_exact[mask_exact][0]:.4f}, u={u_exact[mask_exact][0]:.4f}")

print(f"\n数据范围:")
print(f"  预测 - rho: [{rho_pred.min():.3f}, {rho_pred.max():.3f}]")
print(f"  预测 - p:   [{p_pred.min():.3f}, {p_pred.max():.3f}]")
print(f"  预测 - u:   [{u_pred.min():.3f}, {u_pred.max():.3f}]")
print(f"  精确 - rho: [{rho_exact.min():.3f}, {rho_exact.max():.3f}]")
print(f"  精确 - p:   [{p_exact.min():.3f}, {p_exact.max():.3f}]")
print(f"  精确 - u:   [{u_exact.min():.3f}, {u_exact.max():.3f}]")

# 创建测试图
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 密度图
ax1 = axes[0]
ax1.plot(x_pred, rho_pred, 'b-', linewidth=2.5, label='Prediction', zorder=3)
ax1.plot(x_exact, rho_exact, 'r--', linewidth=2, label='Exact', alpha=0.8, zorder=2)
ax1.set_xlabel('Position x', fontsize=13, fontweight='bold')
ax1.set_ylabel('Density ρ', fontsize=13, fontweight='bold')
ax1.set_title('Density (应该显示密度)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 压力图
ax2 = axes[1]
ax2.plot(x_pred, p_pred, 'b-', linewidth=2.5, label='Prediction', zorder=3)
ax2.plot(x_exact, p_exact, 'r--', linewidth=2, label='Exact', alpha=0.8, zorder=2)
ax2.set_xlabel('Position x', fontsize=13, fontweight='bold')
ax2.set_ylabel('Pressure p', fontsize=13, fontweight='bold')
ax2.set_title('Pressure (应该显示压力)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 速度图
ax3 = axes[2]
ax3.plot(x_pred, u_pred, 'b-', linewidth=2.5, label='Prediction', zorder=3)
ax3.plot(x_exact, u_exact, 'r--', linewidth=2, label='Exact', alpha=0.8, zorder=2)
ax3.set_xlabel('Position x', fontsize=13, fontweight='bold')
ax3.set_ylabel('Velocity u', fontsize=13, fontweight='bold')
ax3.set_title('Velocity (应该显示速度)', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('debug_plot_verification.png', dpi=150, bbox_inches='tight')
print(f"\n调试图已保存: debug_plot_verification.png")
print("\n请检查:")
print("  1. 密度图的范围应该是 [0.12, 1.0]")
print("  2. 压力图的范围应该是 [0.09, 1.0]")
print("  3. 速度图的范围应该是 [-0.05, 1.0]")
print("  4. 如果密度图的范围是 [0.09, 1.0]，说明显示的是压力数据")
