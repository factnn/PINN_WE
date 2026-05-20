#!/usr/bin/env python3
"""
诊断脚本：检查不同α值训练后学到的解
"""

import sys
sys.path.append('/share/project/zpy/PINN_WE/cases/self-discovery/src')

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from model import GuderleyPINN
from physics import compute_total_loss, rankine_hugoniot_bc

def diagnose_alpha_solution(alpha_value: float, output_dir: str = "output_scan_v2"):
    """
    诊断指定α值训练后的解

    Args:
        alpha_value: 要诊断的α值
        output_dir: 输出目录
    """
    print(f"\n{'='*60}")
    print(f"诊断 α = {alpha_value:.3f} 的解")
    print(f"{'='*60}")

    # GPU设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建并训练模型（简化版，只训练1000轮快速查看）
    model = GuderleyPINN(alpha_init=alpha_value)
    model.to(device)
    model.train()
    model.raw_alpha.requires_grad = False  # 固定α

    # 创建训练点
    xi_interior = torch.linspace(0.01, 0.9, 200).reshape(-1, 1).to(device)
    xi_boundary = torch.tensor([[1.0]]).to(device)
    xi_center = torch.tensor([[1e-3]]).to(device)

    # 快速训练（只用Adam，1000轮）
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3
    )

    print("\n快速训练 1000 轮...")
    for epoch in range(1, 1001):
        optimizer.zero_grad()
        total_loss, loss_dict = compute_total_loss(
            model, xi_interior, xi_boundary, xi_center,
            gamma=1.4, n=2, mach_inf=10.0
        )
        total_loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss_dict['total']:.6e}")

    # 评估解
    model.eval()
    with torch.no_grad():
        xi_plot = torch.linspace(0.001, 1.0, 500).reshape(-1, 1).to(device)
        V, C, G = model(xi_plot)

        # 移到CPU
        xi_plot = xi_plot.cpu().numpy()
        V = V.cpu().numpy()
        C = C.cpu().numpy()
        G = G.cpu().numpy()

    # 统计信息
    print(f"\n解的统计信息:")
    print(f"  V: min={V.min():.6f}, max={V.max():.6f}, range={V.max()-V.min():.6f}")
    print(f"  C: min={C.min():.6f}, max={C.max():.6f}, range={C.max()-C.min():.6f}")
    print(f"  G: min={G.min():.6f}, max={G.max():.6f}, range={G.max()-G.min():.6f}")

    # 检查边界条件
    V_bc_exact, C_bc_exact, G_bc_exact = rankine_hugoniot_bc(gamma=1.4, mach_inf=10.0)
    print(f"\n边界条件检查 (ξ=1):")
    print(f"  V: 预测={V[-1,0]:.6f}, 理论={V_bc_exact:.6f}, 误差={abs(V[-1,0]-V_bc_exact):.6e}")
    print(f"  C: 预测={C[-1,0]:.6f}, 理论={C_bc_exact:.6f}, 误差={abs(C[-1,0]-C_bc_exact):.6e}")
    print(f"  G: 预测={G[-1,0]:.6f}, 理论={G_bc_exact:.6f}, 误差={abs(G[-1,0]-G_bc_exact):.6e}")

    # 检查是否是平凡解
    V_variation = np.std(V)
    C_variation = np.std(C)

    print(f"\n变化程度:")
    print(f"  V 标准差: {V_variation:.6f}")
    print(f"  C 标准差: {C_variation:.6f}")

    if V_variation < 0.01 and C_variation < 0.01:
        print("\n⚠️  警告：解几乎是常数（可能是平凡解）")
    else:
        print("\n✓ 解有明显的空间变化")

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # V(ξ)
    axes[0, 0].plot(xi_plot, V, 'b-', linewidth=2)
    axes[0, 0].axhline(y=V_bc_exact, color='r', linestyle='--', label=f'BC理论值={V_bc_exact:.3f}')
    axes[0, 0].set_xlabel('ξ')
    axes[0, 0].set_ylabel('V (无量纲速度)')
    axes[0, 0].set_title(f'V(ξ) - α={alpha_value:.3f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # C(ξ)
    axes[0, 1].plot(xi_plot, C, 'g-', linewidth=2)
    axes[0, 1].axhline(y=C_bc_exact, color='r', linestyle='--', label=f'BC理论值={C_bc_exact:.3f}')
    axes[0, 1].set_xlabel('ξ')
    axes[0, 1].set_ylabel('C (无量纲声速)')
    axes[0, 1].set_title(f'C(ξ) - α={alpha_value:.3f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # G(ξ)
    axes[1, 0].plot(xi_plot, G, 'm-', linewidth=2)
    axes[1, 0].axhline(y=G_bc_exact, color='r', linestyle='--', label=f'BC理论值={G_bc_exact:.1f}')
    axes[1, 0].set_xlabel('ξ')
    axes[1, 0].set_ylabel('G (无量纲密度)')
    axes[1, 0].set_title(f'G(ξ) - α={alpha_value:.3f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 马赫数 M = V/C
    M = V / (C + 1e-6)
    axes[1, 1].plot(xi_plot, M, 'k-', linewidth=2)
    axes[1, 1].axhline(y=1.0, color='r', linestyle='--', label='声速线 M=1')
    axes[1, 1].set_xlabel('ξ')
    axes[1, 1].set_ylabel('M = V/C')
    axes[1, 1].set_title(f'马赫数分布 - α={alpha_value:.3f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    plot_file = output_path / f"solution_alpha_{alpha_value:.3f}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ 图像已保存: {plot_file}")
    plt.close()

    return V, C, G, xi_plot


if __name__ == "__main__":
    # 诊断几个关键的α值
    alpha_values_to_check = [0.717, 1.0]

    print("="*60)
    print("解诊断工具")
    print("="*60)
    print(f"将诊断以下α值: {alpha_values_to_check}")

    for alpha in alpha_values_to_check:
        diagnose_alpha_solution(alpha)

    print("\n"+"="*60)
    print("诊断完成！")
    print("="*60)
