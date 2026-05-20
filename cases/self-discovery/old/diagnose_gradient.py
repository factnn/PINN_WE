#!/usr/bin/env python3
"""
诊断 α 梯度方向问题
"""

import sys
sys.path.append('/share/project/zpy/PINN_WE/cases/self-discovery/src')

import torch
import numpy as np
from model import GuderleyPINN
from physics import guderley_ode_residual, rankine_hugoniot_bc

def test_gradient_direction():
    """测试不同 α 值下的损失函数值"""

    print("=" * 60)
    print("测试 α 对损失函数的影响")
    print("=" * 60)

    # 测试不同的 α 值
    alpha_values = [0.70, 0.717, 0.75, 0.80, 0.85]

    # 创建测试点
    xi_test = torch.linspace(0.1, 1.0, 100).reshape(-1, 1)
    xi_bc = torch.tensor([[1.0]])

    gamma = 1.4
    n = 2

    results = []

    for alpha_val in alpha_values:
        # 创建模型
        model = GuderleyPINN(alpha_init=alpha_val)
        model.train()  # 需要训练模式才能计算梯度

        # 计算 PDE 残差（不使用 no_grad）
        res_V, res_C = guderley_ode_residual(model, xi_test, gamma, n)
        loss_pde = torch.mean(res_V**2 + res_C**2)

        # 计算边界条件损失
        V_bc, C_bc, G_bc = model(xi_bc)
        V_rh, C_rh, G_rh = rankine_hugoniot_bc(model, gamma)
        loss_bc = (V_bc - V_rh)**2 + (C_bc - C_rh)**2 + (G_bc - G_rh)**2

        # 总损失
        loss_total = loss_pde + loss_bc

        with torch.no_grad():
            results.append({
                'alpha': alpha_val,
                'loss_pde': loss_pde.item(),
                'loss_bc': loss_bc.item(),
                'loss_total': loss_total.item()
            })

            print(f"\nα = {alpha_val:.3f}:")
            print(f"  PDE Loss:   {loss_pde.item():.6e}")
            print(f"  BC Loss:    {loss_bc.item():.6e}")
            print(f"  Total Loss: {loss_total.item():.6e}")

    print("\n" + "=" * 60)
    print("损失函数随 α 的变化趋势")
    print("=" * 60)

    for i in range(len(results) - 1):
        alpha1 = results[i]['alpha']
        alpha2 = results[i+1]['alpha']
        loss1 = results[i]['loss_total']
        loss2 = results[i+1]['loss_total']

        delta_alpha = alpha2 - alpha1
        delta_loss = loss2 - loss1
        gradient_sign = "+" if delta_loss > 0 else "-"

        print(f"\nα: {alpha1:.3f} → {alpha2:.3f} (Δα = {delta_alpha:+.3f})")
        print(f"  Loss: {loss1:.6e} → {loss2:.6e}")
        print(f"  ΔLoss = {delta_loss:+.6e} ({gradient_sign})")
        print(f"  ∂L/∂α 符号: {gradient_sign}")

    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)

    # 找到最小损失对应的 α
    min_idx = min(range(len(results)), key=lambda i: results[i]['loss_total'])
    best_alpha = results[min_idx]['alpha']

    print(f"\n在测试的 α 值中，最小损失对应:")
    print(f"  α = {best_alpha:.3f}")
    print(f"  Loss = {results[min_idx]['loss_total']:.6e}")

    print(f"\n真实值 α = 0.717")
    print(f"当前训练结果 α = 0.837")

    # 判断梯度方向
    true_idx = 1  # α = 0.717
    current_idx = 4  # α = 0.85 (接近 0.837)

    if results[true_idx]['loss_total'] < results[current_idx]['loss_total']:
        print("\n✓ 真实值的损失更小，说明优化方向应该是减小 α")
        print("✗ 但训练过程中 α 增大了，说明梯度符号错误！")
    else:
        print("\n✗ 真实值的损失反而更大，说明方程可能有问题")

if __name__ == "__main__":
    test_gradient_direction()
