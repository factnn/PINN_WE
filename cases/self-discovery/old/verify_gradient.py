#!/usr/bin/env python3
"""
Legacy gradient check for the old PINN prototype.

当前默认主线请改看 `cases/self-discovery/canonical/README.md`。

验证 α 的梯度计算是否正确
"""

import sys
sys.path.append('/share/project/zpy/PINN_WE/cases/self-discovery/src')

import torch
import numpy as np
from model import GuderleyPINN
from physics import compute_total_loss

def verify_alpha_gradient():
    """验证 α 的梯度计算"""

    print("=" * 60)
    print("验证 α 梯度计算")
    print("=" * 60)

    # 测试点
    alpha_test = 0.80
    epsilon = 1e-5

    # 创建训练点
    xi_interior = torch.linspace(0.1, 1.0, 100).reshape(-1, 1)
    xi_boundary = torch.tensor([[1.0]])

    gamma = 1.4
    n = 2

    print(f"\n测试点: α = {alpha_test}")
    print(f"扰动: ε = {epsilon}")

    # === 方法 1: 自动微分计算梯度 ===
    print("\n" + "=" * 60)
    print("方法 1: PyTorch 自动微分")
    print("=" * 60)

    model = GuderleyPINN(alpha_init=alpha_test)
    model.train()

    # 前向传播
    total_loss, loss_dict = compute_total_loss(
        model, xi_interior, xi_boundary, gamma, n
    )

    # 反向传播
    total_loss.backward()

    # 获取梯度
    alpha_grad_auto = model.raw_alpha.grad.item()
    alpha_value = model.get_alpha_value()

    print(f"α 值: {alpha_value:.6f}")
    print(f"损失: {loss_dict['total']:.6e}")
    print(f"∂L/∂α (自动微分): {alpha_grad_auto:.6e}")

    # === 方法 2: 数值微分验证 ===
    print("\n" + "=" * 60)
    print("方法 2: 数值微分 (有限差分)")
    print("=" * 60)

    # L(α + ε)
    model_plus = GuderleyPINN(alpha_init=alpha_test + epsilon)
    model_plus.train()
    loss_plus, _ = compute_total_loss(
        model_plus, xi_interior, xi_boundary, gamma, n
    )
    loss_plus_val = loss_plus.item()

    # L(α - ε)
    model_minus = GuderleyPINN(alpha_init=alpha_test - epsilon)
    model_minus.train()
    loss_minus, _ = compute_total_loss(
        model_minus, xi_interior, xi_boundary, gamma, n
    )
    loss_minus_val = loss_minus.item()

    # 中心差分: dL/dα ≈ [L(α+ε) - L(α-ε)] / (2ε)
    alpha_grad_numerical = (loss_plus_val - loss_minus_val) / (2 * epsilon)

    print(f"L(α - ε): {loss_minus_val:.6e}")
    print(f"L(α):     {loss_dict['total']:.6e}")
    print(f"L(α + ε): {loss_plus_val:.6e}")
    print(f"∂L/∂α (数值): {alpha_grad_numerical:.6e}")

    # === 比较结果 ===
    print("\n" + "=" * 60)
    print("结果比较")
    print("=" * 60)

    print(f"\n自动微分: ∂L/∂α = {alpha_grad_auto:.6e}")
    print(f"数值微分: ∂L/∂α = {alpha_grad_numerical:.6e}")

    relative_error = abs(alpha_grad_auto - alpha_grad_numerical) / (abs(alpha_grad_numerical) + 1e-10)
    print(f"相对误差: {relative_error:.2%}")

    if relative_error < 0.01:
        print("\n✓ 梯度计算正确")
    else:
        print("\n✗ 梯度计算可能有误")

    # === 判断优化方向 ===
    print("\n" + "=" * 60)
    print("优化方向分析")
    print("=" * 60)

    print(f"\n当前 α = {alpha_test:.3f}")
    print(f"真实 α = 0.717")
    print(f"∂L/∂α = {alpha_grad_numerical:.6e}")

    if alpha_grad_numerical > 0:
        print("\n∂L/∂α > 0: 损失随 α 增大而增大")
        print("优化器应该: 减小 α (朝向 0.717)")
        print("梯度下降: α_new = α - lr * ∂L/∂α (减小 α) ✓")
    else:
        print("\n∂L/∂α < 0: 损失随 α 增大而减小")
        print("优化器应该: 增大 α (远离 0.717)")
        print("梯度下降: α_new = α - lr * ∂L/∂α (增大 α) ✗")

if __name__ == "__main__":
    verify_alpha_gradient()
