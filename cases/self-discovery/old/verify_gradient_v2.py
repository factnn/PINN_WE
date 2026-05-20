#!/usr/bin/env python3
"""
Legacy gradient check for the old PINN prototype.

当前默认主线请改看 `cases/self-discovery/canonical/README.md`。

验证 α 的梯度计算 - 使用同一个模型
"""

import sys
sys.path.append('/share/project/zpy/PINN_WE/cases/self-discovery/src')

import torch
import numpy as np
from model import GuderleyPINN
from physics import compute_total_loss

def verify_alpha_gradient_v2():
    """验证 α 的梯度计算 - 使用同一个模型"""

    print("=" * 60)
    print("验证 α 梯度计算 (使用同一个模型)")
    print("=" * 60)

    # 测试点
    alpha_test = 0.80
    epsilon = 1e-4  # 使用更大的扰动

    # 创建训练点
    xi_interior = torch.linspace(0.1, 1.0, 100).reshape(-1, 1)
    xi_boundary = torch.tensor([[1.0]])

    gamma = 1.4
    n = 2

    print(f"\n测试点: α = {alpha_test}")
    print(f"扰动: ε = {epsilon}")

    # 创建一个模型
    model = GuderleyPINN(alpha_init=alpha_test)
    model.train()

    # === 方法 1: 自动微分计算梯度 ===
    print("\n" + "=" * 60)
    print("方法 1: PyTorch 自动微分")
    print("=" * 60)

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

    # === 方法 2: 数值微分验证 (使用同一个模型) ===
    print("\n" + "=" * 60)
    print("方法 2: 数值微分 (使用同一个模型)")
    print("=" * 60)

    # 保存当前 α 值
    alpha_original = model.raw_alpha.data.clone()

    # L(α - ε)
    model.raw_alpha.data = torch.tensor(alpha_test - epsilon)
    loss_minus, _ = compute_total_loss(
        model, xi_interior, xi_boundary, gamma, n
    )
    loss_minus_val = loss_minus.item()

    # L(α)
    model.raw_alpha.data = torch.tensor(alpha_test)
    loss_center, _ = compute_total_loss(
        model, xi_interior, xi_boundary, gamma, n
    )
    loss_center_val = loss_center.item()

    # L(α + ε)
    model.raw_alpha.data = torch.tensor(alpha_test + epsilon)
    loss_plus, _ = compute_total_loss(
        model, xi_interior, xi_boundary, gamma, n
    )
    loss_plus_val = loss_plus.item()

    # 恢复原始值
    model.raw_alpha.data = alpha_original

    # 中心差分: dL/dα ≈ [L(α+ε) - L(α-ε)] / (2ε)
    alpha_grad_numerical = (loss_plus_val - loss_minus_val) / (2 * epsilon)

    print(f"L(α - ε = {alpha_test - epsilon:.6f}): {loss_minus_val:.6e}")
    print(f"L(α     = {alpha_test:.6f}): {loss_center_val:.6e}")
    print(f"L(α + ε = {alpha_test + epsilon:.6f}): {loss_plus_val:.6e}")
    print(f"∂L/∂α (数值): {alpha_grad_numerical:.6e}")

    # === 比较结果 ===
    print("\n" + "=" * 60)
    print("结果比较")
    print("=" * 60)

    print(f"\n自动微分: ∂L/∂α = {alpha_grad_auto:.6e}")
    print(f"数值微分: ∂L/∂α = {alpha_grad_numerical:.6e}")

    relative_error = abs(alpha_grad_auto - alpha_grad_numerical) / (abs(alpha_grad_numerical) + 1e-10)
    print(f"相对误差: {relative_error:.2%}")

    if relative_error < 0.1:
        print("\n✓ 梯度计算正确")
    else:
        print("\n✗ 梯度计算有误")
        print(f"  差异: {alpha_grad_auto - alpha_grad_numerical:.6e}")

    # === 判断优化方向 ===
    print("\n" + "=" * 60)
    print("优化方向分析")
    print("=" * 60)

    print(f"\n当前 α = {alpha_test:.3f}")
    print(f"真实 α = 0.717")
    print(f"∂L/∂α (数值) = {alpha_grad_numerical:.6e}")

    if alpha_grad_numerical > 0:
        print("\n∂L/∂α > 0: 损失随 α 增大而增大")
        print("优化器应该: 减小 α (朝向 0.717) ✓")
        print("梯度下降: α_new = α - lr * ∂L/∂α (减小 α)")
    else:
        print("\n∂L/∂α < 0: 损失随 α 增大而减小")
        print("优化器应该: 增大 α (远离 0.717) ✗")
        print("梯度下降: α_new = α - lr * ∂L/∂α (增大 α)")

    print("\n实际训练结果: α 从 0.800 增大到 0.837")
    if alpha_grad_numerical < 0:
        print("这与数值梯度的方向一致")
        print("说明自动微分的梯度符号错误！")

if __name__ == "__main__":
    verify_alpha_gradient_v2()
