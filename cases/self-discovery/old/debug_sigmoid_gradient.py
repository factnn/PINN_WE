#!/usr/bin/env python3
"""
调试 sigmoid 参数化对梯度的影响
"""

import sys
sys.path.append('/share/project/zpy/PINN_WE/cases/self-discovery/src')

import torch
import numpy as np
from model import GuderleyPINN
from physics import compute_total_loss

def debug_sigmoid_gradient():
    """调试 sigmoid 参数化"""

    print("=" * 60)
    print("调试 sigmoid 参数化对梯度的影响")
    print("=" * 60)

    # 创建模型
    alpha_test = 0.80
    model = GuderleyPINN(alpha_init=alpha_test)
    model.train()

    # 创建训练点
    xi_interior = torch.linspace(0.1, 1.0, 100).reshape(-1, 1)
    xi_boundary = torch.tensor([[1.0]])

    gamma = 1.4
    n = 2

    print(f"\n初始状态:")
    print(f"  α = {model.get_alpha_value():.6f}")
    print(f"  raw_alpha = {model.raw_alpha.item():.6f}")

    # 计算损失和梯度
    total_loss, loss_dict = compute_total_loss(
        model, xi_interior, xi_boundary, gamma, n
    )
    total_loss.backward()

    # 获取梯度
    grad_raw_alpha = model.raw_alpha.grad.item()

    print(f"\n梯度信息:")
    print(f"  ∂L/∂(raw_alpha) = {grad_raw_alpha:.6e}")

    # 计算 sigmoid 导数
    sigmoid_val = torch.sigmoid(model.raw_alpha).item()
    sigmoid_derivative = sigmoid_val * (1 - sigmoid_val)

    print(f"\nsigmoid 变换:")
    print(f"  sigmoid(raw_alpha) = {sigmoid_val:.6f}")
    print(f"  d(sigmoid)/d(raw_alpha) = {sigmoid_derivative:.6f}")

    # 计算对 α 的梯度
    # α = alpha_min + (alpha_max - alpha_min) * sigmoid(raw_alpha)
    # dα/d(raw_alpha) = (alpha_max - alpha_min) * sigmoid' = 0.5 * sigmoid'
    dalpha_draw = 0.5 * sigmoid_derivative

    print(f"  dα/d(raw_alpha) = {dalpha_draw:.6f}")

    # 链式法则: ∂L/∂α = ∂L/∂(raw_alpha) / [dα/d(raw_alpha)]
    grad_alpha_implied = grad_raw_alpha / dalpha_draw

    print(f"\n隐含的 ∂L/∂α:")
    print(f"  ∂L/∂α = ∂L/∂(raw_alpha) / [dα/d(raw_alpha)]")
    print(f"  ∂L/∂α = {grad_raw_alpha:.6e} / {dalpha_draw:.6f}")
    print(f"  ∂L/∂α = {grad_alpha_implied:.6e}")

    print("\n" + "=" * 60)
    print("数值验证")
    print("=" * 60)

    # 数值验证：直接扰动 α
    epsilon = 1e-4

    # 保存原始值
    raw_alpha_original = model.raw_alpha.data.clone()

    # 计算当前损失
    loss_center, _ = compute_total_loss(model, xi_interior, xi_boundary, gamma, n)
    loss_center_val = loss_center.item()

    # 计算 α ± ε 对应的 raw_alpha 值
    # α = 0.5 + 0.5 * sigmoid(raw_alpha)
    # sigmoid(raw_alpha) = (α - 0.5) / 0.5 = 2α - 1
    # raw_alpha = logit(2α - 1)

    def alpha_to_raw(alpha_val):
        sigmoid_val = (alpha_val - 0.5) / 0.5
        sigmoid_val = np.clip(sigmoid_val, 0.01, 0.99)
        return np.log(sigmoid_val / (1 - sigmoid_val))

    raw_alpha_plus = alpha_to_raw(alpha_test + epsilon)
    raw_alpha_minus = alpha_to_raw(alpha_test - epsilon)

    # α + ε
    model.raw_alpha.data = torch.tensor(raw_alpha_plus)
    loss_plus, _ = compute_total_loss(model, xi_interior, xi_boundary, gamma, n)
    loss_plus_val = loss_plus.item()

    # α - ε
    model.raw_alpha.data = torch.tensor(raw_alpha_minus)
    loss_minus, _ = compute_total_loss(model, xi_interior, xi_boundary, gamma, n)
    loss_minus_val = loss_minus.item()

    # 恢复
    model.raw_alpha.data = raw_alpha_original

    # 数值梯度
    grad_alpha_numerical = (loss_plus_val - loss_minus_val) / (2 * epsilon)

    print(f"\n直接扰动 α 的数值梯度:")
    print(f"  ∂L/∂α (数值) = {grad_alpha_numerical:.6e}")
    print(f"  ∂L/∂α (隐含) = {grad_alpha_implied:.6e}")
    print(f"  相对误差: {abs(grad_alpha_numerical - grad_alpha_implied) / (abs(grad_alpha_numerical) + 1e-10):.2%}")

    print("\n" + "=" * 60)
    print("优化方向分析")
    print("=" * 60)

    print(f"\n当前 α = {alpha_test:.3f}, 真实 α = 0.717")
    print(f"∂L/∂α = {grad_alpha_numerical:.6e}")

    if grad_alpha_numerical > 0:
        print("\n∂L/∂α > 0: 应该减小 α")
        print("梯度下降更新: α_new = α - lr * ∂L/∂α")
    else:
        print("\n∂L/∂α < 0: 应该增大 α")
        print("梯度下降更新: α_new = α - lr * ∂L/∂α")

    print(f"\n但实际更新的是 raw_alpha:")
    print(f"  raw_alpha_new = raw_alpha - lr * ∂L/∂(raw_alpha)")
    print(f"  ∂L/∂(raw_alpha) = {grad_raw_alpha:.6e}")

    if grad_raw_alpha > 0:
        print(f"  raw_alpha 会减小")
    else:
        print(f"  raw_alpha 会增大")

    # 模拟多个学习率的更新
    print(f"\n模拟不同学习率的更新:")

    for lr in [0.001, 0.01, 0.1, 1.0]:
        raw_alpha_new = model.raw_alpha.item() - lr * grad_raw_alpha

        # 计算新的 α
        sigmoid_new = 1 / (1 + np.exp(-raw_alpha_new))
        alpha_new = 0.5 + 0.5 * sigmoid_new

        delta_alpha = alpha_new - alpha_test
        direction = "减小 ✓" if delta_alpha < 0 else "增大 ✗"

        print(f"\n  lr = {lr}:")
        print(f"    raw_alpha: {model.raw_alpha.item():.6f} → {raw_alpha_new:.6f}")
        print(f"    α: {alpha_test:.6f} → {alpha_new:.6f}")
        print(f"    Δα = {delta_alpha:+.6f} ({direction})")

if __name__ == "__main__":
    debug_sigmoid_gradient()
