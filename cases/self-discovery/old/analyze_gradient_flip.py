#!/usr/bin/env python3
"""
分析训练过程中梯度符号翻转的原因
"""

import sys
sys.path.append('/share/project/zpy/PINN_WE/cases/self-discovery/src')

import torch
import numpy as np
from model import GuderleyPINN
from physics import compute_total_loss

def analyze_gradient_flip():
    """分析梯度符号翻转"""

    print("=" * 60)
    print("分析训练过程中梯度符号翻转的原因")
    print("=" * 60)

    # 创建训练点
    xi_interior = torch.linspace(0.1, 1.0, 100).reshape(-1, 1)
    xi_boundary = torch.tensor([[1.0]])
    gamma = 1.4
    n = 2

    # 测试不同的α值
    alpha_values = [0.717, 0.75, 0.80, 0.85]

    print("\n测试：未训练的网络（随机初始化）")
    print("-" * 60)

    for alpha_val in alpha_values:
        model = GuderleyPINN(alpha_init=alpha_val)
        model.train()

        # 计算损失
        total_loss, loss_dict = compute_total_loss(
            model, xi_interior, xi_boundary, gamma, n
        )

        # 计算梯度
        total_loss.backward()
        grad_raw_alpha = model.raw_alpha.grad.item()

        # 计算对α的隐含梯度
        sigmoid_val = torch.sigmoid(model.raw_alpha).item()
        sigmoid_derivative = sigmoid_val * (1 - sigmoid_val)
        dalpha_draw = 0.5 * sigmoid_derivative
        grad_alpha = grad_raw_alpha / dalpha_draw

        print(f"\nα = {alpha_val:.3f}:")
        print(f"  Loss: {loss_dict['total']:.6e}")
        print(f"  ∂L/∂α: {grad_alpha:.6e}")
        print(f"  方向: {'减小α' if grad_alpha > 0 else '增大α'}")

    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)

    print("\n观察到的现象：")
    print("1. 训练初期（Epoch 1）：∂L/∂α = +1.554e-04 (正值)")
    print("2. 训练后期（Epoch 500+）：∂L/∂α 变为负值")
    print("3. α 从 0.800 增大到 0.837")

    print("\n可能的原因：")
    print("1. 网络权重随机初始化，导致初始损失函数形态不稳定")
    print("2. 当网络开始拟合PDE时，损失函数形态发生变化")
    print("3. 梯度符号翻转，导致优化方向错误")

    print("\n解决方案：")
    print("1. 使用更好的网络初始化")
    print("2. 先固定α训练网络，再联合优化")
    print("3. 使用更大的α学习率")
    print("4. 添加α的正则化项，引导其朝向真实值")

if __name__ == "__main__":
    analyze_gradient_flip()
