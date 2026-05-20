#!/usr/bin/env python3
"""
Legacy verification script for the old PINN prototype.

当前默认主线请改看 `cases/self-discovery/canonical/README.md`。

验证正问题：固定 alpha=0.717，测试网络能否求解 Guderley ODE
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append('/share/project/zpy/PINN_WE/cases/self-discovery/src')
from model import GuderleyPINN
from physics import compute_total_loss


class ForwardProblemVerifier:
    """正问题验证器"""

    def __init__(self, alpha_fixed: float = 0.717):
        """
        初始化验证器

        Args:
            alpha_fixed: 固定的 alpha 值
        """
        self.alpha_fixed = alpha_fixed

        # 创建模型，alpha 初始化为正确值
        self.model = GuderleyPINN(
            hidden_layers=4,
            hidden_neurons=64,
            alpha_init=alpha_fixed,
            alpha_min=alpha_fixed - 0.001,  # 限制在很小的范围内
            alpha_max=alpha_fixed + 0.001
        )

        print(f"模型初始 α: {self.model.get_alpha_value():.6f}")
        print(f"固定 α 值: {alpha_fixed:.6f}")

        self.history = {'epoch': [], 'loss': [], 'loss_pde': [], 'loss_bc': []}

    def train(self, epochs: int = 3000, lr: float = 1e-3, print_every: int = 300):
        """训练网络求解正问题"""
        print("\n" + "=" * 60)
        print("正问题验证：固定 α=0.717，训练网络求解 ODE")
        print("=" * 60)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # 生成采样点
        xi_interior = torch.linspace(0.1, 0.99, 1000).reshape(-1, 1)
        xi_boundary = torch.ones(10, 1)

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            # 计算损失
            total_loss, loss_dict = compute_total_loss(
                self.model, xi_interior, xi_boundary,
                gamma=1.4, n=2, weight_pde=1.0, weight_bc=10.0
            )

            total_loss.backward()
            optimizer.step()

            # 记录历史
            self.history['epoch'].append(epoch)
            self.history['loss'].append(loss_dict['total'])
            self.history['loss_pde'].append(loss_dict['pde'])
            self.history['loss_bc'].append(loss_dict['bc'])

            # 打印进度
            if epoch % print_every == 0 or epoch == 1:
                alpha_current = self.model.get_alpha_value()
                print(f"Epoch {epoch:5d} | Loss: {loss_dict['total']:.6e} | "
                      f"α: {alpha_current:.6f} | "
                      f"PDE: {loss_dict['pde']:.6e} | BC: {loss_dict['bc']:.6e}")

        print(f"\n训练完成，最终 Loss: {self.history['loss'][-1]:.6e}")

    def plot_results(self, output_dir: str = 'output/forward_verification'):
        """绘制结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. 损失收敛曲线
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax = axes[0]
        ax.semilogy(self.history['epoch'], self.history['loss'], 'b-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Loss Convergence (Forward Problem)')
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.semilogy(self.history['epoch'], self.history['loss_pde'], 'r-', linewidth=2, label='PDE')
        ax.semilogy(self.history['epoch'], self.history['loss_bc'], 'g-', linewidth=2, label='BC')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('PDE vs BC Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'forward_loss.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n结果已保存到: {output_path}")
        print(f"  - 损失曲线: forward_loss.png")

        # 2. 解的分布
        xi_test = torch.linspace(0.1, 1.0, 200).reshape(-1, 1)
        with torch.no_grad():
            V, C, G = self.model(xi_test)

        xi_np = xi_test.numpy().flatten()
        V_np = V.numpy().flatten()
        C_np = C.numpy().flatten()
        G_np = G.numpy().flatten()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(xi_np, V_np, 'b-', linewidth=2)
        axes[0].set_xlabel('ξ')
        axes[0].set_ylabel('V')
        axes[0].set_title('Velocity Profile')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(xi_np, C_np, 'r-', linewidth=2)
        axes[1].set_xlabel('ξ')
        axes[1].set_ylabel('C')
        axes[1].set_title('Sound Speed Profile')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(xi_np, G_np, 'g-', linewidth=2)
        axes[2].set_xlabel('ξ')
        axes[2].set_ylabel('G')
        axes[2].set_title('Density Profile')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'forward_solution.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  - 解分布图: forward_solution.png")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Guderley 正问题验证")
    print("目标：固定 α=0.717，测试网络能否求解 ODE")
    print("=" * 60)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 创建验证器
    verifier = ForwardProblemVerifier(alpha_fixed=0.717)

    # 训练
    verifier.train(epochs=3000, lr=1e-3, print_every=300)

    # 绘制结果
    verifier.plot_results()

    # 判断是否成功
    final_loss = verifier.history['loss'][-1]
    print("\n" + "=" * 60)
    print("验证结果:")
    print("=" * 60)
    print(f"最终 Loss: {final_loss:.6e}")

    if final_loss < 1e-2:
        print("✓ 正问题验证成功！网络能够求解 Guderley ODE")
        print("  说明方程实现基本正确")
    elif final_loss < 1e-1:
        print("⚠ 正问题部分成功，但精度不够高")
        print("  可能需要调整网络结构或训练参数")
    else:
        print("✗ 正问题验证失败！")
        print("  说明 Guderley 方程实现可能有问题")

    print("\n下一步建议:")
    if final_loss < 1e-2:
        print("  1. 检查为什么反问题中 α 往错误方向走")
        print("  2. 可能需要调整损失函数权重")
        print("  3. 或者在奇异点附近加密采样")
    else:
        print("  1. 检查 Guderley 方程的公式是否正确")
        print("  2. 查阅文献确认 N_V 和 N_C 的完整表达式")
        print("  3. 检查边界条件是否正确")


if __name__ == "__main__":
    main()

