"""
Guderley Self-Discovery Training Script
训练循环：自动发现 alpha 参数
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List
import sys

sys.path.append('/share/project/zpy/PINN_WE/cases/self-discovery/src')
from model import GuderleyPINN
from physics import compute_total_loss


class GuderleyTrainer:
    """Guderley 自发现训练器"""

    def __init__(
        self,
        model: GuderleyPINN,
        gamma: float = 1.4,
        n: int = 2,
        n_interior: int = 1000,
        n_boundary: int = 10,
        weight_pde: float = 1.0,
        weight_bc: float = 10.0,
        device: str = 'cpu'
    ):
        """
        初始化训练器

        Args:
            model: GuderleyPINN 模型
            gamma: 比热比
            n: 对称性参数 (1=圆柱, 2=球形)
            n_interior: 内部采样点数量
            n_boundary: 边界采样点数量
            weight_pde: PDE 残差权重
            weight_bc: 边界条件权重
            device: 计算设备
        """
        self.model = model.to(device)
        self.gamma = gamma
        self.n = n
        self.n_interior = n_interior
        self.n_boundary = n_boundary
        self.weight_pde = weight_pde
        self.weight_bc = weight_bc
        self.device = device

        # 历史记录
        self.history = {
            'epoch': [],
            'loss': [],
            'alpha': [],
            'loss_pde': [],
            'loss_bc': []
        }

    def generate_training_points(self) -> tuple:
        """生成训练采样点"""
        # 【扩大采样范围】从 [0.4, 1.0] 改为 [0.1, 1.0]
        # 增加靠近球心的采样点，捕捉内部结构信息
        xi_interior = torch.linspace(0.1, 1.0, self.n_interior).reshape(-1, 1)

        # 边界点：只取一个 ξ=1（避免重复）
        xi_boundary = torch.tensor([[1.0]])

        return xi_interior.to(self.device), xi_boundary.to(self.device)

    def train_adam(
        self,
        epochs: int = 5000,
        lr: float = 1e-3,
        print_every: int = 100
    ):
        """
        使用 Adam 优化器训练

        Args:
            epochs: 训练轮数
            lr: 学习率
            print_every: 打印间隔
        """
        print("\n" + "=" * 60)
        print("阶段 1: Adam 优化器训练")
        print("=" * 60)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        xi_interior, xi_boundary = self.generate_training_points()

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            # 计算损失
            total_loss, loss_dict = compute_total_loss(
                self.model, xi_interior, xi_boundary,
                self.gamma, self.n,
                self.weight_pde, self.weight_bc
            )

            # 反向传播
            total_loss.backward()

            # 【诊断】获取 α 的梯度
            alpha_grad = self.model.raw_alpha.grad.item() if self.model.raw_alpha.grad is not None else 0.0

            optimizer.step()

            # 记录历史
            alpha_current = self.model.get_alpha_value()
            self.history['epoch'].append(epoch)
            self.history['loss'].append(loss_dict['total'])
            self.history['alpha'].append(alpha_current)
            self.history['loss_pde'].append(loss_dict['pde'])
            self.history['loss_bc'].append(loss_dict['bc'])

            # 打印进度
            if epoch % print_every == 0 or epoch == 1:
                print(f"Epoch {epoch:5d} | Loss: {loss_dict['total']:.6e} | "
                      f"α: {alpha_current:.6f} | "
                      f"PDE: {loss_dict['pde']:.6e} | BC: {loss_dict['bc']:.6e} | "
                      f"∂L/∂α: {alpha_grad:.6e}")

        print(f"\nAdam 训练完成，最终 α = {self.model.get_alpha_value():.6f}")

    def train_lbfgs(
        self,
        max_iter: int = 1000,
        lr: float = 1.0,
        print_every: int = 50
    ):
        """
        使用 L-BFGS 优化器精调

        Args:
            max_iter: 最大迭代次数
            lr: 学习率
            print_every: 打印间隔
        """
        print("\n" + "=" * 60)
        print("阶段 2: L-BFGS 优化器精调")
        print("=" * 60)

        optimizer = optim.LBFGS(
            self.model.parameters(),
            lr=lr,
            max_iter=20,
            history_size=50,
            line_search_fn='strong_wolfe'
        )

        xi_interior, xi_boundary = self.generate_training_points()

        iteration = [0]  # 使用列表来在闭包中修改

        def closure():
            optimizer.zero_grad()
            total_loss, loss_dict = compute_total_loss(
                self.model, xi_interior, xi_boundary,
                self.gamma, self.n,
                self.weight_pde, self.weight_bc
            )
            total_loss.backward()

            # 【诊断】获取 α 的梯度
            alpha_grad = self.model.raw_alpha.grad.item() if self.model.raw_alpha.grad is not None else 0.0

            # 记录历史
            iteration[0] += 1
            alpha_current = self.model.get_alpha_value()
            epoch_num = self.history['epoch'][-1] + iteration[0] if self.history['epoch'] else iteration[0]

            self.history['epoch'].append(epoch_num)
            self.history['loss'].append(loss_dict['total'])
            self.history['alpha'].append(alpha_current)
            self.history['loss_pde'].append(loss_dict['pde'])
            self.history['loss_bc'].append(loss_dict['bc'])

            # 打印进度
            if iteration[0] % print_every == 0 or iteration[0] == 1:
                print(f"Iter {iteration[0]:5d} | Loss: {loss_dict['total']:.6e} | "
                      f"α: {alpha_current:.6f} | "
                      f"PDE: {loss_dict['pde']:.6e} | BC: {loss_dict['bc']:.6e} | "
                      f"∂L/∂α: {alpha_grad:.6e}")

            return total_loss

        # 执行优化
        for _ in range(max_iter // 20):
            optimizer.step(closure)
            if iteration[0] >= max_iter:
                break

        print(f"\nL-BFGS 训练完成，最终 α = {self.model.get_alpha_value():.6f}")

    def save_results(self, output_dir: str):
        """
        保存训练结果

        Args:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存模型
        torch.save(self.model.state_dict(), output_path / 'model.pth')

        # 保存历史记录
        history_dict = {k: [float(v) if isinstance(v, (np.floating, torch.Tensor)) else v
                           for v in vals] for k, vals in self.history.items()}
        with open(output_path / 'history.json', 'w') as f:
            json.dump(history_dict, f, indent=2)

        # 保存最终结果
        results = {
            'alpha_final': self.model.get_alpha_value(),
            'alpha_true': 0.717,
            'error_percent': abs(self.model.get_alpha_value() - 0.717) / 0.717 * 100,
            'final_loss': self.history['loss'][-1],
            'gamma': self.gamma,
            'n': self.n
        }
        with open(output_path / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n结果已保存到: {output_path}")
        print(f"  - 模型: model.pth")
        print(f"  - 历史: history.json")
        print(f"  - 结果: results.json")

    def plot_convergence(self, output_dir: str):
        """
        绘制收敛历史图

        Args:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Alpha 收敛历史
        ax = axes[0, 0]
        ax.plot(self.history['epoch'], self.history['alpha'], 'b-', linewidth=2)
        ax.axhline(y=0.717, color='r', linestyle='--', linewidth=2, label='True α = 0.717')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('α')
        ax.set_title('Alpha Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 总损失历史
        ax = axes[0, 1]
        ax.semilogy(self.history['epoch'], self.history['loss'], 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Loss Convergence')
        ax.grid(True, alpha=0.3)

        # 3. PDE 损失历史
        ax = axes[1, 0]
        ax.semilogy(self.history['epoch'], self.history['loss_pde'], 'orange', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('PDE Loss')
        ax.set_title('PDE Residual')
        ax.grid(True, alpha=0.3)

        # 4. BC 损失历史
        ax = axes[1, 1]
        ax.semilogy(self.history['epoch'], self.history['loss_bc'], 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('BC Loss')
        ax.set_title('Boundary Condition Loss')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'convergence.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  - 收敛图: convergence.png")

    def plot_solution(self, output_dir: str):
        """
        绘制最终解的分布

        Args:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 生成密集的测试点
        xi_test = torch.linspace(0.1, 1.0, 200).reshape(-1, 1).to(self.device)

        with torch.no_grad():
            V, C, G = self.model(xi_test)

        xi_np = xi_test.cpu().numpy().flatten()
        V_np = V.cpu().numpy().flatten()
        C_np = C.cpu().numpy().flatten()
        G_np = G.cpu().numpy().flatten()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # V(ξ)
        axes[0].plot(xi_np, V_np, 'b-', linewidth=2)
        axes[0].set_xlabel('ξ')
        axes[0].set_ylabel('V (Velocity)')
        axes[0].set_title('Velocity Profile')
        axes[0].grid(True, alpha=0.3)

        # C(ξ)
        axes[1].plot(xi_np, C_np, 'r-', linewidth=2)
        axes[1].set_xlabel('ξ')
        axes[1].set_ylabel('C (Sound Speed)')
        axes[1].set_title('Sound Speed Profile')
        axes[1].grid(True, alpha=0.3)

        # G(ξ)
        axes[2].plot(xi_np, G_np, 'g-', linewidth=2)
        axes[2].set_xlabel('ξ')
        axes[2].set_ylabel('G (Density)')
        axes[2].set_title('Density Profile')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'solution.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  - 解分布图: solution.png")


def main():
    """主训练流程"""
    print("\n" + "=" * 60)
    print("Guderley 自相似指数自发现实验")
    print("=" * 60)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 创建模型
    print("\n创建模型...")
    model = GuderleyPINN(
        hidden_layers=4,
        hidden_neurons=64,
        alpha_init=0.8,  # 【测试修复后的方程】从错误值开始，看能否收敛到 0.717
        alpha_min=0.5,
        alpha_max=1.0
    )

    print(f"初始 α 值: {model.get_alpha_value():.6f}")
    print(f"真实 α 值: 0.717")
    print(f"初始误差: {abs(model.get_alpha_value() - 0.717) / 0.717 * 100:.2f}%")

    # 创建训练器
    trainer = GuderleyTrainer(
        model=model,
        gamma=1.4,
        n=2,  # 球形对称
        n_interior=1000,
        n_boundary=10,
        weight_pde=1.0,  # 【硬约束】只优化 PDE，权重设为 1.0
        weight_bc=0.0,   # 【硬约束】BC 不参与优化，仅用于监控
        device='cpu'
    )

    # 阶段 1: Adam 优化
    trainer.train_adam(epochs=5000, lr=1e-3, print_every=500)

    # 阶段 2: L-BFGS 精调
    trainer.train_lbfgs(max_iter=500, lr=1.0, print_every=50)

    # 输出最终结果
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    alpha_final = model.get_alpha_value()
    alpha_true = 0.717
    error = abs(alpha_final - alpha_true) / alpha_true * 100

    print(f"\n最终结果:")
    print(f"  发现的 α: {alpha_final:.6f}")
    print(f"  真实的 α: {alpha_true:.6f}")
    print(f"  相对误差: {error:.4f}%")

    # 保存结果
    output_dir = '/share/project/zpy/PINN_WE/cases/self-discovery/output'
    print(f"\n保存结果到: {output_dir}")
    trainer.save_results(output_dir)
    trainer.plot_convergence(output_dir)
    trainer.plot_solution(output_dir)

    print("\n✓ 所有任务完成！")


if __name__ == "__main__":
    main()


