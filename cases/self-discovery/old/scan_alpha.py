#!/usr/bin/env python3
"""
Legacy alpha scan for the old self-discovery branch.

当前默认主线请改看 `cases/self-discovery/canonical/README.md`。

Guderley 漏斗扫描实验 (Funnel Plot Scan)

目标：通过网格搜索验证损失函数的真实形状
方法：固定不同的α值，充分训练网络，记录最终损失
"""

import sys
sys.path.append('/share/project/zpy/PINN_WE/cases/self-discovery/src')

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

from model import GuderleyPINN
from physics import compute_total_loss


class AlphaScanExperiment:
    """α扫描实验"""

    def __init__(
        self,
        alpha_values: list,
        gamma: float = 1.4,
        n: int = 2,
        n_interior: int = 200,
        output_dir: str = "output_scan"
    ):
        self.alpha_values = alpha_values
        self.gamma = gamma
        self.n = n
        self.n_interior = n_interior
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 创建训练点
        self.xi_interior = torch.linspace(0.1, 1.0, n_interior).reshape(-1, 1)
        self.xi_boundary = torch.tensor([[1.0]])

        # 存储结果
        self.results = []

    def train_fixed_alpha(
        self,
        alpha_fixed: float,
        adam_epochs: int = 5000,
        lbfgs_epochs: int = 2000,
        print_every: int = 1000
    ) -> float:
        """
        固定α值，充分训练网络

        Returns:
            final_loss: 最终损失值
        """
        print(f"\n{'='*60}")
        print(f"训练 α = {alpha_fixed:.6f}")
        print(f"{'='*60}")

        # 创建新模型（完全重新初始化）
        model = GuderleyPINN(alpha_init=alpha_fixed)
        model.train()

        # 冻结α参数
        model.raw_alpha.requires_grad = False

        # ========================================
        # 阶段1: Adam优化器
        # ========================================
        print(f"\n阶段1: Adam优化 ({adam_epochs}轮)")
        print("-" * 60)

        optimizer_adam = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3
        )

        for epoch in range(1, adam_epochs + 1):
            optimizer_adam.zero_grad()

            total_loss, loss_dict = compute_total_loss(
                model, self.xi_interior, self.xi_boundary,
                self.gamma, self.n
            )

            total_loss.backward()
            optimizer_adam.step()

            if epoch % print_every == 0 or epoch == 1:
                print(f"Epoch {epoch:5d} | Loss: {loss_dict['total']:.6e}")

        print(f"Adam完成，Loss: {loss_dict['total']:.6e}")

        # ========================================
        # 阶段2: L-BFGS优化器
        # ========================================
        print(f"\n阶段2: L-BFGS优化 ({lbfgs_epochs}轮)")
        print("-" * 60)

        optimizer_lbfgs = optim.LBFGS(
            [p for p in model.parameters() if p.requires_grad],
            lr=1.0,
            max_iter=20,
            history_size=50,
            line_search_fn='strong_wolfe'
        )

        iteration = [0]

        def closure():
            optimizer_lbfgs.zero_grad()
            total_loss, loss_dict = compute_total_loss(
                model, self.xi_interior, self.xi_boundary,
                self.gamma, self.n
            )
            total_loss.backward()

            iteration[0] += 1
            if iteration[0] % 100 == 0 or iteration[0] == 1:
                print(f"Iter {iteration[0]:5d} | Loss: {loss_dict['total']:.6e}")

            return total_loss

        for _ in range(lbfgs_epochs // 20):
            optimizer_lbfgs.step(closure)
            if iteration[0] >= lbfgs_epochs:
                break

        # 计算最终损失
        final_loss, _ = compute_total_loss(
            model, self.xi_interior, self.xi_boundary,
            self.gamma, self.n
        )

        print(f"\nL-BFGS完成，最终Loss: {final_loss.item():.6e}")

        return final_loss.item()

    def run_scan(self):
        """运行完整的α扫描实验"""

        print("=" * 60)
        print("Guderley 漏斗扫描实验")
        print("=" * 60)
        print(f"α候选值: {self.alpha_values}")
        print(f"真实α值: 0.717")
        print(f"训练策略: Adam(5000) + L-BFGS(2000)")

        for i, alpha in enumerate(self.alpha_values, 1):
            print(f"\n\n[{i}/{len(self.alpha_values)}] 测试 α = {alpha:.6f}")

            final_loss = self.train_fixed_alpha(alpha)

            self.results.append({
                'alpha': alpha,
                'final_loss': final_loss
            })

            print(f"\n✓ α = {alpha:.6f} 完成，最终Loss = {final_loss:.6e}")

        # 保存和可视化结果
        self.save_results()
        self.plot_results()
        self.analyze_results()

    def save_results(self):
        """保存结果到JSON"""
        results_file = self.output_dir / "scan_results.json"

        with open(results_file, 'w') as f:
            json.dump({
                'alpha_values': self.alpha_values,
                'results': self.results,
                'gamma': self.gamma,
                'n': self.n
            }, f, indent=2)

        print(f"\n✓ 结果已保存: {results_file}")

    def plot_results(self):
        """绘制漏斗图"""

        alphas = [r['alpha'] for r in self.results]
        losses = [r['final_loss'] for r in self.results]

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 图1: 线性尺度
        ax1.plot(alphas, losses, 'b-o', linewidth=2, markersize=8)
        ax1.axvline(x=0.717, color='r', linestyle='--', label='真实值 α=0.717')
        ax1.set_xlabel('α', fontsize=12)
        ax1.set_ylabel('Final Loss', fontsize=12)
        ax1.set_title('Loss vs α (线性尺度)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 图2: 对数尺度
        ax2.semilogy(alphas, losses, 'b-o', linewidth=2, markersize=8)
        ax2.axvline(x=0.717, color='r', linestyle='--', label='真实值 α=0.717')
        ax2.set_xlabel('α', fontsize=12)
        ax2.set_ylabel('Final Loss (log scale)', fontsize=12)
        ax2.set_title('Loss vs α (对数尺度)', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_file = self.output_dir / "funnel_plot.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✓ 漏斗图已保存: {plot_file}")
        plt.close()

    def analyze_results(self):
        """分析结果"""

        print("\n" + "=" * 60)
        print("扫描结果分析")
        print("=" * 60)

        # 打印表格
        print(f"\n{'α值':<12} {'最终Loss':<15} {'备注'}")
        print("-" * 60)

        min_loss = min(r['final_loss'] for r in self.results)
        min_alpha = [r['alpha'] for r in self.results if r['final_loss'] == min_loss][0]

        for r in self.results:
            alpha = r['alpha']
            loss = r['final_loss']

            note = ""
            if alpha == 0.717:
                note = "← 真实值"
            if loss == min_loss:
                note += " ★ 最小Loss"

            print(f"{alpha:<12.6f} {loss:<15.6e} {note}")

        # 分析结论
        print("\n" + "=" * 60)
        print("结论")
        print("=" * 60)

        print(f"\n最小Loss对应的α: {min_alpha:.6f}")
        print(f"真实α值: 0.717")
        print(f"最小Loss值: {min_loss:.6e}")

        # 找到α=0.717处的损失
        loss_at_true = None
        for r in self.results:
            if abs(r['alpha'] - 0.717) < 1e-6:
                loss_at_true = r['final_loss']
                break

        if loss_at_true is not None:
            print(f"α=0.717处的Loss: {loss_at_true:.6e}")

            if abs(min_alpha - 0.717) < 0.05:
                print("\n✓ V型谷底 (SUCCESS)")
                print("  - α=0.717附近的Loss确实最小")
                print("  - 物理方程正确")
                print("  - 之前失败是因为优化策略问题")
                print("  - 建议：使用贝叶斯优化或更好的初始化")
            else:
                print("\n✗ 滑梯型 (FAILURE)")
                print(f"  - 最小Loss在α={min_alpha:.3f}，不在0.717")
                print("  - 损失函数形态与真实物理不符")
                print("  - 可能原因：")
                print("    1. PDE方程形式有误")
                print("    2. 网络学到了平凡解")
                print("    3. 需要添加正则化约束")

        # 检查趋势
        print("\n损失函数趋势:")
        if self.results[-1]['final_loss'] < self.results[0]['final_loss']:
            print("  ✗ Loss随α增大而减小（滑梯型）")
        else:
            print("  ✓ Loss在中间某处有最小值（V型）")


def main():
    """主函数"""

    # 定义α扫描列表
    alpha_values = [0.60, 0.65, 0.68, 0.70, 0.717, 0.73, 0.75, 0.80, 0.90, 1.00]

    # 创建实验
    experiment = AlphaScanExperiment(
        alpha_values=alpha_values,
        gamma=1.4,
        n=2,
        n_interior=200,
        output_dir="output_scan"
    )

    # 运行扫描
    experiment.run_scan()

    print("\n" + "=" * 60)
    print("漏斗扫描实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
