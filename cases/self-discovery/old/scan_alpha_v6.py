#!/usr/bin/env python3
"""
Legacy alpha scan for the old self-discovery branch.

当前默认主线请改看 `cases/self-discovery/canonical/README.md`。

Guderley 漏斗扫描实验 V6 (极大BC权重测试)

基于V5的发现：BC权重越大，最小值越靠近α=0.717
- V5配置1 (BC=100):  最小值在 α=1.0
- V5配置2 (BC=500):  最小值在 α=0.9
- V5配置3 (BC=1000): 最小值在 α=0.8

V6目标：测试极大BC权重，看能否把最小值推到α=0.717
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


class ExtremeBCWeightScanV6:
    """极大BC权重扫描实验"""

    def __init__(
        self,
        alpha_values: list,
        weight_configs: list,
        gamma: float = 1.4,
        n: int = 2,
        n_interior: int = 200,
        mach_inf: float = 10.0,
        output_dir: str = "output_scan_v6"
    ):
        self.alpha_values = alpha_values
        self.weight_configs = weight_configs
        self.gamma = gamma
        self.n = n
        self.n_interior = n_interior
        self.mach_inf = mach_inf
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # GPU设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # 创建训练点
        self.xi_interior = torch.linspace(0.01, 0.9, n_interior).reshape(-1, 1).to(self.device)
        self.xi_boundary = torch.tensor([[1.0]]).to(self.device)
        self.xi_center = torch.tensor([[1e-3]]).to(self.device)

        # 存储结果
        self.all_results = {}

    def train_fixed_alpha(
        self,
        alpha_fixed: float,
        weight_pde: float,
        weight_bc_shock: float,
        weight_bc_center: float,
        adam_epochs: int = 5000,
        lbfgs_epochs: int = 2000,
        print_every: int = 1000
    ) -> float:
        """固定α值和权重，训练网络"""

        model = GuderleyPINN(alpha_init=alpha_fixed)
        model.to(self.device)
        model.train()
        model.raw_alpha.requires_grad = False

        # Adam优化
        optimizer_adam = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3
        )

        for epoch in range(1, adam_epochs + 1):
            optimizer_adam.zero_grad()

            total_loss, loss_dict = compute_total_loss(
                model,
                self.xi_interior,
                self.xi_boundary,
                self.xi_center,
                self.gamma,
                self.n,
                self.mach_inf,
                weight_pde=weight_pde,
                weight_bc_shock=weight_bc_shock,
                weight_bc_center=weight_bc_center
            )

            total_loss.backward()
            optimizer_adam.step()

            if epoch % print_every == 0 or epoch == 1:
                print(f"  Epoch {epoch:4d} | Loss: {loss_dict['total']:.6e}")

        # L-BFGS优化
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
            total_loss, _ = compute_total_loss(
                model,
                self.xi_interior,
                self.xi_boundary,
                self.xi_center,
                self.gamma,
                self.n,
                self.mach_inf,
                weight_pde=weight_pde,
                weight_bc_shock=weight_bc_shock,
                weight_bc_center=weight_bc_center
            )
            total_loss.backward()
            iteration[0] += 1
            return total_loss

        for _ in range(lbfgs_epochs // 20):
            optimizer_lbfgs.step(closure)
            if iteration[0] >= lbfgs_epochs:
                break

        # 计算最终损失
        final_loss, _ = compute_total_loss(
            model,
            self.xi_interior,
            self.xi_boundary,
            self.xi_center,
            self.gamma,
            self.n,
            self.mach_inf,
            weight_pde=weight_pde,
            weight_bc_shock=weight_bc_shock,
            weight_bc_center=weight_bc_center
        )

        return final_loss.item()

    def run_scan(self):
        """运行极大BC权重扫描实验"""

        print("=" * 60)
        print("Guderley 极大BC权重扫描实验 V6")
        print("=" * 60)
        print(f"α候选值: {self.alpha_values}")
        print(f"权重配置数: {len(self.weight_configs)}")

        for config_idx, (w_pde, w_bc_shock, w_bc_center) in enumerate(self.weight_configs, 1):
            print(f"\n\n{'='*60}")
            print(f"[配置 {config_idx}/{len(self.weight_configs)}]")
            print(f"weight_pde={w_pde}, weight_bc_shock={w_bc_shock}, weight_bc_center={w_bc_center}")
            print(f"{'='*60}")

            results = []

            for i, alpha in enumerate(self.alpha_values, 1):
                print(f"\n[{i}/{len(self.alpha_values)}] α = {alpha:.3f}")

                final_loss = self.train_fixed_alpha(
                    alpha, w_pde, w_bc_shock, w_bc_center
                )

                results.append({
                    'alpha': alpha,
                    'final_loss': final_loss
                })

                print(f"  ✓ Loss = {final_loss:.6e}")

            config_key = f"pde{w_pde}_shock{w_bc_shock}_center{w_bc_center}"
            self.all_results[config_key] = results
            self.analyze_config(config_key, results)

        self.save_all_results()
        self.plot_all_configs()

    def analyze_config(self, config_key, results):
        """分析单个权重配置的结果"""

        print(f"\n{'='*60}")
        print(f"配置分析: {config_key}")
        print(f"{'='*60}")

        min_loss = min(r['final_loss'] for r in results)
        min_alpha = [r['alpha'] for r in results if r['final_loss'] == min_loss][0]

        loss_at_717 = None
        for r in results:
            if abs(r['alpha'] - 0.717) < 1e-6:
                loss_at_717 = r['final_loss']
                break

        print(f"\n最小Loss: {min_loss:.6e} (α={min_alpha:.3f})")
        if loss_at_717:
            print(f"α=0.717处Loss: {loss_at_717:.6e}")

        if abs(min_alpha - 0.717) < 0.05:
            print("✓✓✓ V型谷底 - 成功！")
        else:
            print(f"✗ 滑梯型 - 最小值在α={min_alpha:.3f}")

    def save_all_results(self):
        """保存所有配置的结果"""
        results_file = self.output_dir / "extreme_bc_results.json"

        with open(results_file, 'w') as f:
            json.dump({
                'alpha_values': self.alpha_values,
                'weight_configs': self.weight_configs,
                'all_results': self.all_results,
                'gamma': self.gamma,
                'n': self.n,
                'mach_inf': self.mach_inf,
                'version': 'v6_extreme_bc_weight'
            }, f, indent=2)

        print(f"\n✓ 所有结果已保存: {results_file}")

    def plot_all_configs(self):
        """绘制所有配置的对比图"""

        n_configs = len(self.all_results)
        fig, axes = plt.subplots(1, n_configs, figsize=(6*n_configs, 5))

        if n_configs == 1:
            axes = [axes]

        for idx, (config_key, results) in enumerate(self.all_results.items()):
            alphas = [r['alpha'] for r in results]
            losses = [r['final_loss'] for r in results]

            axes[idx].plot(alphas, losses, 'b-o', linewidth=2, markersize=8)
            axes[idx].axvline(x=0.717, color='r', linestyle='--', label='True alpha=0.717')
            axes[idx].set_xlabel('alpha', fontsize=12)
            axes[idx].set_ylabel('Final Loss', fontsize=12)
            axes[idx].set_title(f'{config_key}', fontsize=9)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = self.output_dir / "extreme_bc_comparison.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✓ 对比图已保存: {plot_file}")
        plt.close()


def main():
    """主函数"""

    # 定义α扫描列表
    alpha_values = [0.60, 0.68, 0.717, 0.75, 0.80, 0.90, 1.00]

    # 定义极大BC权重配置
    # 基于V5趋势：BC权重 100→500→1000，最小值从 α=1.0→0.9→0.8
    # V6测试：BC权重 2000, 5000, 10000
    weight_configs = [
        (1.0, 2000.0, 200.0),   # 配置1: BC权重2000
        (1.0, 5000.0, 500.0),   # 配置2: BC权重5000
        (1.0, 10000.0, 1000.0), # 配置3: BC权重10000
    ]

    print("=" * 60)
    print("极大BC权重扫描实验配置")
    print("=" * 60)
    print(f"α测试值: {alpha_values}")
    print(f"\n权重配置:")
    for i, (w_pde, w_shock, w_center) in enumerate(weight_configs, 1):
        print(f"  配置{i}: pde={w_pde}, bc_shock={w_shock}, bc_center={w_center}")

    print(f"\nV5趋势回顾:")
    print(f"  BC=100  → 最小值在 α=1.0")
    print(f"  BC=500  → 最小值在 α=0.9")
    print(f"  BC=1000 → 最小值在 α=0.8")
    print(f"\nV6目标: 继续增大BC权重，推动最小值到 α=0.717")

    # 创建实验
    experiment = ExtremeBCWeightScanV6(
        alpha_values=alpha_values,
        weight_configs=weight_configs,
        gamma=1.4,
        n=2,
        n_interior=200,
        mach_inf=10.0,
        output_dir="output_scan_v6"
    )

    # 运行扫描
    experiment.run_scan()

    print("\n" + "=" * 60)
    print("极大BC权重扫描实验V6完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()




