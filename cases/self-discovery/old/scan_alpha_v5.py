#!/usr/bin/env python3
"""
Legacy alpha scan for the old self-discovery branch.

当前默认主线请改看 `cases/self-discovery/canonical/README.md`。

Guderley 漏斗扫描实验 V5 (软约束版本)

修正内容:
1. 移除硬约束，网络直接输出
2. 使用大权重的软约束强制BC
3. 测试不同的BC权重

目标：验证软约束下损失函数是否在α=0.717处最小
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


class SoftConstraintScanV5:
    """软约束扫描实验"""

    def __init__(
        self,
        alpha_values: list,
        weight_configs: list,  # [(weight_pde, weight_bc_shock, weight_bc_center), ...]
        gamma: float = 1.4,
        n: int = 2,
        n_interior: int = 200,
        mach_inf: float = 10.0,
        output_dir: str = "output_scan_v5"
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

        # 创建新模型
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
        """运行软约束扫描实验"""

        print("=" * 60)
        print("Guderley 软约束扫描实验 V5")
        print("=" * 60)
        print(f"α候选值: {self.alpha_values}")
        print(f"权重配置数: {len(self.weight_configs)}")
        print(f"训练策略: Adam(5000) + L-BFGS(2000)")

        for config_idx, (w_pde, w_bc_shock, w_bc_center) in enumerate(self.weight_configs, 1):
            print(f"\n\n{'='*60}")
            print(f"[配置 {config_idx}/{len(self.weight_configs)}]")
            print(f"weight_pde={w_pde}, weight_bc_shock={w_bc_shock}, weight_bc_center={w_bc_center}")
            print(f"{'='*60}")

            results = []

            for i, alpha in enumerate(self.alpha_values, 1):
                print(f"\n[{i}/{len(self.alpha_values)}] α = {alpha:.3f}")

                final_loss = self.train_fixed_alpha(
                    alpha, w_pde, w_bc_shock, w_bc_center,
                    adam_epochs=5000, lbfgs_epochs=2000, print_every=1000
                )

                results.append({
                    'alpha': alpha,
                    'final_loss': final_loss
                })

                print(f"  ✓ Loss = {final_loss:.6e}")

            # 保存这个配置的结果
            config_key = f"pde{w_pde}_shock{w_bc_shock}_center{w_bc_center}"
            self.all_results[config_key] = results

            # 分析这个配置
            self.analyze_config(config_key, results, w_pde, w_bc_shock, w_bc_center)

        # 保存所有结果
        self.save_all_results()
        self.plot_all_configs()

    def analyze_config(self, config_key, results, w_pde, w_bc_shock, w_bc_center):
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
            print("✓ V型谷底 - 成功！")
        else:
            print(f"✗ 滑梯型 - 最小值在α={min_alpha:.3f}")

    def save_all_results(self):
        """保存所有配置的结果"""
        results_file = self.output_dir / "soft_constraint_results.json"

        with open(results_file, 'w') as f:
            json.dump({
                'alpha_values': self.alpha_values,
                'weight_configs': self.weight_configs,
                'all_results': self.all_results,
                'gamma': self.gamma,
                'n': self.n,
                'mach_inf': self.mach_inf,
                'version': 'v5_soft_constraint'
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
            axes[idx].set_title(f'{config_key}', fontsize=10)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = self.output_dir / "soft_constraint_comparison.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✓ 对比图已保存: {plot_file}")
        plt.close()


def main():
    """主函数"""

    # 定义α扫描列表
    alpha_values = [0.60, 0.68, 0.717, 0.75, 0.80, 0.90, 1.00]

    # 定义权重配置 (weight_pde, weight_bc_shock, weight_bc_center)
    # 软约束下，BC权重需要很大才能强制满足边界条件
    weight_configs = [
        (1.0, 100.0, 10.0),   # 配置1: BC权重100
        (1.0, 500.0, 50.0),   # 配置2: BC权重500
        (1.0, 1000.0, 100.0), # 配置3: BC权重1000
    ]

    print("=" * 60)
    print("软约束扫描实验配置")
    print("=" * 60)
    print(f"α测试值: {alpha_values}")
    print(f"\n权重配置:")
    for i, (w_pde, w_shock, w_center) in enumerate(weight_configs, 1):
        print(f"  配置{i}: pde={w_pde}, bc_shock={w_shock}, bc_center={w_center}")
    print(f"\n对比: V3硬约束版本 BC损失≈0")

    # 创建实验
    experiment = SoftConstraintScanV5(
        alpha_values=alpha_values,
        weight_configs=weight_configs,
        gamma=1.4,
        n=2,
        n_interior=200,
        mach_inf=10.0,
        output_dir="output_scan_v5"
    )

    # 运行扫描
    experiment.run_scan()

    print("\n" + "=" * 60)
    print("软约束扫描实验V5完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()






