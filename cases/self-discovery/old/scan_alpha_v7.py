#!/usr/bin/env python3
"""
Legacy alpha scan for the old self-discovery branch.

当前默认主线请改看 `cases/self-discovery/canonical/README.md`。

Guderley 漏斗扫描实验 V7
修正: 1.移除冗余G变量 2.n=3球面(对应α=0.717) 3.软约束
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


class AlphaScanV7:
    def __init__(self, alpha_values, gamma=1.4, n=3, n_interior=200,
                 mach_inf=10.0, output_dir="output_scan_v7"):
        self.alpha_values = alpha_values
        self.gamma = gamma
        self.n = n
        self.mach_inf = mach_inf
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"设备: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.xi_interior = torch.linspace(0.01, 0.9, n_interior).reshape(-1, 1).to(self.device)
        self.xi_boundary = torch.tensor([[1.0]]).to(self.device)
        self.xi_center = torch.tensor([[1e-3]]).to(self.device)
        self.results = []

    def train_fixed_alpha(self, alpha_fixed, adam_epochs=5000,
                          lbfgs_epochs=2000, print_every=1000,
                          weight_pde=1.0, weight_bc_shock=100.0,
                          weight_bc_center=10.0):
        """固定α值训练网络"""

        model = GuderleyPINN(alpha_init=alpha_fixed)
        model.to(self.device)
        model.train()
        model.raw_alpha.requires_grad = False

        optimizer_adam = optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )

        for epoch in range(1, adam_epochs + 1):
            optimizer_adam.zero_grad()
            total_loss, loss_dict = compute_total_loss(
                model, self.xi_interior, self.xi_boundary, self.xi_center,
                self.gamma, self.n, self.mach_inf,
                weight_pde=weight_pde,
                weight_bc_shock=weight_bc_shock,
                weight_bc_center=weight_bc_center
            )
            total_loss.backward()
            optimizer_adam.step()

            if epoch % print_every == 0 or epoch == 1:
                print(f"  Epoch {epoch:4d} | Total: {loss_dict['total']:.4e} "
                      f"PDE: {loss_dict['pde']:.4e} BC: {loss_dict['bc_shock']:.4e}")

        print(f"  Adam done, Loss: {loss_dict['total']:.6e}")

        # L-BFGS
        optimizer_lbfgs = optim.LBFGS(
            [p for p in model.parameters() if p.requires_grad],
            lr=1.0, max_iter=20, history_size=50,
            line_search_fn='strong_wolfe'
        )
        iteration = [0]

        def closure():
            optimizer_lbfgs.zero_grad()
            tl, _ = compute_total_loss(
                model, self.xi_interior, self.xi_boundary, self.xi_center,
                self.gamma, self.n, self.mach_inf,
                weight_pde=weight_pde,
                weight_bc_shock=weight_bc_shock,
                weight_bc_center=weight_bc_center
            )
            tl.backward()
            iteration[0] += 1
            return tl

        for _ in range(lbfgs_epochs // 20):
            optimizer_lbfgs.step(closure)
            if iteration[0] >= lbfgs_epochs:
                break

        # 最终损失
        final_loss, final_dict = compute_total_loss(
            model, self.xi_interior, self.xi_boundary, self.xi_center,
            self.gamma, self.n, self.mach_inf,
            weight_pde=weight_pde,
            weight_bc_shock=weight_bc_shock,
            weight_bc_center=weight_bc_center
        )
        print(f"  L-BFGS done, Loss: {final_loss.item():.6e}")
        print(f"    PDE: {final_dict['pde']:.4e}, BC_shock: {final_dict['bc_shock']:.4e}, BC_center: {final_dict['bc_center']:.4e}")

        return final_loss.item(), final_dict

    def run_scan(self):
        """运行完整扫描"""
        print("=" * 60)
        print("Guderley V7: 移除G + n=3球面 + 软约束")
        print("=" * 60)
        print(f"α值: {self.alpha_values}")
        print(f"n={self.n} (球面), γ={self.gamma}, M∞={self.mach_inf}")

        for i, alpha in enumerate(self.alpha_values, 1):
            print(f"\n[{i}/{len(self.alpha_values)}] α = {alpha:.3f}")
            final_loss, final_dict = self.train_fixed_alpha(alpha)
            self.results.append({
                'alpha': alpha,
                'final_loss': final_loss,
                'pde': final_dict['pde'],
                'bc_shock': final_dict['bc_shock'],
                'bc_center': final_dict['bc_center']
            })

        self.save_results()
        self.plot_results()
        self.analyze_results()

    def save_results(self):
        with open(self.output_dir / "scan_results.json", 'w') as f:
            json.dump({
                'alpha_values': self.alpha_values,
                'results': self.results,
                'gamma': self.gamma, 'n': self.n,
                'mach_inf': self.mach_inf,
                'version': 'v7_no_G_spherical'
            }, f, indent=2)
        print(f"\n结果已保存")

    def plot_results(self):
        alphas = [r['alpha'] for r in self.results]
        losses = [r['final_loss'] for r in self.results]
        pdes = [r['pde'] for r in self.results]
        bcs = [r['bc_shock'] for r in self.results]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 总Loss
        axes[0].plot(alphas, losses, 'b-o', linewidth=2, markersize=8)
        axes[0].axvline(x=0.717, color='r', linestyle='--', label='True alpha=0.717')
        axes[0].set_xlabel('alpha')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Total Loss vs alpha')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # PDE Loss
        axes[1].plot(alphas, pdes, 'g-o', linewidth=2, markersize=8)
        axes[1].axvline(x=0.717, color='r', linestyle='--', label='True alpha=0.717')
        axes[1].set_xlabel('alpha')
        axes[1].set_ylabel('PDE Loss')
        axes[1].set_title('PDE Loss vs alpha')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # BC Loss
        axes[2].plot(alphas, bcs, 'm-o', linewidth=2, markersize=8)
        axes[2].axvline(x=0.717, color='r', linestyle='--', label='True alpha=0.717')
        axes[2].set_xlabel('alpha')
        axes[2].set_ylabel('BC Shock Loss')
        axes[2].set_title('BC Shock Loss vs alpha')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "funnel_plot_v7.png", dpi=150, bbox_inches='tight')
        print("图已保存")
        plt.close()

    def analyze_results(self):
        print("\n" + "=" * 60)
        print("V7 扫描结果 (n=3球面, 无G变量)")
        print("=" * 60)

        min_loss = min(r['final_loss'] for r in self.results)
        min_alpha = [r['alpha'] for r in self.results if r['final_loss'] == min_loss][0]

        print(f"\n{'alpha':<10} {'Total':<12} {'PDE':<12} {'BC_shock':<12}")
        print("-" * 46)
        for r in self.results:
            note = ""
            if abs(r['alpha'] - 0.717) < 1e-6:
                note = " <- true"
            if r['final_loss'] == min_loss:
                note += " * min"
            print(f"{r['alpha']:<10.3f} {r['final_loss']:<12.4e} "
                  f"{r['pde']:<12.4e} {r['bc_shock']:<12.4e}{note}")

        print(f"\n最小Loss: {min_loss:.6e} at α={min_alpha:.3f}")

        if abs(min_alpha - 0.717) < 0.05:
            print("V型谷底 - 成功!")
        else:
            print(f"滑梯型 - 最小值在α={min_alpha:.3f}")


def main():
    alpha_values = [0.60, 0.68, 0.717, 0.75, 0.80, 0.90, 1.00]

    experiment = AlphaScanV7(
        alpha_values=alpha_values,
        gamma=1.4,
        n=3,  # 球面几何，对应α=0.717
        n_interior=200,
        mach_inf=10.0,
        output_dir="output_scan_v7"
    )

    experiment.run_scan()
    print("\nV7实验完成!")


if __name__ == "__main__":
    main()


