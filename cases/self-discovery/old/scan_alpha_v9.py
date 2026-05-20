#!/usr/bin/env python3
"""
Legacy alpha scan for the old self-discovery branch.

当前默认主线请改看 `cases/self-discovery/canonical/README.md`。

Guderley V9: 回归基础 - 修正训练策略
关键改进:
  1. 采样范围 [0.001, 0.99]，边界加密
  2. 学习率 1e-4
  3. 权重比 PDE:BC_shock:BC_center = 1:10:5
  4. 无sonic条件，无G变量，n=3球面
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


def create_dense_sampling(n_total=300, device='cpu'):
    """
    创建边界加密的采样点分布
    在 ξ→0 和 ξ→1 附近加密采样

    分布策略:
      - [0.001, 0.05]: 30点 (中心附近加密)
      - [0.05, 0.90]:  200点 (内部均匀)
      - [0.90, 0.99]:  70点 (激波附近加密)
    """
    xi_center_region = torch.linspace(0.001, 0.05, 30)
    xi_interior = torch.linspace(0.05, 0.90, 200)
    xi_shock_region = torch.linspace(0.90, 0.99, 70)

    xi_all = torch.cat([xi_center_region, xi_interior[1:], xi_shock_region[1:]])
    xi_all = xi_all.reshape(-1, 1).to(device)
    return xi_all


class AlphaScanV9:
    def __init__(self, alpha_values, gamma=1.4, n=3,
                 n_interior=300, mach_inf=10.0,
                 output_dir="output_scan_v9"):
        self.alpha_values = alpha_values
        self.gamma = gamma
        self.n = n
        self.mach_inf = mach_inf
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"设备: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # 边界加密采样
        self.xi_interior = create_dense_sampling(
            n_total=n_interior, device=self.device)
        self.xi_boundary = torch.tensor(
            [[1.0]]).to(self.device)
        self.xi_center = torch.tensor(
            [[1e-3]]).to(self.device)

        print(f"采样点数: {self.xi_interior.shape[0]}")
        print(f"采样范围: [{self.xi_interior.min().item():.4f}, "
              f"{self.xi_interior.max().item():.4f}]")

        self.results = []

    def train_fixed_alpha(self, alpha_fixed,
                          adam_epochs=8000, lbfgs_epochs=3000,
                          w_pde=1.0, w_bc=10.0, w_center=5.0,
                          lr=1e-4):
        """固定α值训练网络"""
        model = GuderleyPINN(alpha_init=alpha_fixed)
        model.to(self.device)
        model.train()
        model.raw_alpha.requires_grad = False

        opt = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr)

        # Adam 训练
        for ep in range(1, adam_epochs + 1):
            opt.zero_grad()
            tl, ld = compute_total_loss(
                model, self.xi_interior, self.xi_boundary,
                self.xi_center, self.gamma, self.n,
                self.mach_inf, weight_pde=w_pde,
                weight_bc_shock=w_bc,
                weight_bc_center=w_center)
            tl.backward()
            opt.step()
            if ep % 1000 == 0 or ep == 1:
                print(f"  Ep {ep:5d} | T:{ld['total']:.3e} "
                      f"PDE:{ld['pde']:.3e} "
                      f"BC:{ld['bc_shock']:.3e} "
                      f"Ctr:{ld['bc_center']:.3e}")

        # L-BFGS 精细优化
        lbfgs = optim.LBFGS(
            [p for p in model.parameters() if p.requires_grad],
            lr=1.0, max_iter=20, history_size=50,
            line_search_fn='strong_wolfe')
        it = [0]

        def closure():
            lbfgs.zero_grad()
            tl, _ = compute_total_loss(
                model, self.xi_interior, self.xi_boundary,
                self.xi_center, self.gamma, self.n,
                self.mach_inf, weight_pde=w_pde,
                weight_bc_shock=w_bc,
                weight_bc_center=w_center)
            tl.backward()
            it[0] += 1
            return tl

        for _ in range(lbfgs_epochs // 20):
            lbfgs.step(closure)
            if it[0] >= lbfgs_epochs:
                break

        # 最终评估
        fl, fd = compute_total_loss(
            model, self.xi_interior, self.xi_boundary,
            self.xi_center, self.gamma, self.n,
            self.mach_inf, weight_pde=w_pde,
            weight_bc_shock=w_bc,
            weight_bc_center=w_center)
        print(f"  Done | T:{fd['total']:.3e} "
              f"PDE:{fd['pde']:.3e} "
              f"BC:{fd['bc_shock']:.3e} "
              f"Ctr:{fd['bc_center']:.3e}")
        return fl.item(), fd

    def run_scan(self):
        """运行完整扫描"""
        print("=" * 60)
        print("V9: 回归基础 - 修正训练策略")
        print("=" * 60)
        print(f"α值: {self.alpha_values}")
        print(f"n={self.n} (球面), γ={self.gamma}, M∞={self.mach_inf}")
        print(f"权重: PDE=1, BC=10, Center=5")
        print(f"学习率: 1e-4, Adam=8000, L-BFGS=3000")

        for i, a in enumerate(self.alpha_values, 1):
            print(f"\n[{i}/{len(self.alpha_values)}] α={a:.3f}")
            fl, fd = self.train_fixed_alpha(a)
            self.results.append({
                'alpha': a, 'total': fl,
                'pde': fd['pde'],
                'bc_shock': fd['bc_shock'],
                'bc_center': fd['bc_center']
            })

        self.save_and_analyze()

    def save_and_analyze(self):
        """保存结果并分析"""
        # 保存JSON
        with open(self.output_dir / "results.json", 'w') as f:
            json.dump({'results': self.results}, f, indent=2)

        # 分析
        print("\n" + "=" * 60)
        print("V9 结果")
        print("=" * 60)

        mn = min(r['total'] for r in self.results)
        ma = [r['alpha'] for r in self.results
              if r['total'] == mn][0]

        print(f"\n{'alpha':<8} {'Total':<12} "
              f"{'PDE':<12} {'BC':<12} {'Center':<12}")
        print("-" * 56)

        for r in self.results:
            note = ""
            if abs(r['alpha'] - 0.717) < 1e-6:
                note = " <-true"
            if r['total'] == mn:
                note += " *min"
            print(f"{r['alpha']:<8.3f} "
                  f"{r['total']:<12.3e} "
                  f"{r['pde']:<12.3e} "
                  f"{r['bc_shock']:<12.3e} "
                  f"{r['bc_center']:<12.3e}{note}")

        print(f"\n最小Loss: {mn:.3e} at α={ma:.3f}")
        if abs(ma - 0.717) < 0.05:
            print("V型谷底 - 成功!")
        else:
            print(f"最小值在α={ma:.3f}")

        self._plot_results()

    def _plot_results(self):
        """绘制结果图"""
        alphas = [r['alpha'] for r in self.results]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, key, title in zip(
            axes,
            ['total', 'pde', 'bc_shock'],
            ['Total Loss', 'PDE Loss', 'BC Shock Loss']
        ):
            vals = [r[key] for r in self.results]
            ax.plot(alphas, vals, 'b-o', lw=2, ms=8)
            ax.axvline(x=0.717, color='r',
                       ls='--', label='true α=0.717')
            ax.set_xlabel('alpha')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "v9_plot.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("图已保存")


def main():
    alphas = [0.60, 0.68, 0.717, 0.75,
              0.80, 0.90, 1.00]

    exp = AlphaScanV9(
        alpha_values=alphas,
        gamma=1.4, n=3,
        n_interior=300, mach_inf=10.0,
        output_dir="output_scan_v9")

    exp.run_scan()
    print("\nV9完成!")


if __name__ == "__main__":
    main()
