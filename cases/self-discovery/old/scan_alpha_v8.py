#!/usr/bin/env python3
"""
Legacy alpha scan for the old self-discovery branch.

当前默认主线请改看 `cases/self-discovery/canonical/README.md`。

Guderley V8: 加入奇点穿越条件
"""

import sys
sys.path.append('/share/project/zpy/PINN_WE/cases/self-discovery/src')

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import json

from model import GuderleyPINN
from physics import compute_total_loss


class AlphaScanV8:
    def __init__(self, alpha_values, gamma=1.4, n=3,
                 n_interior=200, mach_inf=10.0,
                 output_dir="output_scan_v8"):
        self.alpha_values = alpha_values
        self.gamma = gamma
        self.n = n
        self.mach_inf = mach_inf
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"设备: {self.device}")

        self.xi_interior = torch.linspace(
            0.01, 0.9, n_interior
        ).reshape(-1, 1).to(self.device)
        self.xi_boundary = torch.tensor(
            [[1.0]]).to(self.device)
        self.xi_center = torch.tensor(
            [[1e-3]]).to(self.device)
        self.results = []

    def train_fixed_alpha(self, alpha_fixed,
                          adam_epochs=5000, lbfgs_epochs=2000,
                          w_pde=1.0, w_bc=100.0,
                          w_center=10.0, w_sonic=50.0):
        model = GuderleyPINN(alpha_init=alpha_fixed)
        model.to(self.device)
        model.train()
        model.raw_alpha.requires_grad = False

        opt = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3)

        for ep in range(1, adam_epochs + 1):
            opt.zero_grad()
            tl, ld = compute_total_loss(
                model, self.xi_interior, self.xi_boundary,
                self.xi_center, self.gamma, self.n,
                self.mach_inf, weight_pde=w_pde,
                weight_bc_shock=w_bc,
                weight_bc_center=w_center,
                weight_sonic=w_sonic)
            tl.backward()
            opt.step()
            if ep % 1000 == 0 or ep == 1:
                print(f"  Ep {ep:4d} | T:{ld['total']:.3e} "
                      f"PDE:{ld['pde']:.3e} "
                      f"BC:{ld['bc_shock']:.3e} "
                      f"Son:{ld['sonic']:.3e}")

        # L-BFGS
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
                weight_bc_center=w_center,
                weight_sonic=w_sonic)
            tl.backward()
            it[0] += 1
            return tl

        for _ in range(lbfgs_epochs // 20):
            lbfgs.step(closure)
            if it[0] >= lbfgs_epochs:
                break

        fl, fd = compute_total_loss(
            model, self.xi_interior, self.xi_boundary,
            self.xi_center, self.gamma, self.n,
            self.mach_inf, weight_pde=w_pde,
            weight_bc_shock=w_bc,
            weight_bc_center=w_center,
            weight_sonic=w_sonic)
        print(f"  Done | T:{fd['total']:.3e} "
              f"PDE:{fd['pde']:.3e} "
              f"BC:{fd['bc_shock']:.3e} "
              f"Son:{fd['sonic']:.3e}")
        return fl.item(), fd

    def run_scan(self):
        print("=" * 60)
        print("V8: 奇点穿越条件 + n=3球面 + 软约束")
        print("=" * 60)
        print(f"α值: {self.alpha_values}")
        print(f"n={self.n}, γ={self.gamma}")

        for i, a in enumerate(self.alpha_values, 1):
            print(f"\n[{i}/{len(self.alpha_values)}] α={a:.3f}")
            fl, fd = self.train_fixed_alpha(a)
            self.results.append({
                'alpha': a, 'total': fl,
                'pde': fd['pde'],
                'bc_shock': fd['bc_shock'],
                'sonic': fd['sonic']
            })

        self.save_and_analyze()

    def save_and_analyze(self):
        # 保存
        with open(self.output_dir / "results.json", 'w') as f:
            json.dump({'results': self.results}, f, indent=2)

        # 分析
        print("\n" + "=" * 60)
        print("V8 结果")
        print("=" * 60)

        mn = min(r['total'] for r in self.results)
        ma = [r['alpha'] for r in self.results
              if r['total'] == mn][0]

        print(f"\n{'alpha':<8} {'Total':<11} "
              f"{'PDE':<11} {'BC':<11} {'Sonic':<11}")
        print("-" * 52)
        for r in self.results:
            note = ""
            if abs(r['alpha'] - 0.717) < 1e-6:
                note = " <-true"
            if r['total'] == mn:
                note += " *min"
            print(f"{r['alpha']:<8.3f} "
                  f"{r['total']:<11.3e} "
                  f"{r['pde']:<11.3e} "
                  f"{r['bc_shock']:<11.3e} "
                  f"{r['sonic']:<11.3e}{note}")

        print(f"\n最小Loss: {mn:.3e} at α={ma:.3f}")
        if abs(ma - 0.717) < 0.05:
            print("V型谷底 - 成功!")
        else:
            print(f"最小值在α={ma:.3f}")

        # 绘图
        alphas = [r['alpha'] for r in self.results]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, key, title in zip(
            axes,
            ['total', 'pde', 'sonic'],
            ['Total Loss', 'PDE Loss', 'Sonic Loss']
        ):
            vals = [r[key] for r in self.results]
            ax.plot(alphas, vals, 'b-o', lw=2, ms=8)
            ax.axvline(x=0.717, color='r',
                       ls='--', label='true')
            ax.set_xlabel('alpha')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "v8_plot.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("图已保存")


def main():
    alphas = [0.60, 0.68, 0.717, 0.75,
              0.80, 0.90, 1.00]

    exp = AlphaScanV8(
        alpha_values=alphas,
        gamma=1.4, n=3,
        n_interior=200, mach_inf=10.0,
        output_dir="output_scan_v8")

    exp.run_scan()
    print("\nV8完成!")


if __name__ == "__main__":
    main()
