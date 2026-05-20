#!/usr/bin/env python3
"""
Guderley 自相似方程 - 传统数值sanity check

目的: 用shooting/BVP方法验证:
  1. 方程本身是否正确
  2. 球面情况下α≈0.717处是否真的有正则穿越
  3. 为后续PINN实现提供可信的数值基准

方程来源: Lazarus (1981), Whitham Ch.8
约定: u = Ṙ·V(ξ), c = Ṙ·C(ξ), ρ = ρ₁·G(ξ), ξ = r/R(t), R ~ (-t)^α

标准一阶ODE (消去G后的V-C系统):

从三个原始方程:
  (I)   (V-ξ)·G'/G + V' + j·V/ξ = 0
  (II)  (V-ξ)·V' + C²·G'/(γG) + V·(V-1/α)/ξ = 0
  (III) (V-ξ)·C'/C + β·(V-ξ)·G'/G + (1/α-1) = 0

消去G'/G后:
  Δ·V' = N_V/ξ
  (V-ξ)·C' = -β·C·(V'+j·V/ξ) - C·(1/α-1)

其中:
  Δ = (V-ξ)² - C²/γ
  N_V = C²·j·V/γ - (V-ξ)·V·(V-1/α)

几何因子: j=0平面, j=1柱面, j=2球面

边界条件(ξ=1, 强激波RH):
  V(1) = 2/(γ+1)
  C(1) = sqrt(2γ(γ-1))/(γ+1)
  G(1) = (γ+1)/(γ-1)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========== 物理参数 ==========
gamma = 1.4
j_geom = 2      # 球面对称 (j=0平面, j=1柱面, j=2球面)
beta = (gamma - 1.0) / 2.0

# ========== 强激波RH边界条件 (ξ=1) ==========
V_bc = 2.0 / (gamma + 1.0)                              # = 0.8333
C_bc = np.sqrt(2.0 * gamma * (gamma - 1.0)) / (gamma + 1.0)  # ≈ 0.4410
G_bc = (gamma + 1.0) / (gamma - 1.0)                    # = 6.0

print(f"Guderley 自相似方程数值验证")
print(f"γ={gamma}, j={j_geom} (球面)")
print(f"RH边界条件 (ξ=1):")
print(f"  V(1) = {V_bc:.6f}")
print(f"  C(1) = {C_bc:.6f}")
print(f"  G(1) = {G_bc:.6f}")


def guderley_rhs(xi, y, alpha):
    """
    一阶ODE右端项: dy/dξ = f(ξ, y)
    y = [V, C]

    方程:
      Δ·V' = N_V/ξ   =>  V' = N_V/(ξ·Δ)
      (V-ξ)·C' = -β·C·(V'+jV/ξ) - C·(1/α-1)
                =>  C' = [-β·C·(V'+jV/ξ) - C·(1/α-1)] / (V-ξ)
    """
    V, C = y

    # 避免除以0
    eps = 1e-12
    xi_s = max(xi, eps)

    # Delta = (V-ξ)² - C²/γ
    Delta = (V - xi)**2 - C**2 / gamma

    # 避免Delta=0 (声速线奇点)
    if abs(Delta) < 1e-10:
        return [0.0, 0.0]  # 在奇点处停止

    # V方程 (正确形式: N_V = C²jV/γ - V(α-1)(V-ξ)/α)
    N_V = C**2 * j_geom * V / gamma - (V - xi) * V * (alpha - 1.0) / alpha
    dV = N_V / (xi_s * Delta)

    # C方程 (用V'代入)
    Vm_xi = V - xi
    if abs(Vm_xi) < 1e-10:
        return [dV, 0.0]

    dC = (-beta * C * (dV + j_geom * V / xi_s) - C * (1.0/alpha - 1.0)) / Vm_xi

    return [dV, dC]


def check_regularity(alpha, xi_eval=None, verbose=False):
    """
    对给定α, 从ξ=1向ξ=0积分, 检查是否正则穿越声速线

    返回: (是否成功到达, 最小|Δ|, 最终ξ值)
    """
    # 从ξ=1-δ开始(稍微离开边界避免数值问题)
    xi_start = 0.999
    xi_end = 0.01

    # 在ξ=1处的初值用RH条件
    V0, C0 = V_bc, C_bc

    try:
        sol = solve_ivp(
            lambda xi, y: guderley_rhs(xi, y, alpha),
            [xi_start, xi_end],
            [V0, C0],
            method='RK45',
            max_step=0.001,
            rtol=1e-10,
            atol=1e-12,
            dense_output=True
        )

        if sol.success:
            V_sol = sol.y[0]
            C_sol = sol.y[1]
            xi_sol = sol.t

            # 检查Delta
            Delta_sol = (V_sol - xi_sol)**2 - C_sol**2 / gamma
            min_abs_Delta = np.min(np.abs(Delta_sol))

            # 检查物理性 (V>0, C>0)
            if np.any(V_sol < -1) or np.any(C_sol < 0):
                return False, min_abs_Delta, xi_sol[-1], None

            if verbose:
                print(f"  积分成功: ξ从{xi_start}到{xi_sol[-1]:.4f}")
                print(f"  V范围: [{V_sol.min():.4f}, {V_sol.max():.4f}]")
                print(f"  C范围: [{C_sol.min():.4f}, {C_sol.max():.4f}]")
                print(f"  min|Δ|: {min_abs_Delta:.6e}")

            return True, min_abs_Delta, xi_sol[-1], sol
        else:
            return False, float('inf'), xi_start, None

    except Exception as e:
        if verbose:
            print(f"  积分失败: {e}")
        return False, float('inf'), xi_start, None


# ========== 扫描α ==========
print(f"\n{'='*60}")
print(f"扫描α值，检查声速线穿越")
print(f"{'='*60}")

alphas = np.linspace(0.60, 0.90, 61)
results = []

for alpha in alphas:
    success, min_delta, final_xi, sol = check_regularity(alpha)
    results.append({
        'alpha': alpha,
        'success': success,
        'min_delta': min_delta,
        'final_xi': final_xi
    })

# 打印关键结果
print(f"\n{'alpha':>8s}  {'min|Δ|':>12s}  {'final_ξ':>10s}  {'status':>10s}")
print("-" * 50)
for r in results:
    if abs(r['alpha'] - 0.717) < 0.003 or r['min_delta'] < 0.01:
        mark = " <--" if abs(r['alpha'] - 0.717) < 0.003 else ""
        print(f"{r['alpha']:8.4f}  {r['min_delta']:12.4e}  {r['final_xi']:10.4f}  "
              f"{'OK' if r['success'] else 'FAIL'}{mark}")

# 找最小|Δ|对应的α (这应该就是Guderley特征值)
min_delta_vals = [r['min_delta'] for r in results if r['success']]
alpha_vals = [r['alpha'] for r in results if r['success']]

if min_delta_vals:
    idx_min = np.argmin(min_delta_vals)
    print(f"\nmin|Δ|最小值: {min_delta_vals[idx_min]:.6e} 对应 α={alpha_vals[idx_min]:.4f}")
    print(f"理论值: α≈0.717")

# ========== 画图 ==========
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. min|Δ| vs α
ax = axes[0, 0]
alpha_ok = [r['alpha'] for r in results if r['success']]
delta_ok = [r['min_delta'] for r in results if r['success']]
ax.semilogy(alpha_ok, delta_ok, 'b.-', markersize=4)
ax.axvline(x=0.717, color='r', ls='--', label='α=0.717')
ax.set_xlabel('α')
ax.set_ylabel('min|Δ|')
ax.set_title('声速线接近程度 vs α')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 积分终止位置 vs α
ax = axes[0, 1]
final_xi = [r['final_xi'] for r in results]
ax.plot([r['alpha'] for r in results], final_xi, 'b.-', markersize=4)
ax.axvline(x=0.717, color='r', ls='--', label='α=0.717')
ax.set_xlabel('α')
ax.set_ylabel('积分终止ξ')
ax.set_title('积分能走多远')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. α=0.717时的V,C分布
success, _, _, sol_717 = check_regularity(0.717, verbose=True)
if sol_717 is not None:
    ax = axes[1, 0]
    ax.plot(sol_717.t, sol_717.y[0], 'b-', label='V(ξ)')
    ax.plot(sol_717.t, sol_717.y[1], 'r-', label='C(ξ)')
    Delta_717 = (sol_717.y[0] - sol_717.t)**2 - sol_717.y[1]**2/gamma
    ax.plot(sol_717.t, np.abs(Delta_717), 'g--', alpha=0.5, label='|Δ|')
    ax.set_xlabel('ξ')
    ax.set_ylabel('V, C')
    ax.set_title('α=0.717 解的分布')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 4. 对比几个α值的V分布
ax = axes[1, 1]
for alpha_test in [0.70, 0.717, 0.75, 0.80]:
    _, _, _, sol_t = check_regularity(alpha_test)
    if sol_t is not None:
        ax.plot(sol_t.t, sol_t.y[0], label=f'α={alpha_test}')
ax.set_xlabel('ξ')
ax.set_ylabel('V(ξ)')
ax.set_title('不同α下的V分布')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
outpath = '/share/project/zpy/PINN_WE/cases/self-discovery/numerical_sanity_check.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\n图已保存到 {outpath}")
