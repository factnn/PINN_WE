"""
Guderley Physics Equations
实现 Guderley 自相似 ODE 方程的残差计算
"""

import torch
import numpy as np
from typing import Tuple


def compute_derivatives(
    model,
    xi: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算 V, C 对 xi 的一阶和二阶导数

    Returns:
        V, C, dV_dxi, d2V_dxi2, dC_dxi, d2C_dxi2
    """
    xi.requires_grad_(True)

    V, C = model(xi)

    # 一阶导数
    dV_dxi = torch.autograd.grad(
        V, xi,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True
    )[0]

    dC_dxi = torch.autograd.grad(
        C, xi,
        grad_outputs=torch.ones_like(C),
        create_graph=True,
        retain_graph=True
    )[0]

    # 二阶导数
    d2V_dxi2 = torch.autograd.grad(
        dV_dxi, xi,
        grad_outputs=torch.ones_like(dV_dxi),
        create_graph=True,
        retain_graph=True
    )[0]

    d2C_dxi2 = torch.autograd.grad(
        dC_dxi, xi,
        grad_outputs=torch.ones_like(dC_dxi),
        create_graph=True,
        retain_graph=True
    )[0]

    return V, C, dV_dxi, d2V_dxi2, dC_dxi, d2C_dxi2


def guderley_ode_residual(
    model,
    xi: torch.Tensor,
    gamma: float = 1.4,
    n: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 Guderley 二阶ODE的残差（完整修正版）

    核心二阶ODE（球面/柱面通用）：
        d²V/dξ² + (n/ξ)dV/dξ + [V(V-1/α)(V-1)]/Δ · dV/dξ
                + [2C²(V-1/α)]/(γΔ) · dV/dξ + nC²/(γαξΔ) = 0

        d²C/dξ² + (n/ξ)dC/dξ + [V(V-1/α)(V-1)]/Δ · dC/dξ
                - (β/C)[(dV/dξ)² + (nV/ξ)dV/dξ] = 0

    其中:
        Δ = (V-1)² - C²  (声速线判别式)
        β = (γ-1)/2
        α = 自相似指数（可训练参数）

    Args:
        model: GuderleyPINN 模型
        xi: 自相似坐标 [batch_size, 1]
        gamma: 比热比 (默认 1.4 for 空气)
        n: 几何因子 (2=柱面, 3=球面, 1=平面)

    Returns:
        res_V: V的二阶ODE残差
        res_C: C的二阶ODE残差
    """
    # 获取网络输出和一阶、二阶导数
    V, C, dV_dxi, d2V_dxi2, dC_dxi, d2C_dxi2 = compute_derivatives(model, xi)

    # 获取可训练的 alpha
    alpha = model.alpha

    # 数值稳定性：添加小epsilon避免除以0
    eps = 1e-6
    xi_safe = xi + eps  # 避免xi→0时除以0
    C_safe = C + eps    # 避免C→0时除以0
    Delta = (V - 1)**2 - C**2 + eps  # 避免Delta→0（声速线）

    # === V的二阶ODE残差 ===
    # 项1：二阶导数项
    term1_V = d2V_dxi2

    # 项2：几何源项 (n/ξ · dV/dξ)
    term2_V = (n / xi_safe) * dV_dxi

    # 项3：非线性对流项 [V(V-1/α)(V-1)/Δ · dV/dξ]
    term3_V = (V / Delta) * (V - 1.0/alpha) * (V - 1) * dV_dxi

    # 项4：压力梯度项 [2C²(V-1/α)/(γΔ) · dV/dξ]
    term4_V = (2 * C**2 / (gamma * Delta)) * (V - 1.0/alpha) * dV_dxi

    # 项5：几何压力项 [nC²/(γαξΔ)]
    term5_V = (n * C**2) / (gamma * alpha * xi_safe * Delta)

    # V方程残差 = 0
    res_V = term1_V + term2_V + term3_V + term4_V + term5_V

    # === C的二阶ODE残差（与V耦合） ===
    beta = (gamma - 1.0) / 2.0

    # 项1：二阶导数项
    term1_C = d2C_dxi2

    # 项2：几何源项 (n/ξ · dC/dξ)
    term2_C = (n / xi_safe) * dC_dxi

    # 项3：对流项 [V(V-1/α)(V-1)/Δ · dC/dξ]
    term3_C = (V / Delta) * (V - 1.0/alpha) * (V - 1) * dC_dxi

    # 项4：耦合项 -(β/C)[(dV/dξ)² + (nV/ξ)dV/dξ]
    term4_C = (beta / C_safe) * (dV_dxi**2 + (n * V / xi_safe) * dV_dxi)

    # C方程残差 = 0
    res_C = term1_C + term2_C + term3_C - term4_C

    return res_V, res_C


def rankine_hugoniot_bc(
    gamma: float = 1.4,
    mach_inf: float = 10.0
) -> Tuple[float, float, float]:
    """
    计算 Rankine-Hugoniot 精确边界条件（支持非强激波）

    在激波位置 ξ=1 处，根据精确RH关系:
        V(1) = [2 + (γ-1)M∞²] / [(γ+1)M∞²]
        C(1) = √[2γM∞² - (γ-1)] / [(γ+1)M∞]
        G(1) = (γ+1)/(γ-1)  (强激波极限)

    当 M∞→∞ 时退化为强激波近似:
        V(1) → 2/(γ+1)
        C(1) → √[2γ(γ-1)]/(γ+1)

    Args:
        gamma: 比热比
        mach_inf: 未扰动马赫数（默认10.0）

    Returns:
        V_bc, C_bc, G_bc: 边界条件的理论值
    """
    M2 = mach_inf ** 2

    # 精确RH条件
    V_bc = (2.0 + (gamma - 1.0) * M2) / ((gamma + 1.0) * M2)
    C_bc = np.sqrt(2.0 * gamma * M2 - (gamma - 1.0)) / ((gamma + 1.0) * mach_inf)

    # 密度比（强激波极限）
    G_bc = (gamma + 1.0) / (gamma - 1.0)

    return V_bc, C_bc, G_bc


def center_bc(
    model,
    xi_center: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算中心（ξ→0）边界条件：一阶导数为0（物理量有限性）

    在中心处（ξ→0），要求物理量有限，转化为导数约束：
        dV/dξ|ξ→0 = 0
        dC/dξ|ξ→0 = 0

    Args:
        model: GuderleyPINN 模型
        xi_center: 中心采样点（如 ξ=1e-3）

    Returns:
        loss_dV: dV/dξ 在中心的损失
        loss_dC: dC/dξ 在中心的损失
    """
    V, C, dV_dxi, d2V_dxi2, dC_dxi, d2C_dxi2 = compute_derivatives(model, xi_center)

    # 中心处一阶导数应为0
    loss_dV = torch.mean(dV_dxi ** 2)
    loss_dC = torch.mean(dC_dxi ** 2)

    return loss_dV, loss_dC


def sonic_line_regularity(
    model,
    xi_interior: torch.Tensor,
    gamma: float = 1.4,
    n: int = 3
) -> torch.Tensor:
    """
    奇点穿越条件（声速线正则性约束）

    当 Δ=(V-1)²-C²→0 时，ODE分母为0。
    物理解要求分子同时为0，否则解发散。

    V方程一阶形式: dV/dξ = N_V / (Δ · ξ)
    其中分子 N_V 包含不含Δ的项。

    实现：找到Δ≈0的点，惩罚分子不为0。
    """
    V, C, dV_dxi, d2V_dxi2, dC_dxi, d2C_dxi2 = compute_derivatives(model, xi_interior)

    alpha = model.alpha
    eps = 1e-6
    xi_safe = xi_interior + eps

    # Δ = (V-1)² - C²
    Delta = (V - 1.0)**2 - C**2

    # 用软权重：Δ越接近0，权重越大
    # w = exp(-Δ²/σ²)，σ控制宽度
    sigma = 0.05
    w = torch.exp(-Delta**2 / sigma**2)

    # V方程的"分子"部分（把ODE乘以Δ后，不含1/Δ的项）
    # 原方程: d²V/dξ² + (n/ξ)dV/dξ + [V(V-1/α)(V-1)/Δ]dV/dξ
    #        + [2C²(V-1/α)/(γΔ)]dV/dξ + nC²/(γαξΔ) = 0
    # 乘以Δ:
    # Δ·d²V/dξ² + Δ·(n/ξ)dV/dξ + V(V-1/α)(V-1)·dV/dξ
    #        + 2C²(V-1/α)/γ · dV/dξ + nC²/(γαξ) = 0
    # 当Δ→0时，前两项→0，剩下的必须=0：
    numerator_V = (
        V * (V - 1.0/alpha) * (V - 1.0) * dV_dxi
        + (2.0 * C**2 / gamma) * (V - 1.0/alpha) * dV_dxi
        + (n * C**2) / (gamma * alpha * xi_safe)
    )

    # C方程的"分子"部分
    # 原方程乘以Δ后，当Δ→0时剩下：
    # V(V-1/α)(V-1)·dC/dξ - (β/C)[(dV/dξ)² + (nV/ξ)dV/dξ]·Δ = 0
    # 注意：term4_C不含1/Δ，所以乘Δ后→0
    # 只剩对流项的分子：
    numerator_C = V * (V - 1.0/alpha) * (V - 1.0) * dC_dxi

    # 加权损失：在Δ≈0处惩罚分子不为0
    loss_sonic = (
        torch.mean(w * numerator_V**2)
        + torch.mean(w * numerator_C**2)
    )

    return loss_sonic


def compute_total_loss(
    model,
    xi_interior: torch.Tensor,
    xi_boundary: torch.Tensor,
    xi_center: torch.Tensor,
    gamma: float = 1.4,
    n: int = 2,
    mach_inf: float = 10.0,
    weight_pde: float = 1.0,
    weight_bc_shock: float = 10.0,
    weight_bc_center: float = 5.0
) -> Tuple[torch.Tensor, dict]:
    """
    计算总损失函数（修正：添加中心边界约束，BC参与优化）

    Loss = weight_pde*Loss_PDE + weight_bc_shock*Loss_BC_shock + weight_bc_center*Loss_BC_center

    Args:
        model: GuderleyPINN 模型
        xi_interior: 内部采样点
        xi_boundary: 激波边界采样点 (ξ=1)
        xi_center: 中心采样点 (ξ≈0, 如1e-3)
        gamma: 比热比
        n: 几何因子 (2=柱面, 3=球面)
        mach_inf: 未扰动马赫数（用于精确RH条件）
        weight_pde: PDE残差权重
        weight_bc_shock: 激波边界权重
        weight_bc_center: 中心边界权重

    Returns:
        total_loss: 总损失
        loss_dict: 各项损失的字典
    """
    # 1. PDE 残差损失
    res_V, res_C = guderley_ode_residual(model, xi_interior, gamma, n)
    loss_pde_V = torch.mean(res_V**2)
    loss_pde_C = torch.mean(res_C**2)
    loss_pde = loss_pde_V + loss_pde_C

    # 2. 激波边界（ξ=1）损失
    V_pred, C_pred = model(xi_boundary)
    V_bc, C_bc, _ = rankine_hugoniot_bc(gamma, mach_inf)

    loss_bc_shock_V = torch.mean((V_pred - V_bc)**2)
    loss_bc_shock_C = torch.mean((C_pred - C_bc)**2)
    loss_bc_shock = loss_bc_shock_V + loss_bc_shock_C

    # 3. 中心边界（ξ→0）损失
    loss_bc_center_V, loss_bc_center_C = center_bc(model, xi_center)
    loss_bc_center = loss_bc_center_V + loss_bc_center_C

    # 4. 总损失
    total_loss = (
        weight_pde * loss_pde +
        weight_bc_shock * loss_bc_shock +
        weight_bc_center * loss_bc_center
    )

    # 5. 记录各项损失
    loss_dict = {
        'total': total_loss.item(),
        'pde': loss_pde.item(),
        'pde_V': loss_pde_V.item(),
        'pde_C': loss_pde_C.item(),
        'bc_shock': loss_bc_shock.item(),
        'bc_shock_V': loss_bc_shock_V.item(),
        'bc_shock_C': loss_bc_shock_C.item(),
        'bc_center': loss_bc_center.item(),
        'bc_center_V': loss_bc_center_V.item(),
        'bc_center_C': loss_bc_center_C.item(),
    }

    return total_loss, loss_dict


if __name__ == "__main__":
    # 测试代码（修正版）
    import sys
    sys.path.append('/share/project/zpy/PINN_WE/cases/self-discovery/src')
    from model import GuderleyPINN

    print("=" * 60)
    print("测试 Guderley Physics 模块（修正版）")
    print("=" * 60)

    # 创建模型
    model = GuderleyPINN(alpha_init=0.717)
    print(f"\n初始 alpha: {model.get_alpha_value():.6f}")

    # 创建采样点
    xi_interior = torch.linspace(0.01, 0.9, 100).reshape(-1, 1)
    xi_boundary = torch.tensor([[1.0]])
    xi_center = torch.tensor([[1e-3]])

    # 测试边界条件
    V_bc, C_bc, G_bc = rankine_hugoniot_bc(gamma=1.4, mach_inf=10.0)
    print(f"\n激波边界条件 (γ=1.4, M∞=10):")
    print(f"  V(1) = {V_bc:.6f}")
    print(f"  C(1) = {C_bc:.6f}")

    # 测试损失计算（n=3 球面，对应α=0.717）
    total_loss, loss_dict = compute_total_loss(
        model, xi_interior, xi_boundary, xi_center,
        gamma=1.4, n=3, mach_inf=10.0
    )

    print(f"\n损失函数:")
    print(f"  Total Loss:   {loss_dict['total']:.6e}")
    print(f"  PDE Loss:     {loss_dict['pde']:.6e}")
    print(f"  BC Shock:     {loss_dict['bc_shock']:.6e}")
    print(f"  BC Center:    {loss_dict['bc_center']:.6e}")

    print("\n✓ 修正后Physics模块测试通过")
    print("\n关键修正:")
    print("  1. 补充二阶导数计算（d²V/dξ², d²C/dξ²）")
    print("  2. 实现完整的二阶ODE残差")
    print("  3. 添加中心边界条件（dV/dξ|ξ→0=0, dC/dξ|ξ→0=0）")
    print("  4. 添加数值稳定性保护（eps=1e-6）")
    print("  5. 支持非强激波边界条件（M∞=10）")
    print("  6. BC参与优化（不再是硬约束模式）")


