"""
Taylor-Maccoll 锥面绕流: 共享模块

物理问题: 超音速流绕锥体，给定(M∞, γ, θ_c)，求激波角β。
这是一个eigenvalue BVP，结构与Guderley完全类似。

ODE: Taylor-Maccoll方程 (球坐标下轴对称锥面流)
  dv_r/dθ = v_θ
  dv_θ/dθ = [v_θ²·v_r - c̃²·(2v_r + v_θ·cotθ)] / [c̃² - v_θ²]

其中 c̃² = (γ-1)/2·(1 - v_r² - v_θ²), 速度由V_max归一化。

边界条件:
  θ=β (激波): v_r, v_θ 由斜激波关系确定
  θ=θ_c (锥面): v_θ = 0 (切向条件)
"""
import numpy as np
import torch


# ── 基本物理量 ──────────────────────────────────

def v1_over_vmax(M_inf, gamma):
    """V∞/V_max = M∞·c∞/V_max, 其中 V_max² = V∞² + 2c∞²/(γ-1)"""
    return np.sqrt(M_inf**2 * (gamma - 1)) / np.sqrt(M_inf**2 * (gamma - 1) + 2)


def oblique_shock_post(M_inf, beta_rad, gamma):
    """
    斜激波后条件 (锥面激波, 法方向为θ方向)

    Returns: (v_r, v_θ) normalized by V_max
    v_r = tangential component (preserved)
    v_θ = normal component (compressed, < 0)
    """
    v1 = v1_over_vmax(M_inf, gamma)
    M_n1 = M_inf * np.sin(beta_rad)

    # 法向速度比 (normal shock relation)
    vn_ratio = ((gamma - 1) * M_n1**2 + 2) / ((gamma + 1) * M_n1**2)

    v_r = v1 * np.cos(beta_rad)       # tangential preserved
    v_theta = -v1 * np.sin(beta_rad) * vn_ratio  # normal compressed, < 0

    return v_r, v_theta


def oblique_shock_post_torch(M_inf, beta, gamma):
    """PyTorch可微版本 (β为tensor)"""
    v1_sq_num = M_inf**2 * (gamma - 1)
    v1 = torch.sqrt(torch.tensor(v1_sq_num, dtype=torch.float64)
                     / (v1_sq_num + 2))

    M_n1_sq = M_inf**2 * torch.sin(beta)**2
    vn_ratio = ((gamma - 1) * M_n1_sq + 2) / ((gamma + 1) * M_n1_sq)

    v_r = v1 * torch.cos(beta)
    v_theta = -v1 * torch.sin(beta) * vn_ratio

    return v_r, v_theta


# ── Taylor-Maccoll ODE ─────────────────────────

def taylor_maccoll_rhs(theta, y, gamma):
    """
    Taylor-Maccoll ODE右端项: dy/dθ = f(θ, y)

    y = [v_r, v_θ]
    dv_r/dθ = v_θ
    dv_θ/dθ = [v_θ²·v_r - c̃²·(2v_r + v_θ·cotθ)] / [c̃² - v_θ²]

    Anderson, Modern Compressible Flow, Ch.10
    """
    vr, vt = y

    # 归一化声速平方
    c2 = (gamma - 1) / 2 * (1 - vr**2 - vt**2)

    if c2 <= 0:
        return [0.0, 1e10]  # 超出物理范围

    denom = c2 - vt**2

    if abs(denom) < 1e-14:
        return [0.0, 1e10]  # sonic line

    cot_theta = np.cos(theta) / np.sin(theta) if np.sin(theta) > 1e-14 else 1e14

    numer = vt**2 * vr - c2 * (2 * vr + vt * cot_theta)

    dvr_dtheta = vt
    dvt_dtheta = numer / denom

    return [dvr_dtheta, dvt_dtheta]


# ── 参考值和测试案例 ────────────────────────────

# (M_inf, gamma, theta_c_deg) → 待打靶法求解的β
TEST_CASES = [
    {"M_inf": 2.0, "gamma": 1.4, "theta_c_deg": 20.0, "label": "M2_g1.4_tc20"},
    {"M_inf": 3.0, "gamma": 1.4, "theta_c_deg": 20.0, "label": "M3_g1.4_tc20"},
    {"M_inf": 2.0, "gamma": 1.4, "theta_c_deg": 30.0, "label": "M2_g1.4_tc30"},
    {"M_inf": 3.0, "gamma": 1.4, "theta_c_deg": 15.0, "label": "M3_g1.4_tc15"},
]
