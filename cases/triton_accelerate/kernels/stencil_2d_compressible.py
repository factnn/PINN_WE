"""
Fully-Fused Triton Kernels for 2D Compressible Euler Equations (Sod Shock Tube).
Physical domain is 1D Space (x) + 1D Time (t), exactly mapped to a 2D grid [Nt, Nx].
Data Layout: Contiguous Row-Major [Nt, Nx].
"""
import torch
import triton
import triton.language as tl

# ── FORWARD KERNEL ──────────────────────────────────────────────────────────
@triton.jit
def euler_fwd_kernel(
    Rho_ptr, M_ptr, E_ptr,
    resRho_ptr, resM_ptr, resE_ptr,
    Nt: tl.constexpr, Nx: tl.constexpr,
    dx: tl.constexpr, dt: tl.constexpr, gamma: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_x = tl.program_id(1)

    it = pid_t + 1
    ix = pid_x * BLOCK_X + tl.arange(0, BLOCK_X) + 1
    mask = (it < Nt - 1) & (ix < Nx - 1)

    stride = Nx
    base = it * stride + ix

    # 1. Load (Center, X-Neighbors, Time-Neighbors)
    r_c  = tl.load(Rho_ptr + base,          mask=mask)
    r_xp = tl.load(Rho_ptr + base + 1,      mask=mask)
    r_xm = tl.load(Rho_ptr + base - 1,      mask=mask)
    r_tp = tl.load(Rho_ptr + base + stride, mask=mask)
    r_tm = tl.load(Rho_ptr + base - stride, mask=mask)

    m_c  = tl.load(M_ptr + base,          mask=mask)
    m_xp = tl.load(M_ptr + base + 1,      mask=mask)
    m_xm = tl.load(M_ptr + base - 1,      mask=mask)
    m_tp = tl.load(M_ptr + base + stride, mask=mask)
    m_tm = tl.load(M_ptr + base - stride, mask=mask)

    e_c  = tl.load(E_ptr + base,          mask=mask)
    e_xp = tl.load(E_ptr + base + 1,      mask=mask)
    e_xm = tl.load(E_ptr + base - 1,      mask=mask)
    e_tp = tl.load(E_ptr + base + stride, mask=mask)
    e_tm = tl.load(E_ptr + base - stride, mask=mask)

    # 2. Physics & EOS
    u_c = m_c / r_c
    p_c = (gamma - 1.0) * (e_c - 0.5 * m_c * u_c)
    
    u_xp = m_xp / r_xp
    p_xp = (gamma - 1.0) * (e_xp - 0.5 * m_xp * u_xp)
    
    u_xm = m_xm / r_xm
    p_xm = (gamma - 1.0) * (e_xm - 0.5 * m_xm * u_xm)

    # 3. Derivatives
    r_t = (r_tp - r_tm) / (2.0 * dt)
    m_t = (m_tp - m_tm) / (2.0 * dt)
    e_t = (e_tp - e_tm) / (2.0 * dt)

    f1_x = (m_xp - m_xm) / (2.0 * dx)
    
    f2_c_xp = m_xp * u_xp + p_xp
    f2_c_xm = m_xm * u_xm + p_xm
    f2_x = (f2_c_xp - f2_c_xm) / (2.0 * dx)

    f3_c_xp = (e_xp + p_xp) * u_xp
    f3_c_xm = (e_xm + p_xm) * u_xm
    f3_x = (f3_c_xp - f3_c_xm) / (2.0 * dx)

    # 4. Residuals
    res_rho = r_t + f1_x
    res_m   = m_t + f2_x
    res_e   = e_t + f3_x

    # 5. Store to flat interior layout
    out_t = pid_t
    out_x = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    out_mask = (out_x < Nx - 2) & mask
    
    out_st = Nx - 2
    out_idx = out_t * out_st + out_x

    tl.store(resRho_ptr + out_idx, res_rho, mask=out_mask)
    tl.store(resM_ptr   + out_idx, res_m,   mask=out_mask)
    tl.store(resE_ptr   + out_idx, res_e,   mask=out_mask)


# ── BACKWARD (ADJOINT) KERNEL ───────────────────────────────────────────────
@triton.jit
def euler_bwd_kernel(
    Rho_ptr, M_ptr, E_ptr,
    Grho_ptr, Gm_ptr, Ge_ptr,
    gradRho_ptr, gradM_ptr, gradE_ptr,
    Nt: tl.constexpr, Nx: tl.constexpr,
    dx: tl.constexpr, dt: tl.constexpr, gamma: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_x = tl.program_id(1)

    it = pid_t + 1
    ix = pid_x * BLOCK_X + tl.arange(0, BLOCK_X) + 1
    mask = (it < Nt - 1) & (ix < Nx - 1)

    stride = Nx
    base = it * stride + ix

    # 1. Load ONLY Center States
    r_c = tl.load(Rho_ptr + base, mask=mask)
    m_c = tl.load(M_ptr + base, mask=mask)
    e_c = tl.load(E_ptr + base, mask=mask)

    u_c = m_c / r_c
    e_c_rho = e_c / r_c
    u2_c = u_c * u_c
    u3_c = u2_c * u_c

    # 2. Load Upstream Gradients (Time and X-Neighbors)
    out_st = Nx - 2
    g_idx = (it - 1) * out_st + (ix - 1)

    m_tp = (it + 1 < Nt - 1) & mask
    m_tm = (it - 1 >= 1) & mask
    m_xp = (ix + 1 < Nx - 1) & mask
    m_xm = (ix - 1 >= 1) & mask

    gr_tp = tl.load(Grho_ptr + g_idx + out_st, mask=m_tp, other=0.0)
    gr_tm = tl.load(Grho_ptr + g_idx - out_st, mask=m_tm, other=0.0)
    gm_tp = tl.load(Gm_ptr + g_idx + out_st,   mask=m_tp, other=0.0)
    gm_tm = tl.load(Gm_ptr + g_idx - out_st,   mask=m_tm, other=0.0)
    ge_tp = tl.load(Ge_ptr + g_idx + out_st,   mask=m_tp, other=0.0)
    ge_tm = tl.load(Ge_ptr + g_idx - out_st,   mask=m_tm, other=0.0)

    gr_xp = tl.load(Grho_ptr + g_idx + 1, mask=m_xp, other=0.0)
    gr_xm = tl.load(Grho_ptr + g_idx - 1, mask=m_xm, other=0.0)
    gm_xp = tl.load(Gm_ptr + g_idx + 1,   mask=m_xp, other=0.0)
    gm_xm = tl.load(Gm_ptr + g_idx - 1,   mask=m_xm, other=0.0)
    ge_xp = tl.load(Ge_ptr + g_idx + 1,   mask=m_xp, other=0.0)
    ge_xm = tl.load(Ge_ptr + g_idx - 1,   mask=m_xm, other=0.0)

    idx = 1.0 / (2.0 * dx)
    idt = 1.0 / (2.0 * dt)

    # 3. Exact Jacobian Multiplications (Evaluated at Center Point)
    
    # Gradient for RHO
    grad_rho = (gr_tm - gr_tp) * idt
    df2_drho_c = -0.5 * (3.0 - gamma) * u2_c
    df3_drho_c = -gamma * e_c_rho * u_c + (gamma - 1.0) * u3_c
    grad_rho += (gm_xm - gm_xp) * df2_drho_c * idx
    grad_rho += (ge_xm - ge_xp) * df3_drho_c * idx

    # Gradient for M
    grad_m = (gm_tm - gm_tp) * idt
    grad_m += (gr_xm - gr_xp) * idx  # df1_dm_c = 1.0
    df2_dm_c = (3.0 - gamma) * u_c
    df3_dm_c = gamma * e_c_rho - 1.5 * (gamma - 1.0) * u2_c
    grad_m += (gm_xm - gm_xp) * df2_dm_c * idx
    grad_m += (ge_xm - ge_xp) * df3_dm_c * idx

    # Gradient for E
    grad_e = (ge_tm - ge_tp) * idt
    grad_e += (gm_xm - gm_xp) * (gamma - 1.0) * idx
    df3_de_c = gamma * u_c
    grad_e += (ge_xm - ge_xp) * df3_de_c * idx

    tl.store(gradRho_ptr + base, grad_rho, mask=mask)
    tl.store(gradM_ptr   + base, grad_m,   mask=mask)
    tl.store(gradE_ptr   + base, grad_e,   mask=mask)


# ── BOUNDARY GRADIENT EXTENSION ─────────────────────────────────────────────
def _add_boundary_gradients_euler(U, M, E, Gr, Gm, Ge, grad_r, grad_m, grad_e, dx, dt, gamma):
    """Analytic boundary injection evaluated EXACTLY at boundary Jacobians."""
    Nt, Nx = U.shape
    idx = 1.0 / (2.0 * dx)
    idt = 1.0 / (2.0 * dt)
    
    # --- Time Boundaries ---
    grad_r[0, 1:-1] += -Gr[0, :] * idt
    grad_m[0, 1:-1] += -Gm[0, :] * idt
    grad_e[0, 1:-1] += -Ge[0, :] * idt

    grad_r[Nt-1, 1:-1] += Gr[Nt-3, :] * idt
    grad_m[Nt-1, 1:-1] += Gm[Nt-3, :] * idt
    grad_e[Nt-1, 1:-1] += Ge[Nt-3, :] * idt

    # --- X=0 Boundary ---
    u_0 = M[1:-1, 0] / U[1:-1, 0]
    df2_dr_0 = -0.5 * (3.0 - gamma) * (u_0**2)
    df3_dr_0 = -gamma * E[1:-1, 0] * u_0 / U[1:-1, 0] + (gamma - 1.0) * (u_0**3)
    df2_dm_0 = (3.0 - gamma) * u_0
    df3_dm_0 = gamma * E[1:-1, 0] / U[1:-1, 0] - 1.5 * (gamma - 1.0) * (u_0**2)
    df3_de_0 = gamma * u_0

    grad_m[1:-1, 0] += -Gr[:, 0] * idx
    grad_r[1:-1, 0] += -Gm[:, 0] * df2_dr_0 * idx
    grad_m[1:-1, 0] += -Gm[:, 0] * df2_dm_0 * idx
    grad_e[1:-1, 0] += -Gm[:, 0] * (gamma - 1.0) * idx
    grad_r[1:-1, 0] += -Ge[:, 0] * df3_dr_0 * idx
    grad_m[1:-1, 0] += -Ge[:, 0] * df3_dm_0 * idx
    grad_e[1:-1, 0] += -Ge[:, 0] * df3_de_0 * idx

    # --- X=Nx-1 Boundary ---
    u_n = M[1:-1, Nx-1] / U[1:-1, Nx-1]
    df2_dr_n = -0.5 * (3.0 - gamma) * (u_n**2)
    df3_dr_n = -gamma * E[1:-1, Nx-1] * u_n / U[1:-1, Nx-1] + (gamma - 1.0) * (u_n**3)
    df2_dm_n = (3.0 - gamma) * u_n
    df3_dm_n = gamma * E[1:-1, Nx-1] / U[1:-1, Nx-1] - 1.5 * (gamma - 1.0) * (u_n**2)
    df3_de_n = gamma * u_n

    grad_m[1:-1, Nx-1] += Gr[:, Nx-3] * idx
    grad_r[1:-1, Nx-1] += Gm[:, Nx-3] * df2_dr_n * idx
    grad_m[1:-1, Nx-1] += Gm[:, Nx-3] * df2_dm_n * idx
    grad_e[1:-1, Nx-1] += Gm[:, Nx-3] * (gamma - 1.0) * idx
    grad_r[1:-1, Nx-1] += Ge[:, Nx-3] * df3_dr_n * idx
    grad_m[1:-1, Nx-1] += Ge[:, Nx-3] * df3_dm_n * idx
    grad_e[1:-1, Nx-1] += Ge[:, Nx-3] * df3_de_n * idx


# ── PYTORCH AUTOGRAD WRAPPER BOX ─────────────────────────────────────────────
class _CompressibleEulerTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Rho, M, E, dx, dt, gamma):
        Rho, M, E = Rho.contiguous(), M.contiguous(), E.contiguous()
        Nt, Nx = Rho.shape
        N_total = (Nt - 2) * (Nx - 2)

        res_rho = torch.empty(N_total, device=Rho.device, dtype=Rho.dtype)
        res_m   = torch.empty_like(res_rho)
        res_e   = torch.empty_like(res_rho)

        BLOCK_X = 256
        grid = (Nt - 2, (Nx - 2 + BLOCK_X - 1) // BLOCK_X)
        
        euler_fwd_kernel[grid](
            Rho, M, E, res_rho, res_m, res_e,
            Nt, Nx, dx, dt, gamma, BLOCK_X
        )

        ctx.save_for_backward(Rho, M, E)
        ctx.res_rho, ctx.res_m, ctx.res_e = res_rho, res_m, res_e
        ctx.dx, ctx.dt, ctx.gamma = dx, dt, gamma

        return (res_rho**2 + res_m**2 + res_e**2).mean()

    @staticmethod
    def backward(ctx, grad_output):
        Rho, M, E = ctx.saved_tensors
        res_rho, res_m, res_e = ctx.res_rho, ctx.res_m, ctx.res_e
        dx, dt, gamma = ctx.dx, ctx.dt, ctx.gamma
        Nt, Nx = Rho.shape
        N_total = res_rho.numel()

        scale = (2.0 / N_total) * grad_output
        Gr_flat, Gm_flat, Ge_flat = res_rho * scale, res_m * scale, res_e * scale

        grad_rho = torch.zeros_like(Rho)
        grad_m   = torch.zeros_like(M)
        grad_e   = torch.zeros_like(E)

        BLOCK_X = 256
        grid = (Nt - 2, (Nx - 2 + BLOCK_X - 1) // BLOCK_X)
        
        euler_bwd_kernel[grid](
            Rho, M, E, Gr_flat, Gm_flat, Ge_flat,
            grad_rho, grad_m, grad_e,
            Nt, Nx, dx, dt, gamma, BLOCK_X
        )

        # Reshape for boundary fixes
        Gr = Gr_flat.reshape(Nt - 2, Nx - 2)
        Gm = Gm_flat.reshape(Nt - 2, Nx - 2)
        Ge = Ge_flat.reshape(Nt - 2, Nx - 2)

        _add_boundary_gradients_euler(Rho, M, E, Gr, Gm, Ge, grad_rho, grad_m, grad_e, dx, dt, gamma)

        return grad_rho, grad_m, grad_e, None, None, None


def compressible_sod_residual_triton(Rho, M, E, dx, dt, gamma=1.4):
    return _CompressibleEulerTriton.apply(Rho, M, E, dx, dt, gamma)


# ── UNIT GRADIENT CONVERGENCE TESTS ─────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np
    
    Nt, Nx = 10, 64
    gamma_val = 1.4
    device = "cuda"
    dtype = torch.float64

    t = torch.linspace(0, 1, Nt, device=device, dtype=dtype)
    x = torch.linspace(0, 1, Nx, device=device, dtype=dtype)
    dt_v = float(t[1] - t[0]); dx_v = float(x[1] - x[0])
    T, X = torch.meshgrid(t, x, indexing='ij')

    Rho_init = torch.where(X < 0.5, 1.0, 0.125)
    P_init   = torch.where(X < 0.5, 1.0, 0.1)
    U_init   = torch.zeros_like(X)
    M_init   = Rho_init * U_init
    E_init   = P_init / (gamma_val - 1.0) + 0.5 * Rho_init * (U_init**2)

    R_pt = Rho_init.clone().requires_grad_(True)
    M_pt = M_init.clone().requires_grad_(True)
    E_pt = E_init.clone().requires_grad_(True)

    R_tr = Rho_init.clone().requires_grad_(True)
    M_tr = M_init.clone().requires_grad_(True)
    E_tr = E_init.clone().requires_grad_(True)

    # --- 1. PyTorch Autograd ---
    r_t = (R_pt[2:, 1:-1] - R_pt[:-2, 1:-1]) / (2.0 * dt_v)
    m_t = (M_pt[2:, 1:-1] - M_pt[:-2, 1:-1]) / (2.0 * dt_v)
    e_t = (E_pt[2:, 1:-1] - E_pt[:-2, 1:-1]) / (2.0 * dt_v)

    u_c = M_pt[1:-1, 1:-1] / R_pt[1:-1, 1:-1]
    p_c = (gamma_val - 1.0) * (E_pt[1:-1, 1:-1] - 0.5 * M_pt[1:-1, 1:-1] * u_c)

    u_xp = M_pt[1:-1, 2:] / R_pt[1:-1, 2:]
    p_xp = (gamma_val - 1.0) * (E_pt[1:-1, 2:] - 0.5 * M_pt[1:-1, 2:] * u_xp)

    u_xm = M_pt[1:-1, :-2] / R_pt[1:-1, :-2]
    p_xm = (gamma_val - 1.0) * (E_pt[1:-1, :-2] - 0.5 * M_pt[1:-1, :-2] * u_xm)

    f1_x = (M_pt[1:-1, 2:] - M_pt[1:-1, :-2]) / (2.0 * dx_v)
    f2_x = ((M_pt[1:-1, 2:] * u_xp + p_xp) - (M_pt[1:-1, :-2] * u_xm + p_xm)) / (2.0 * dx_v)
    f3_x = (((E_pt[1:-1, 2:] + p_xp) * u_xp) - ((E_pt[1:-1, :-2] + p_xm) * u_xm)) / (2.0 * dx_v)

    loss_pt = ((r_t + f1_x)**2 + (m_t + f2_x)**2 + (e_t + f3_x)**2).mean()
    loss_pt.backward()

    # --- 2. Fused Triton ---
    loss_tr = compressible_sod_residual_triton(R_tr, M_tr, E_tr, dx_v, dt_v, gamma_val)
    loss_tr.backward()

    # --- 3. Diagnostics ---
    print("=" * 70)
    print("🚀 2D COMPRESSIBLE EULER (SOD SHOCK TUBE) DIAGNOSTICS")
    print("=" * 70)
    print(f"▶ Forward Loss Convergence Error : {abs(loss_pt.item() - loss_tr.item()):.2e}")
    
    err_R = (R_pt.grad - R_tr.grad).abs().max().item()
    err_M = (M_pt.grad - M_tr.grad).abs().max().item()
    err_E = (E_pt.grad - E_tr.grad).abs().max().item()
    
    print(f"▶ Adjoint Backprop Max Error (Rho) : {err_R:.2e}")
    print(f"▶ Adjoint Backprop Max Error (M)   : {err_M:.2e}")
    print(f"▶ Adjoint Backprop Max Error (E)   : {err_E:.2e}")
    print("-" * 70)
    if max(err_R, err_M, err_E) < 1e-10:
        print("🎉 [STATUS: PASS] Strict Conservative Form Adjoints Matched!")
    else:
        print("⚠️ [STATUS: FAIL] Jacobian evaluation mismatch.")