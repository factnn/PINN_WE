"""3D steady-state NS Triton kernel for LDC 3D.
u*u_x + v*u_y + w*u_z + p_x = nu*(u_xx + u_yy + u_zz), same for v, w.
div = u_x + v_y + w_z = 0.
Data: U, V, W, P: [Nx, Ny, Nz] contiguous, row-major (z fastest).
Stencil: 7-point in 3D — center + 6 neighbors (x±1, y±1, z±1).
"""
import torch
import triton
import triton.language as tl


# ── Forward kernel ─────────────────────────────────────────────────────────
@triton.jit
def ns3d_fwd_kernel(
    U_ptr, V_ptr, W_ptr, P_ptr,
    resU_ptr, resV_ptr, resW_ptr, resDiv_ptr,
    Nx: tl.constexpr, Ny: tl.constexpr, Nz: tl.constexpr,
    dx: tl.constexpr, dy: tl.constexpr, dz: tl.constexpr,
    nu: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Each program handles BLOCK interior points.
    Interior domain: x in [1, Nx-2], y in [1, Ny-2], z in [1, Nz-2].
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)

    Ni_nx = Nx - 2
    Ni_ny = Ny - 2
    Ni_nz = Nz - 2

    ix = idx // (Ni_ny * Ni_nz) + 1
    rem = idx % (Ni_ny * Ni_nz)
    iy = rem // Ni_nz + 1
    iz = rem % Ni_nz + 1

    mask = (ix < Nx - 1) & (iy < Ny - 1) & (iz < Nz - 1)

    sx = Ny * Nz
    sy = Nz

    base = ix * sx + iy * sy + iz

    # Load 7-point stencil for U
    u_c  = tl.load(U_ptr + base,          mask=mask)
    u_xp = tl.load(U_ptr + base + sx,     mask=mask)
    u_xm = tl.load(U_ptr + base - sx,     mask=mask)
    u_yp = tl.load(U_ptr + base + sy,     mask=mask)
    u_ym = tl.load(U_ptr + base - sy,     mask=mask)
    u_zp = tl.load(U_ptr + base + 1,      mask=mask)
    u_zm = tl.load(U_ptr + base - 1,      mask=mask)

    # Load 7-point stencil for V
    v_c  = tl.load(V_ptr + base,          mask=mask)
    v_xp = tl.load(V_ptr + base + sx,     mask=mask)
    v_xm = tl.load(V_ptr + base - sx,     mask=mask)
    v_yp = tl.load(V_ptr + base + sy,     mask=mask)
    v_ym = tl.load(V_ptr + base - sy,     mask=mask)
    v_zp = tl.load(V_ptr + base + 1,      mask=mask)
    v_zm = tl.load(V_ptr + base - 1,      mask=mask)

    # Load 7-point stencil for W
    w_c  = tl.load(W_ptr + base,          mask=mask)
    w_xp = tl.load(W_ptr + base + sx,     mask=mask)
    w_xm = tl.load(W_ptr + base - sx,     mask=mask)
    w_yp = tl.load(W_ptr + base + sy,     mask=mask)
    w_ym = tl.load(W_ptr + base - sy,     mask=mask)
    w_zp = tl.load(W_ptr + base + 1,      mask=mask)
    w_zm = tl.load(W_ptr + base - 1,      mask=mask)

    # Load 7-point stencil for P
    p_xp = tl.load(P_ptr + base + sx,     mask=mask)
    p_xm = tl.load(P_ptr + base - sx,     mask=mask)
    p_yp = tl.load(P_ptr + base + sy,     mask=mask)
    p_ym = tl.load(P_ptr + base - sy,     mask=mask)
    p_zp = tl.load(P_ptr + base + 1,      mask=mask)
    p_zm = tl.load(P_ptr + base - 1,      mask=mask)

    # U derivatives
    u_x  = (u_xp - u_xm) / (2.0 * dx)
    u_y  = (u_yp - u_ym) / (2.0 * dy)
    u_z  = (u_zp - u_zm) / (2.0 * dz)
    u_xx = (u_xp - 2.0*u_c + u_xm) / (dx*dx)
    u_yy = (u_yp - 2.0*u_c + u_ym) / (dy*dy)
    u_zz = (u_zp - 2.0*u_c + u_zm) / (dz*dz)

    # V derivatives
    v_x  = (v_xp - v_xm) / (2.0 * dx)
    v_y  = (v_yp - v_ym) / (2.0 * dy)
    v_z  = (v_zp - v_zm) / (2.0 * dz)
    v_xx = (v_xp - 2.0*v_c + v_xm) / (dx*dx)
    v_yy = (v_yp - 2.0*v_c + v_ym) / (dy*dy)
    v_zz = (v_zp - 2.0*v_c + v_zm) / (dz*dz)

    # W derivatives
    w_x  = (w_xp - w_xm) / (2.0 * dx)
    w_y  = (w_yp - w_ym) / (2.0 * dy)
    w_z  = (w_zp - w_zm) / (2.0 * dz)
    w_xx = (w_xp - 2.0*w_c + w_xm) / (dx*dx)
    w_yy = (w_yp - 2.0*w_c + w_ym) / (dy*dy)
    w_zz = (w_zp - 2.0*w_c + w_zm) / (dz*dz)

    # Pressure gradient
    p_x = (p_xp - p_xm) / (2.0 * dx)
    p_y = (p_yp - p_ym) / (2.0 * dy)
    p_z = (p_zp - p_zm) / (2.0 * dz)

    # Steady NS residuals (no time derivative)
    res_u   = u_c*u_x + v_c*u_y + w_c*u_z + p_x - nu*(u_xx + u_yy + u_zz)
    res_v   = u_c*v_x + v_c*v_y + w_c*v_z + p_y - nu*(v_xx + v_yy + v_zz)
    res_w   = u_c*w_x + v_c*w_y + w_c*w_z + p_z - nu*(w_xx + w_yy + w_zz)
    res_div = u_x + v_y + w_z

    # Store to output
    tl.store(resU_ptr   + pid * BLOCK + tl.arange(0, BLOCK), res_u,   mask=mask)
    tl.store(resV_ptr   + pid * BLOCK + tl.arange(0, BLOCK), res_v,   mask=mask)
    tl.store(resW_ptr   + pid * BLOCK + tl.arange(0, BLOCK), res_w,   mask=mask)
    tl.store(resDiv_ptr + pid * BLOCK + tl.arange(0, BLOCK), res_div, mask=mask)


# ── Backward (adjoint) kernel ──────────────────────────────────────────────
@triton.jit
def ns3d_bwd_kernel(
    U_ptr, V_ptr, W_ptr,
    Gu_ptr, Gv_ptr, Gw_ptr, Gdiv_ptr,
    gradU_ptr, gradV_ptr, gradW_ptr, gradP_ptr,
    Nx: tl.constexpr, Ny: tl.constexpr, Nz: tl.constexpr,
    dx: tl.constexpr, dy: tl.constexpr, dz: tl.constexpr,
    nu: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Adjoint of the steady 3D NS stencil operator.
    Computes grad_U, grad_V, grad_W, grad_P from upstream gradients Gu, Gv, Gw, Gdiv.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)

    Ni_nx = Nx - 2; Ni_ny = Ny - 2; Ni_nz = Nz - 2
    Ni_vol = Ni_nx * Ni_ny * Ni_nz

    ix = idx // (Ni_ny * Ni_nz) + 1
    rem = idx % (Ni_ny * Ni_nz)
    iy = rem // Ni_nz + 1
    iz = rem % Ni_nz + 1

    mask = (ix < Nx - 1) & (iy < Ny - 1) & (iz < Nz - 1)

    sx = Ny * Nz; sy = Nz
    base = ix * sx + iy * sy + iz

    # Load U, V, W stencils (needed for nonlinear adjoint terms)
    u_c  = tl.load(U_ptr + base,          mask=mask)
    u_xp = tl.load(U_ptr + base + sx,     mask=mask)
    u_xm = tl.load(U_ptr + base - sx,     mask=mask)
    u_yp = tl.load(U_ptr + base + sy,     mask=mask)
    u_ym = tl.load(U_ptr + base - sy,     mask=mask)
    u_zp = tl.load(U_ptr + base + 1,      mask=mask)
    u_zm = tl.load(U_ptr + base - 1,      mask=mask)

    v_c  = tl.load(V_ptr + base,          mask=mask)
    v_xp = tl.load(V_ptr + base + sx,     mask=mask)
    v_xm = tl.load(V_ptr + base - sx,     mask=mask)
    v_yp = tl.load(V_ptr + base + sy,     mask=mask)
    v_ym = tl.load(V_ptr + base - sy,     mask=mask)
    v_zp = tl.load(V_ptr + base + 1,      mask=mask)
    v_zm = tl.load(V_ptr + base - 1,      mask=mask)

    w_c  = tl.load(W_ptr + base,          mask=mask)
    w_xp = tl.load(W_ptr + base + sx,     mask=mask)
    w_xm = tl.load(W_ptr + base - sx,     mask=mask)
    w_yp = tl.load(W_ptr + base + sy,     mask=mask)
    w_ym = tl.load(W_ptr + base - sy,     mask=mask)
    w_zp = tl.load(W_ptr + base + 1,      mask=mask)
    w_zm = tl.load(W_ptr + base - 1,      mask=mask)

    # Flat interior grid indices
    xg = ix - 1; yg = iy - 1; zg = iz - 1
    bg = xg * Ni_ny * Ni_nz + yg * Ni_nz + zg

    # Neighbor existence masks
    m_xp = (xg + 1 < Nx - 2) & mask
    m_xm = (xg - 1 >= 0) & mask
    m_yp = (yg + 1 < Ny - 2) & mask
    m_ym = (yg - 1 >= 0) & mask
    m_zp = (zg + 1 < Nz - 2) & mask
    m_zm = (zg - 1 >= 0) & mask

    # Load G neighbors
    gu_c  = tl.load(Gu_ptr + bg,               mask=mask, other=0.0)
    gv_c  = tl.load(Gv_ptr + bg,               mask=mask, other=0.0)
    gw_c  = tl.load(Gw_ptr + bg,               mask=mask, other=0.0)
    gd_c  = tl.load(Gdiv_ptr + bg,             mask=mask, other=0.0)

    gu_xp = tl.load(Gu_ptr + bg + Ni_ny*Ni_nz, mask=m_xp, other=0.0)
    gu_xm = tl.load(Gu_ptr + bg - Ni_ny*Ni_nz, mask=m_xm, other=0.0)
    gu_yp = tl.load(Gu_ptr + bg + Ni_nz,       mask=m_yp, other=0.0)
    gu_ym = tl.load(Gu_ptr + bg - Ni_nz,       mask=m_ym, other=0.0)
    gu_zp = tl.load(Gu_ptr + bg + 1,           mask=m_zp, other=0.0)
    gu_zm = tl.load(Gu_ptr + bg - 1,           mask=m_zm, other=0.0)

    gv_xp = tl.load(Gv_ptr + bg + Ni_ny*Ni_nz, mask=m_xp, other=0.0)
    gv_xm = tl.load(Gv_ptr + bg - Ni_ny*Ni_nz, mask=m_xm, other=0.0)
    gv_yp = tl.load(Gv_ptr + bg + Ni_nz,       mask=m_yp, other=0.0)
    gv_ym = tl.load(Gv_ptr + bg - Ni_nz,       mask=m_ym, other=0.0)
    gv_zp = tl.load(Gv_ptr + bg + 1,           mask=m_zp, other=0.0)
    gv_zm = tl.load(Gv_ptr + bg - 1,           mask=m_zm, other=0.0)

    gw_xp = tl.load(Gw_ptr + bg + Ni_ny*Ni_nz, mask=m_xp, other=0.0)
    gw_xm = tl.load(Gw_ptr + bg - Ni_ny*Ni_nz, mask=m_xm, other=0.0)
    gw_yp = tl.load(Gw_ptr + bg + Ni_nz,       mask=m_yp, other=0.0)
    gw_ym = tl.load(Gw_ptr + bg - Ni_nz,       mask=m_ym, other=0.0)
    gw_zp = tl.load(Gw_ptr + bg + 1,           mask=m_zp, other=0.0)
    gw_zm = tl.load(Gw_ptr + bg - 1,           mask=m_zm, other=0.0)

    gd_xp = tl.load(Gdiv_ptr + bg + Ni_ny*Ni_nz, mask=m_xp, other=0.0)
    gd_xm = tl.load(Gdiv_ptr + bg - Ni_ny*Ni_nz, mask=m_xm, other=0.0)
    gd_yp = tl.load(Gdiv_ptr + bg + Ni_nz,     mask=m_yp, other=0.0)
    gd_ym = tl.load(Gdiv_ptr + bg - Ni_nz,     mask=m_ym, other=0.0)
    gd_zp = tl.load(Gdiv_ptr + bg + 1,         mask=m_zp, other=0.0)
    gd_zm = tl.load(Gdiv_ptr + bg - 1,         mask=m_zm, other=0.0)

    idx_val = 1.0/(2.0*dx); idy = 1.0/(2.0*dy); idz = 1.0/(2.0*dz)
    idx2 = 1.0/(dx*dx); idy2 = 1.0/(dy*dy); idz2 = 1.0/(dz*dz)

    # ── grad_U ────────────────────────────────────────────────────────────
    # Center contributions
    grad_u  = gu_c * ((u_xp - u_xm) * idx_val + 2.0*nu*(idx2 + idy2 + idz2))
    grad_u += gv_c * (v_xp - v_xm) * idx_val   # u_c * v_x in res_v
    grad_u += gw_c * (w_xp - w_xm) * idx_val   # u_c * w_x in res_w
    # Neighbor nonlinear terms
    grad_u += (gu_xm * u_xm - gu_xp * u_xp) * idx_val    # u in u_x neighbors
    grad_u += (gu_ym * v_ym - gu_yp * v_yp) * idy        # u in u_y neighbors (res_u: v_c*u_y)
    grad_u += (gu_zm * w_zm - gu_zp * w_zp) * idz        # u in u_z neighbors (res_u: w_c*u_z)
    # Div adjoint
    grad_u += (gd_xm - gd_xp) * idx_val
    # Diffusion neighbor adjoint
    grad_u -= nu * ((gu_xp + gu_xm) * idx2 + (gu_yp + gu_ym) * idy2 + (gu_zp + gu_zm) * idz2)

    # ── grad_V ────────────────────────────────────────────────────────────
    grad_v  = gu_c * (u_yp - u_ym) * idy                  # v_c * u_y in res_u
    grad_v += gv_c * ((v_yp - v_ym) * idy + 2.0*nu*(idx2 + idy2 + idz2))
    grad_v += gw_c * (w_yp - w_ym) * idy                  # v_c * w_y in res_w
    grad_v += (gv_xm * u_xm - gv_xp * u_xp) * idx_val    # v in v_x neighbors (u_c from U)
    grad_v += (gv_ym * v_ym - gv_yp * v_yp) * idy
    grad_v += (gv_zm * w_zm - gv_zp * w_zp) * idz
    grad_v += (gd_ym - gd_yp) * idy
    grad_v -= nu * ((gv_xp + gv_xm) * idx2 + (gv_yp + gv_ym) * idy2 + (gv_zp + gv_zm) * idz2)

    # ── grad_W ────────────────────────────────────────────────────────────
    grad_w  = gu_c * (u_zp - u_zm) * idz                  # w_c * u_z in res_u
    grad_w += gv_c * (v_zp - v_zm) * idz                  # w_c * v_z in res_v
    grad_w += gw_c * ((w_zp - w_zm) * idz + 2.0*nu*(idx2 + idy2 + idz2))
    grad_w += (gw_xm * u_xm - gw_xp * u_xp) * idx_val    # w in w_x neighbors
    grad_w += (gw_ym * v_ym - gw_yp * v_yp) * idy
    grad_w += (gw_zm * w_zm - gw_zp * w_zp) * idz
    grad_w += (gd_zm - gd_zp) * idz
    grad_w -= nu * ((gw_xp + gw_xm) * idx2 + (gw_yp + gw_ym) * idy2 + (gw_zp + gw_zm) * idz2)

    # ── grad_P ────────────────────────────────────────────────────────────
    grad_p  = (gu_xm - gu_xp) * idx_val
    grad_p += (gv_ym - gv_yp) * idy
    grad_p += (gw_zm - gw_zp) * idz

    # Store gradients
    tl.store(gradU_ptr + base, grad_u, mask=mask)
    tl.store(gradV_ptr + base, grad_v, mask=mask)
    tl.store(gradW_ptr + base, grad_w, mask=mask)
    tl.store(gradP_ptr + base, grad_p, mask=mask)


# ── Boundary gradient fix ──────────────────────────────────────────────────
def _add_boundary_gradients_3d(U, V, W, Gu, Gv, Gw, Gdiv, grad_u, grad_v, grad_w, grad_p,
                                dx, dy, dz, nu_val):
    """Add gradient contributions for 6 boundary faces not covered by ns3d_bwd_kernel.
    U, V, W, P: [Nx, Ny, Nz] (3D, no time dim).
    Gu, Gv, Gw, Gdiv: [Nx-2, Ny-2, Nz-2].
    """
    Nx, Ny, Nz = U.shape
    inv_2dx = 1.0/(2.0*dx); inv_2dy = 1.0/(2.0*dy); inv_2dz = 1.0/(2.0*dz)
    inv_dx2 = 1.0/(dx*dx); inv_dy2 = 1.0/(dy*dy); inv_dz2 = 1.0/(dz*dz)

    i = slice(1, -1)

    # --- x=0 ---
    grad_u[0, i, i] += -Gu[0] * (U[1, i, i] * inv_2dx + nu_val * inv_dx2)
    grad_v[0, i, i] += -Gv[0] * (U[1, i, i] * inv_2dx + nu_val * inv_dx2)  # u_c from U!
    grad_w[0, i, i] += -Gw[0] * (U[1, i, i] * inv_2dx + nu_val * inv_dx2)
    grad_u[0, i, i] += -Gdiv[0] * inv_2dx  # div: -Gdiv/(2dx) for u_l

    # --- x=Nx-1 ---
    grad_u[Nx-1, i, i] += Gu[Nx-3] * (U[Nx-2, i, i] * inv_2dx - nu_val * inv_dx2)
    grad_v[Nx-1, i, i] += Gv[Nx-3] * (U[Nx-2, i, i] * inv_2dx - nu_val * inv_dx2)
    grad_w[Nx-1, i, i] += Gw[Nx-3] * (U[Nx-2, i, i] * inv_2dx - nu_val * inv_dx2)
    grad_u[Nx-1, i, i] += Gdiv[Nx-3] * inv_2dx  # div: +Gdiv/(2dx) for u_r

    # --- y=0 ---
    grad_u[i, 0, i] += -Gu[:, 0] * (V[i, 1, i] * inv_2dy + nu_val * inv_dy2)
    grad_v[i, 0, i] += -Gv[:, 0] * (V[i, 1, i] * inv_2dy + nu_val * inv_dy2)
    grad_w[i, 0, i] += -Gw[:, 0] * (V[i, 1, i] * inv_2dy + nu_val * inv_dy2)
    grad_v[i, 0, i] += -Gdiv[:, 0] * inv_2dy

    # --- y=Ny-1 ---
    grad_u[i, Ny-1, i] += Gu[:, Ny-3] * (V[i, Ny-2, i] * inv_2dy - nu_val * inv_dy2)
    grad_v[i, Ny-1, i] += Gv[:, Ny-3] * (V[i, Ny-2, i] * inv_2dy - nu_val * inv_dy2)
    grad_w[i, Ny-1, i] += Gw[:, Ny-3] * (V[i, Ny-2, i] * inv_2dy - nu_val * inv_dy2)
    grad_v[i, Ny-1, i] += Gdiv[:, Ny-3] * inv_2dy

    # --- z=0 ---
    grad_u[i, i, 0] += -Gu[:, :, 0] * (W[i, i, 1] * inv_2dz + nu_val * inv_dz2)
    grad_v[i, i, 0] += -Gv[:, :, 0] * (W[i, i, 1] * inv_2dz + nu_val * inv_dz2)
    grad_w[i, i, 0] += -Gw[:, :, 0] * (W[i, i, 1] * inv_2dz + nu_val * inv_dz2)
    grad_w[i, i, 0] += -Gdiv[:, :, 0] * inv_2dz

    # --- z=Nz-1 ---
    grad_u[i, i, Nz-1] += Gu[:, :, Nz-3] * (W[i, i, Nz-2] * inv_2dz - nu_val * inv_dz2)
    grad_v[i, i, Nz-1] += Gv[:, :, Nz-3] * (W[i, i, Nz-2] * inv_2dz - nu_val * inv_dz2)
    grad_w[i, i, Nz-1] += Gw[:, :, Nz-3] * (W[i, i, Nz-2] * inv_2dz - nu_val * inv_dz2)
    grad_w[i, i, Nz-1] += Gdiv[:, :, Nz-3] * inv_2dz

    # --- P boundaries ---
    grad_p[0, i, i] += -Gu[0] * inv_2dx
    grad_p[Nx-1, i, i] += Gu[Nx-3] * inv_2dx
    grad_p[i, 0, i] += -Gv[:, 0] * inv_2dy
    grad_p[i, Ny-1, i] += Gv[:, Ny-3] * inv_2dy
    grad_p[i, i, 0] += -Gw[:, :, 0] * inv_2dz
    grad_p[i, i, Nz-1] += Gw[:, :, Nz-3] * inv_2dz


# ── Autograd Function ──────────────────────────────────────────────────────
class _NS3DSteadyTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, V, W, P, dx, dy, dz, nu_val):
        U, V, W, P = U.contiguous(), V.contiguous(), W.contiguous(), P.contiguous()
        Nx, Ny, Nz = U.shape
        Ni_nx, Ni_ny, Ni_nz = Nx - 2, Ny - 2, Nz - 2
        N_total = Ni_nx * Ni_ny * Ni_nz

        res_u   = torch.empty(N_total, device=U.device, dtype=U.dtype)
        res_v   = torch.empty_like(res_u)
        res_w   = torch.empty_like(res_u)
        res_div = torch.empty_like(res_u)

        BLOCK = 256
        grid = ((N_total + BLOCK - 1) // BLOCK,)
        ns3d_fwd_kernel[grid](U, V, W, P, res_u, res_v, res_w, res_div,
                              Nx, Ny, Nz, dx, dy, dz, nu_val, BLOCK)

        ctx.save_for_backward(U, V, W)
        ctx.res_u, ctx.res_v, ctx.res_w, ctx.res_div = res_u, res_v, res_w, res_div
        ctx.dx, ctx.dy, ctx.dz, ctx.nu_val = dx, dy, dz, nu_val
        return (res_u**2 + res_v**2 + res_w**2 + res_div**2).mean()

    @staticmethod
    def backward(ctx, grad_out):
        U, V, W = ctx.saved_tensors
        res_u, res_v, res_w, res_div = ctx.res_u, ctx.res_v, ctx.res_w, ctx.res_div
        dx, dy, dz, nu_val = ctx.dx, ctx.dy, ctx.dz, ctx.nu_val
        Nx, Ny, Nz = U.shape
        N_total = res_u.numel()

        scale = (2.0 / N_total) * grad_out
        Gu   = res_u * scale
        Gv   = res_v * scale
        Gw   = res_w * scale
        Gdiv = res_div * scale

        grad_u = torch.zeros_like(U)
        grad_v = torch.zeros_like(V)
        grad_w = torch.zeros_like(W)
        grad_p = torch.zeros_like(U)

        BLOCK = 256
        grid = ((N_total + BLOCK - 1) // BLOCK,)
        ns3d_bwd_kernel[grid](U, V, W, Gu, Gv, Gw, Gdiv,
                              grad_u, grad_v, grad_w, grad_p,
                              Nx, Ny, Nz, dx, dy, dz, nu_val, BLOCK)

        # Reshape G values from flat to 3D for boundary slicing
        Ni_ny, Ni_nz = Ny - 2, Nz - 2
        Gu_3d   = Gu.reshape(Nx - 2, Ni_ny, Ni_nz)
        Gv_3d   = Gv.reshape(Nx - 2, Ni_ny, Ni_nz)
        Gw_3d   = Gw.reshape(Nx - 2, Ni_ny, Ni_nz)
        Gdiv_3d = Gdiv.reshape(Nx - 2, Ni_ny, Ni_nz)

        _add_boundary_gradients_3d(U, V, W, Gu_3d, Gv_3d, Gw_3d, Gdiv_3d,
                                    grad_u, grad_v, grad_w, grad_p,
                                    dx, dy, dz, nu_val)

        return grad_u, grad_v, grad_w, grad_p, None, None, None, None


def ldc_residual_triton(U, V, W, P, dx, dy, dz, nu_val):
    return _NS3DSteadyTriton.apply(U.contiguous(), V.contiguous(), W.contiguous(), P.contiguous(),
                                   dx, dy, dz, nu_val)


# ── Correctness check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, numpy as np, os
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

    os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/triton_stencil3d_steady_test')

    from cases.ldc_3d.physics import pde_residual_pytorch

    Nx, Ny, Nz = 32, 32, 32
    nu_val = 1.0 / 100.0  # Re=100
    device = "cuda"
    dtype = torch.float64

    x = torch.linspace(0, 1, Nx, device=device)
    y = torch.linspace(0, 1, Ny, device=device)
    z = torch.linspace(0, 1, Nz, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    dx = float(x[1] - x[0]); dy = float(y[1] - y[0]); dz = float(z[1] - z[0])

    def make_uvwp(rg=False):
        U = (torch.sin(X)*torch.cos(Y)*torch.cos(Z)).to(dtype)
        V = (-torch.cos(X)*torch.sin(Y)*torch.cos(Z)).to(dtype)
        W = (0.5*torch.sin(X)*torch.sin(Y)*torch.cos(Z)).to(dtype)
        P = (0.25*(torch.cos(2*X)+torch.cos(2*Y))*(torch.cos(2*Z)+2)).to(dtype)
        return (U.requires_grad_(rg), V.requires_grad_(rg),
                W.requires_grad_(rg), P.requires_grad_(rg))

    # Forward check
    print("1. Forward check...")
    U_pt, V_pt, W_pt, P_pt = make_uvwp(True)
    U_tr, V_tr, W_tr, P_tr = make_uvwp(True)
    loss_pt = pde_residual_pytorch(U_pt.float(), V_pt.float(), W_pt.float(), P_pt.float(), dx, dy, dz)
    loss_tr = ldc_residual_triton(U_tr.float(), V_tr.float(), W_tr.float(), P_tr.float(), dx, dy, dz, nu_val)
    print(f"  Forward rel_err: {abs(loss_pt.item()-loss_tr.item())/(abs(loss_pt.item())+1e-12):.2e}")

    # Gradient check
    print("2. Gradient check (float64)...")
    U_pt2, V_pt2, W_pt2, P_pt2 = make_uvwp(True)
    U_tr2, V_tr2, W_tr2, P_tr2 = make_uvwp(True)
    loss_pt_d = pde_residual_pytorch(U_pt2, V_pt2, W_pt2, P_pt2, dx, dy, dz)
    loss_tr_d = ldc_residual_triton(U_tr2, V_tr2, W_tr2, P_tr2, dx, dy, dz, nu_val)
    loss_pt_d.backward(); loss_tr_d.backward()

    eU = (U_pt2.grad - U_tr2.grad).abs().max().item()
    eV = (V_pt2.grad - V_tr2.grad).abs().max().item()
    eW = (W_pt2.grad - W_tr2.grad).abs().max().item()
    eP = (P_pt2.grad - P_tr2.grad).abs().max().item()
    print(f"  Grad U err: {eU:.2e}")
    print(f"  Grad V err: {eV:.2e}")
    print(f"  Grad W err: {eW:.2e}")
    print(f"  Grad P err: {eP:.2e}")
    print(f"  {'ALL PASS' if max(eU,eV,eW,eP) < 1e-5 else 'FAIL'}")

    # MLP field gradient check (float32, like real training)
    print("3. MLP field gradient check (float32)...")
    from cases.ldc_3d.physics import MLP

    model = MLP().to(device)
    xyz = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1).float().to(device)

    # Get MLP output fields (detached for independent grad tests)
    with torch.no_grad():
        u0, v0, w0, p0 = model(xyz)
        U0 = u0.reshape(Nx, Ny, Nz).clone()
        V0 = v0.reshape(Nx, Ny, Nz).clone()
        W0 = w0.reshape(Nx, Ny, Nz).clone()
        P0 = p0.reshape(Nx, Ny, Nz).clone()

    # PyTorch autograd
    U_pt = U0.detach().requires_grad_(True)
    V_pt = V0.detach().requires_grad_(True)
    W_pt = W0.detach().requires_grad_(True)
    P_pt = P0.detach().requires_grad_(True)
    loss_pt2 = pde_residual_pytorch(U_pt, V_pt, W_pt, P_pt, dx, dy, dz)
    loss_pt2.backward()

    # Triton
    U_tr = U0.detach().requires_grad_(True)
    V_tr = V0.detach().requires_grad_(True)
    W_tr = W0.detach().requires_grad_(True)
    P_tr = P0.detach().requires_grad_(True)
    loss_tr2 = ldc_residual_triton(U_tr, V_tr, W_tr, P_tr, dx, dy, dz, nu_val)
    loss_tr2.backward()

    eU_mlp = (U_tr.grad - U_pt.grad).abs().max().item()
    eV_mlp = (V_tr.grad - V_pt.grad).abs().max().item()
    eW_mlp = (W_tr.grad - W_pt.grad).abs().max().item()
    eP_mlp = (P_tr.grad - P_pt.grad).abs().max().item()
    print(f"  MLP Grad U err: {eU_mlp:.2e}")
    print(f"  MLP Grad V err: {eV_mlp:.2e}")
    print(f"  MLP Grad W err: {eW_mlp:.2e}")
    print(f"  MLP Grad P err: {eP_mlp:.2e}")
    print(f"  {'ALL PASS' if max(eU_mlp,eV_mlp,eW_mlp,eP_mlp) < 1e-5 else 'FAIL'}")

    # Benchmark
    print("4. Benchmark...")
    U_b, V_b, W_b, P_b = make_uvwp()
    U_b, V_b, W_b, P_b = U_b.float(), V_b.float(), W_b.float(), P_b.float()
    ms_tr = triton.testing.do_bench(lambda: ldc_residual_triton(U_b, V_b, W_b, P_b, dx, dy, dz, nu_val))
    print(f"  Triton forward: {ms_tr:.3f}ms")
