"""
3D NS Triton kernels: Forward + Adjoint Backward.
Forward: fused stencil for res_u, res_v, res_w, res_div.
Backward: adjoint operator kernel for grad_U, grad_V, grad_W, grad_P.

Data layout: U, V, W, P: [Nt, Nx, Ny, Nz] contiguous, row-major (z fastest).
Stencil: 9-point in 4D — center + 8 neighbors (t±1, x±1, y±1, z±1).
"""
import torch
import triton
import triton.language as tl


# ── Forward kernel ─────────────────────────────────────────────────────────
@triton.jit
def ns3d_fwd_kernel(
    U_ptr, V_ptr, W_ptr, P_ptr,
    resU_ptr, resV_ptr, resW_ptr, resDiv_ptr,
    Nt: tl.constexpr, Nx: tl.constexpr, Ny: tl.constexpr, Nz: tl.constexpr,
    dx: tl.constexpr, dy: tl.constexpr, dz: tl.constexpr, dt: tl.constexpr,
    nu: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Each program handles BLOCK interior points.
    Interior domain: t in [1, Nt-2], x in [1, Nx-2], y in [1, Ny-2], z in [1, Nz-2].
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)

    # Convert flat interior index to 4D coordinates
    Ni_nx = Nx - 2
    Ni_ny = Ny - 2
    Ni_nz = Nz - 2

    it = idx // (Ni_nx * Ni_ny * Ni_nz) + 1
    rem = idx % (Ni_nx * Ni_ny * Ni_nz)
    ix = rem // (Ni_ny * Ni_nz) + 1
    rem = rem % (Ni_ny * Ni_nz)
    iy = rem // Ni_nz + 1
    iz = rem % Ni_nz + 1

    mask = (it < Nt - 1) & (ix < Nx - 1) & (iy < Ny - 1) & (iz < Nz - 1)

    # Strides: row-major [t, x, y, z]
    st = Nx * Ny * Nz  # time stride
    sx = Ny * Nz       # x stride
    sy = Nz            # y stride
    # z stride = 1

    base = it * st + ix * sx + iy * sy + iz

    # Load 9-point stencil for U
    u_c  = tl.load(U_ptr + base,            mask=mask)
    u_xp = tl.load(U_ptr + base + sx,       mask=mask)
    u_xm = tl.load(U_ptr + base - sx,       mask=mask)
    u_yp = tl.load(U_ptr + base + sy,       mask=mask)
    u_ym = tl.load(U_ptr + base - sy,       mask=mask)
    u_zp = tl.load(U_ptr + base + 1,        mask=mask)
    u_zm = tl.load(U_ptr + base - 1,        mask=mask)
    u_tp = tl.load(U_ptr + base + st,       mask=mask)
    u_tm = tl.load(U_ptr + base - st,       mask=mask)

    # Load 9-point stencil for V
    v_c  = tl.load(V_ptr + base,            mask=mask)
    v_xp = tl.load(V_ptr + base + sx,       mask=mask)
    v_xm = tl.load(V_ptr + base - sx,       mask=mask)
    v_yp = tl.load(V_ptr + base + sy,       mask=mask)
    v_ym = tl.load(V_ptr + base - sy,       mask=mask)
    v_zp = tl.load(V_ptr + base + 1,        mask=mask)
    v_zm = tl.load(V_ptr + base - 1,        mask=mask)
    v_tp = tl.load(V_ptr + base + st,       mask=mask)
    v_tm = tl.load(V_ptr + base - st,       mask=mask)

    # Load 9-point stencil for W
    w_c  = tl.load(W_ptr + base,            mask=mask)
    w_xp = tl.load(W_ptr + base + sx,       mask=mask)
    w_xm = tl.load(W_ptr + base - sx,       mask=mask)
    w_yp = tl.load(W_ptr + base + sy,       mask=mask)
    w_ym = tl.load(W_ptr + base - sy,       mask=mask)
    w_zp = tl.load(W_ptr + base + 1,        mask=mask)
    w_zm = tl.load(W_ptr + base - 1,        mask=mask)
    w_tp = tl.load(W_ptr + base + st,       mask=mask)
    w_tm = tl.load(W_ptr + base - st,       mask=mask)

    # Load 7-point stencil for P (no time derivative for pressure)
    p_xp = tl.load(P_ptr + base + sx,       mask=mask)
    p_xm = tl.load(P_ptr + base - sx,       mask=mask)
    p_yp = tl.load(P_ptr + base + sy,       mask=mask)
    p_ym = tl.load(P_ptr + base - sy,       mask=mask)
    p_zp = tl.load(P_ptr + base + 1,        mask=mask)
    p_zm = tl.load(P_ptr + base - 1,        mask=mask)

    # U derivatives
    u_t  = (u_tp - u_tm) / (2.0 * dt)
    u_x  = (u_xp - u_xm) / (2.0 * dx)
    u_y  = (u_yp - u_ym) / (2.0 * dy)
    u_z  = (u_zp - u_zm) / (2.0 * dz)
    u_xx = (u_xp - 2.0*u_c + u_xm) / (dx*dx)
    u_yy = (u_yp - 2.0*u_c + u_ym) / (dy*dy)
    u_zz = (u_zp - 2.0*u_c + u_zm) / (dz*dz)

    # V derivatives
    v_t  = (v_tp - v_tm) / (2.0 * dt)
    v_x  = (v_xp - v_xm) / (2.0 * dx)
    v_y  = (v_yp - v_ym) / (2.0 * dy)
    v_z  = (v_zp - v_zm) / (2.0 * dz)
    v_xx = (v_xp - 2.0*v_c + v_xm) / (dx*dx)
    v_yy = (v_yp - 2.0*v_c + v_ym) / (dy*dy)
    v_zz = (v_zp - 2.0*v_c + v_zm) / (dz*dz)

    # W derivatives
    w_t  = (w_tp - w_tm) / (2.0 * dt)
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

    # NS residuals
    res_u   = u_t + u_c*u_x + v_c*u_y + w_c*u_z + p_x - nu*(u_xx + u_yy + u_zz)
    res_v   = v_t + u_c*v_x + v_c*v_y + w_c*v_z + p_y - nu*(v_xx + v_yy + v_zz)
    res_w   = w_t + u_c*w_x + v_c*w_y + w_c*w_z + p_z - nu*(w_xx + w_yy + w_zz)
    res_div = u_x + v_y + w_z

    # Store to output (flat index matching the interior grid)
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
    Nt: tl.constexpr, Nx: tl.constexpr, Ny: tl.constexpr, Nz: tl.constexpr,
    dx: tl.constexpr, dy: tl.constexpr, dz: tl.constexpr, dt: tl.constexpr,
    nu: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Adjoint of the 3D NS stencil operator.
    Computes grad_U, grad_V, grad_W, grad_P from upstream gradients Gu, Gv, Gw, Gdiv.
    Uses 1D flat index: each thread computes gradient for one interior point.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)

    Ni_nx = Nx - 2; Ni_ny = Ny - 2; Ni_nz = Nz - 2
    Ni_vol = Ni_nx * Ni_ny * Ni_nz

    it = idx // Ni_vol + 1
    rem = idx % Ni_vol
    ix = rem // (Ni_ny * Ni_nz) + 1
    rem = rem % (Ni_ny * Ni_nz)
    iy = rem // Ni_nz + 1
    iz = rem % Ni_nz + 1

    mask = (it < Nt - 1) & (ix < Nx - 1) & (iy < Ny - 1) & (iz < Nz - 1)

    st = Nx * Ny * Nz; sx = Ny * Nz; sy = Nz

    base = it * st + ix * sx + iy * sy + iz

    # Load U, V, W stencils (needed for nonlinear adjoint terms)
    u_c  = tl.load(U_ptr + base,            mask=mask)
    u_xp = tl.load(U_ptr + base + sx,       mask=mask)
    u_xm = tl.load(U_ptr + base - sx,       mask=mask)
    u_yp = tl.load(U_ptr + base + sy,       mask=mask)
    u_ym = tl.load(U_ptr + base - sy,       mask=mask)
    u_zp = tl.load(U_ptr + base + 1,        mask=mask)
    u_zm = tl.load(U_ptr + base - 1,        mask=mask)

    v_c  = tl.load(V_ptr + base,            mask=mask)
    v_xp = tl.load(V_ptr + base + sx,       mask=mask)
    v_xm = tl.load(V_ptr + base - sx,       mask=mask)
    v_yp = tl.load(V_ptr + base + sy,       mask=mask)
    v_ym = tl.load(V_ptr + base - sy,       mask=mask)
    v_zp = tl.load(V_ptr + base + 1,        mask=mask)
    v_zm = tl.load(V_ptr + base - 1,        mask=mask)

    w_c  = tl.load(W_ptr + base,            mask=mask)
    w_xp = tl.load(W_ptr + base + sx,       mask=mask)
    w_xm = tl.load(W_ptr + base - sx,       mask=mask)
    w_yp = tl.load(W_ptr + base + sy,       mask=mask)
    w_ym = tl.load(W_ptr + base - sy,       mask=mask)
    w_zp = tl.load(W_ptr + base + 1,        mask=mask)
    w_zm = tl.load(W_ptr + base - 1,        mask=mask)

    # Load upstream gradients from flat interior grid
    # tg = it-1, xg = ix-1, yg = iy-1, zg = iz-1
    tg = it - 1; xg = ix - 1; yg = iy - 1; zg = iz - 1
    bg = tg * Ni_vol + xg * Ni_ny * Ni_nz + yg * Ni_nz + zg

    # Neighbor existence masks
    m_tp = (tg + 1 < Nt - 2) & mask
    m_tm = (tg - 1 >= 0) & mask
    m_xp = (xg + 1 < Nx - 2) & mask
    m_xm = (xg - 1 >= 0) & mask
    m_yp = (yg + 1 < Ny - 2) & mask
    m_ym = (yg - 1 >= 0) & mask
    m_zp = (zg + 1 < Nz - 2) & mask
    m_zm = (zg - 1 >= 0) & mask

    # Load G neighbors
    gu_c  = tl.load(Gu_ptr + bg,                    mask=mask, other=0.0)
    gv_c  = tl.load(Gv_ptr + bg,                    mask=mask, other=0.0)
    gw_c  = tl.load(Gw_ptr + bg,                    mask=mask, other=0.0)
    gd_c  = tl.load(Gdiv_ptr + bg,                  mask=mask, other=0.0)

    gu_tp = tl.load(Gu_ptr + bg + Ni_vol,           mask=m_tp, other=0.0)
    gu_tm = tl.load(Gu_ptr + bg - Ni_vol,           mask=m_tm, other=0.0)
    gu_xp = tl.load(Gu_ptr + bg + Ni_ny*Ni_nz,      mask=m_xp, other=0.0)
    gu_xm = tl.load(Gu_ptr + bg - Ni_ny*Ni_nz,      mask=m_xm, other=0.0)
    gu_yp = tl.load(Gu_ptr + bg + Ni_nz,            mask=m_yp, other=0.0)
    gu_ym = tl.load(Gu_ptr + bg - Ni_nz,            mask=m_ym, other=0.0)
    gu_zp = tl.load(Gu_ptr + bg + 1,                mask=m_zp, other=0.0)
    gu_zm = tl.load(Gu_ptr + bg - 1,                mask=m_zm, other=0.0)

    gv_tp = tl.load(Gv_ptr + bg + Ni_vol,           mask=m_tp, other=0.0)
    gv_tm = tl.load(Gv_ptr + bg - Ni_vol,           mask=m_tm, other=0.0)
    gv_xp = tl.load(Gv_ptr + bg + Ni_ny*Ni_nz,      mask=m_xp, other=0.0)
    gv_xm = tl.load(Gv_ptr + bg - Ni_ny*Ni_nz,      mask=m_xm, other=0.0)
    gv_yp = tl.load(Gv_ptr + bg + Ni_nz,            mask=m_yp, other=0.0)
    gv_ym = tl.load(Gv_ptr + bg - Ni_nz,            mask=m_ym, other=0.0)
    gv_zp = tl.load(Gv_ptr + bg + 1,                mask=m_zp, other=0.0)
    gv_zm = tl.load(Gv_ptr + bg - 1,                mask=m_zm, other=0.0)

    gw_tp = tl.load(Gw_ptr + bg + Ni_vol,           mask=m_tp, other=0.0)
    gw_tm = tl.load(Gw_ptr + bg - Ni_vol,           mask=m_tm, other=0.0)
    gw_xp = tl.load(Gw_ptr + bg + Ni_ny*Ni_nz,      mask=m_xp, other=0.0)
    gw_xm = tl.load(Gw_ptr + bg - Ni_ny*Ni_nz,      mask=m_xm, other=0.0)
    gw_yp = tl.load(Gw_ptr + bg + Ni_nz,            mask=m_yp, other=0.0)
    gw_ym = tl.load(Gw_ptr + bg - Ni_nz,            mask=m_ym, other=0.0)
    gw_zp = tl.load(Gw_ptr + bg + 1,                mask=m_zp, other=0.0)
    gw_zm = tl.load(Gw_ptr + bg - 1,                mask=m_zm, other=0.0)

    gd_tp = tl.load(Gdiv_ptr + bg + Ni_vol,         mask=m_tp, other=0.0)
    gd_tm = tl.load(Gdiv_ptr + bg - Ni_vol,         mask=m_tm, other=0.0)
    gd_xp = tl.load(Gdiv_ptr + bg + Ni_ny*Ni_nz,    mask=m_xp, other=0.0)
    gd_xm = tl.load(Gdiv_ptr + bg - Ni_ny*Ni_nz,    mask=m_xm, other=0.0)
    gd_yp = tl.load(Gdiv_ptr + bg + Ni_nz,          mask=m_yp, other=0.0)
    gd_ym = tl.load(Gdiv_ptr + bg - Ni_nz,          mask=m_ym, other=0.0)
    gd_zp = tl.load(Gdiv_ptr + bg + 1,              mask=m_zp, other=0.0)
    gd_zm = tl.load(Gdiv_ptr + bg - 1,              mask=m_zm, other=0.0)

    idx_val = 1.0/(2.0*dx); idy = 1.0/(2.0*dy); idz = 1.0/(2.0*dz); idt = 1.0/(2.0*dt)
    idx2 = 1.0/(dx*dx); idy2 = 1.0/(dy*dy); idz2 = 1.0/(dz*dz)

    # ── grad_U ────────────────────────────────────────────────────────────
    # Center
    grad_u  = gu_c * ((u_xp - u_xm) * idx_val + 2.0*nu*(idx2 + idy2 + idz2))
    grad_u += gv_c * (v_xp - v_xm) * idx_val   # u_c * v_x in res_v
    grad_u += gw_c * (w_xp - w_xm) * idx_val   # u_c * w_x in res_w
    # Neighbors
    grad_u += (gu_xm * u_xm - gu_xp * u_xp) * idx_val    # u in u_x neighbors
    grad_u += (gu_ym * v_ym - gu_yp * v_yp) * idy        # u_yp/v_yp neighbors
    grad_u += (gu_zm * w_zm - gu_zp * w_zp) * idz        # u_zp/w_zp neighbors
    grad_u += (gu_tm - gu_tp) * idt                       # time neighbors
    grad_u += (gd_xm - gd_xp) * idx_val                   # div

    # ── grad_V ────────────────────────────────────────────────────────────
    grad_v  = gu_c * (u_yp - u_ym) * idy                  # v_c * u_y in res_u
    grad_v += gv_c * ((v_yp - v_ym) * idy + 2.0*nu*(idx2 + idy2 + idz2))
    grad_v += gw_c * (w_yp - w_ym) * idy                  # v_c * w_y in res_w
    grad_v += (gv_xm * u_xm - gv_xp * u_xp) * idx_val    # v in v_x neighbors (u_c is U!)
    grad_v += (gv_ym * v_ym - gv_yp * v_yp) * idy
    grad_v += (gv_zm * w_zm - gv_zp * w_zp) * idz
    grad_v += (gv_tm - gv_tp) * idt
    grad_v += (gd_ym - gd_yp) * idy

    # ── grad_W ────────────────────────────────────────────────────────────
    grad_w  = gu_c * (u_zp - u_zm) * idz                  # w_c * u_z in res_u
    grad_w += gv_c * (v_zp - v_zm) * idz                  # w_c * v_z in res_v
    grad_w += gw_c * ((w_zp - w_zm) * idz + 2.0*nu*(idx2 + idy2 + idz2))
    grad_w += (gw_xm * u_xm - gw_xp * u_xp) * idx_val    # w in w_x neighbors
    grad_w += (gw_ym * v_ym - gw_yp * v_yp) * idy
    grad_w += (gw_zm * w_zm - gw_zp * w_zp) * idz
    grad_w += (gw_tm - gw_tp) * idt
    grad_w += (gd_zm - gd_zp) * idz

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
                                dx, dy, dz, dt, nu_val):
    """Add gradient contributions for 8 boundary faces not covered by ns3d_bwd_kernel."""
    Nt, Nx, Ny, Nz = U.shape
    inv_2dx = 1.0/(2.0*dx); inv_2dy = 1.0/(2.0*dy); inv_2dz = 1.0/(2.0*dz)
    inv_2dt = 1.0/(2.0*dt)
    inv_dx2 = 1.0/(dx*dx); inv_dy2 = 1.0/(dy*dy); inv_dz2 = 1.0/(dz*dz)

    i = slice(1, -1)

    # --- t=0 / t=Nt-1 ---
    grad_u[0, i, i, i] += -Gu[0] * inv_2dt
    grad_v[0, i, i, i] += -Gv[0] * inv_2dt
    grad_w[0, i, i, i] += -Gw[0] * inv_2dt
    grad_u[Nt-1, i, i, i] += Gu[Nt-3] * inv_2dt
    grad_v[Nt-1, i, i, i] += Gv[Nt-3] * inv_2dt
    grad_w[Nt-1, i, i, i] += Gw[Nt-3] * inv_2dt

    # --- x=0 ---
    grad_u[i, 0, i, i] += -Gu[:, 0] * (U[i, 1, i, i] * inv_2dx + nu_val * inv_dx2)
    grad_v[i, 0, i, i] += -Gv[:, 0] * (U[i, 1, i, i] * inv_2dx + nu_val * inv_dx2)  # u_c from U!
    grad_w[i, 0, i, i] += -Gw[:, 0] * (U[i, 1, i, i] * inv_2dx + nu_val * inv_dx2)
    grad_u[i, 0, i, i] += -Gdiv[:, 0] * inv_2dx  # div: -Gdiv/(2dx) for u_l

    # --- x=Nx-1 ---
    grad_u[i, Nx-1, i, i] += Gu[:, Nx-3] * (U[i, Nx-2, i, i] * inv_2dx - nu_val * inv_dx2)
    grad_v[i, Nx-1, i, i] += Gv[:, Nx-3] * (U[i, Nx-2, i, i] * inv_2dx - nu_val * inv_dx2)
    grad_w[i, Nx-1, i, i] += Gw[:, Nx-3] * (U[i, Nx-2, i, i] * inv_2dx - nu_val * inv_dx2)
    grad_u[i, Nx-1, i, i] += Gdiv[:, Nx-3] * inv_2dx  # div: +Gdiv/(2dx) for u_r

    # --- y=0 ---
    grad_u[i, i, 0, i] += -Gu[:, :, 0] * (V[i, i, 1, i] * inv_2dy + nu_val * inv_dy2)
    grad_v[i, i, 0, i] += -Gv[:, :, 0] * (V[i, i, 1, i] * inv_2dy + nu_val * inv_dy2)  # v_c from V
    grad_w[i, i, 0, i] += -Gw[:, :, 0] * (V[i, i, 1, i] * inv_2dy + nu_val * inv_dy2)
    grad_v[i, i, 0, i] += -Gdiv[:, :, 0] * inv_2dy  # div: -Gdiv/(2dy)

    # --- y=Ny-1 ---
    grad_u[i, i, Ny-1, i] += Gu[:, :, Ny-3] * (V[i, i, Ny-2, i] * inv_2dy - nu_val * inv_dy2)
    grad_v[i, i, Ny-1, i] += Gv[:, :, Ny-3] * (V[i, i, Ny-2, i] * inv_2dy - nu_val * inv_dy2)
    grad_w[i, i, Ny-1, i] += Gw[:, :, Ny-3] * (V[i, i, Ny-2, i] * inv_2dy - nu_val * inv_dy2)
    grad_v[i, i, Ny-1, i] += Gdiv[:, :, Ny-3] * inv_2dy  # div: +Gdiv/(2dy)

    # --- z=0 ---
    grad_u[i, i, i, 0] += -Gu[:, :, :, 0] * (W[i, i, i, 1] * inv_2dz + nu_val * inv_dz2)
    grad_v[i, i, i, 0] += -Gv[:, :, :, 0] * (W[i, i, i, 1] * inv_2dz + nu_val * inv_dz2)
    grad_w[i, i, i, 0] += -Gw[:, :, :, 0] * (W[i, i, i, 1] * inv_2dz + nu_val * inv_dz2)  # w_c from W
    grad_w[i, i, i, 0] += -Gdiv[:, :, :, 0] * inv_2dz  # div: -Gdiv/(2dz)

    # --- z=Nz-1 ---
    grad_u[i, i, i, Nz-1] += Gu[:, :, :, Nz-3] * (W[i, i, i, Nz-2] * inv_2dz - nu_val * inv_dz2)
    grad_v[i, i, i, Nz-1] += Gv[:, :, :, Nz-3] * (W[i, i, i, Nz-2] * inv_2dz - nu_val * inv_dz2)
    grad_w[i, i, i, Nz-1] += Gw[:, :, :, Nz-3] * (W[i, i, i, Nz-2] * inv_2dz - nu_val * inv_dz2)
    grad_w[i, i, i, Nz-1] += Gdiv[:, :, :, Nz-3] * inv_2dz  # div: +Gdiv/(2dz)

    # --- P boundaries ---
    grad_p[i, 0, i, i] += -Gu[:, 0] * inv_2dx
    grad_p[i, Nx-1, i, i] += Gu[:, Nx-3] * inv_2dx
    grad_p[i, i, 0, i] += -Gv[:, :, 0] * inv_2dy
    grad_p[i, i, Ny-1, i] += Gv[:, :, Ny-3] * inv_2dy
    grad_p[i, i, i, 0] += -Gw[:, :, :, 0] * inv_2dz
    grad_p[i, i, i, Nz-1] += Gw[:, :, :, Nz-3] * inv_2dz


# ── Autograd Function ──────────────────────────────────────────────────────
class _NS3DTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, V, W, P, dx, dy, dz, dt, nu_val):
        U, V, W, P = U.contiguous(), V.contiguous(), W.contiguous(), P.contiguous()
        Nt, Nx, Ny, Nz = U.shape
        Ni_nx, Ni_ny, Ni_nz = Nx - 2, Ny - 2, Nz - 2
        N_total = (Nt - 2) * Ni_nx * Ni_ny * Ni_nz

        res_u   = torch.empty(N_total, device=U.device, dtype=U.dtype)
        res_v   = torch.empty_like(res_u)
        res_w   = torch.empty_like(res_u)
        res_div = torch.empty_like(res_u)

        BLOCK = 256
        grid = ((N_total + BLOCK - 1) // BLOCK,)
        ns3d_fwd_kernel[grid](U, V, W, P, res_u, res_v, res_w, res_div,
                              Nt, Nx, Ny, Nz, dx, dy, dz, dt, nu_val, BLOCK)

        ctx.save_for_backward(U, V, W)
        ctx.res_u, ctx.res_v, ctx.res_w, ctx.res_div = res_u, res_v, res_w, res_div
        ctx.dx, ctx.dy, ctx.dz, ctx.dt, ctx.nu_val = dx, dy, dz, dt, nu_val
        return (res_u**2 + res_v**2 + res_w**2 + res_div**2).mean()

    @staticmethod
    def backward(ctx, grad_out):
        U, V, W = ctx.saved_tensors
        res_u, res_v, res_w, res_div = ctx.res_u, ctx.res_v, ctx.res_w, ctx.res_div
        dx, dy, dz, dt, nu_val = ctx.dx, ctx.dy, ctx.dz, ctx.dt, ctx.nu_val
        Nt, Nx, Ny, Nz = U.shape
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
                              Nt, Nx, Ny, Nz, dx, dy, dz, dt, nu_val, BLOCK)

        # Reshape G values from flat to 4D for boundary slicing
        Ni_ny, Ni_nz = Ny - 2, Nz - 2
        Gu_4d   = Gu.reshape(Nt - 2, Nx - 2, Ni_ny, Ni_nz)
        Gv_4d   = Gv.reshape(Nt - 2, Nx - 2, Ni_ny, Ni_nz)
        Gw_4d   = Gw.reshape(Nt - 2, Nx - 2, Ni_ny, Ni_nz)
        Gdiv_4d = Gdiv.reshape(Nt - 2, Nx - 2, Ni_ny, Ni_nz)

        _add_boundary_gradients_3d(U, V, W, Gu_4d, Gv_4d, Gw_4d, Gdiv_4d,
                                    grad_u, grad_v, grad_w, grad_p,
                                    dx, dy, dz, dt, nu_val)

        return grad_u, grad_v, grad_w, grad_p, None, None, None, None, None


def ns3d_residual_triton(U, V, W, P, dx, dy, dz, dt, nu_val):
    return _NS3DTriton.apply(U.contiguous(), V.contiguous(), W.contiguous(), P.contiguous(),
                             dx, dy, dz, dt, nu_val)


# ── Correctness check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, numpy as np, os
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

    os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/triton_stencil3d_test')

    from cases.tgv_3d.common import pde_residual_pytorch, make_grid, exact_uvwp

    Nt, Nx, Ny, Nz = 10, 32, 32, 32
    nu_val = 0.01
    device = "cuda"
    dtype = torch.float64

    X, Y, Z, T, dx, dy, dz, dt = make_grid()

    def make_uvp(rg=False):
        U = (torch.sin(X)*torch.cos(Y)*torch.cos(Z)).to(dtype).requires_grad_(rg)
        V = (-torch.cos(X)*torch.sin(Y)*torch.cos(Z)).to(dtype).requires_grad_(rg)
        W = torch.zeros_like(U).requires_grad_(rg)
        P = (1.0/16.0*(torch.cos(2*X)+torch.cos(2*Y))*(torch.cos(2*Z)+2)).to(dtype).requires_grad_(rg)
        return U, V, W, P

    # Forward check
    print("1. Forward check...")
    U_pt, V_pt, W_pt, P_pt = make_uvp(True)
    U_tr, V_tr, W_tr, P_tr = make_uvp(True)
    loss_pt = pde_residual_pytorch(U_pt.float(), V_pt.float(), W_pt.float(), P_pt.float(), dx, dy, dz, dt)
    loss_tr = ns3d_residual_triton(U_tr.float(), V_tr.float(), W_tr.float(), P_tr.float(), dx, dy, dz, dt, nu_val)
    print(f"  Forward rel_err: {abs(loss_pt.item()-loss_tr.item())/(abs(loss_pt.item())+1e-12):.2e}")

    # Gradient check
    print("2. Gradient check (float64)...")
    U_pt2, V_pt2, W_pt2, P_pt2 = make_uvp(True)
    U_tr2, V_tr2, W_tr2, P_tr2 = make_uvp(True)
    loss_pt_d = pde_residual_pytorch(U_pt2, V_pt2, W_pt2, P_pt2, dx, dy, dz, dt)
    loss_tr_d = ns3d_residual_triton(U_tr2, V_tr2, W_tr2, P_tr2, dx, dy, dz, dt, nu_val)
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

    # Benchmark
    import triton as _triton
    U_b, V_b, W_b, P_b = [v.float() for v in make_uvp()]
    ms_tr = _triton.testing.do_bench(lambda: ns3d_residual_triton(U_b, V_b, W_b, P_b, dx, dy, dz, dt, nu_val))
    print(f"3. Triton forward: {ms_tr:.3f}ms")
