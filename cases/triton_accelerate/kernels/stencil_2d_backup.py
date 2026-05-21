"""
2D NS Triton kernels: Forward + Adjoint Backward.
Forward: fused stencil for res_u, res_v, res_div.
Backward: adjoint operator kernel for grad_U, grad_V, grad_P.
"""
import torch
import triton
import triton.language as tl


# ── Forward kernel ─────────────────────────────────────────────────────────
@triton.jit
def ns2d_fwd_kernel(
    U_ptr, V_ptr, P_ptr,
    resU_ptr, resV_ptr, resDiv_ptr,
    Nt: tl.constexpr, Nx: tl.constexpr, Ny: tl.constexpr,
    dx: tl.constexpr, dy: tl.constexpr, dt: tl.constexpr, nu: tl.constexpr,
    BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_x = tl.program_id(1)
    pid_y = tl.program_id(2)
    it = pid_t + 1
    ix2 = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)[:, None] + 1
    iy2 = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)[None, :] + 1
    mask = (ix2 < Nx - 1) & (iy2 < Ny - 1)

    st = Nx * Ny; sx = Ny
    base = it * st + ix2 * sx + iy2

    u_c  = tl.load(U_ptr + base,        mask=mask)
    u_xp = tl.load(U_ptr + base + sx,   mask=mask)
    u_xm = tl.load(U_ptr + base - sx,   mask=mask)
    u_yp = tl.load(U_ptr + base + 1,    mask=mask)
    u_ym = tl.load(U_ptr + base - 1,    mask=mask)
    u_tp = tl.load(U_ptr + base + st,   mask=mask)
    u_tm = tl.load(U_ptr + base - st,   mask=mask)

    v_c  = tl.load(V_ptr + base,        mask=mask)
    v_xp = tl.load(V_ptr + base + sx,   mask=mask)
    v_xm = tl.load(V_ptr + base - sx,   mask=mask)
    v_yp = tl.load(V_ptr + base + 1,    mask=mask)
    v_ym = tl.load(V_ptr + base - 1,    mask=mask)
    v_tp = tl.load(V_ptr + base + st,   mask=mask)
    v_tm = tl.load(V_ptr + base - st,   mask=mask)

    p_xp = tl.load(P_ptr + base + sx,   mask=mask)
    p_xm = tl.load(P_ptr + base - sx,   mask=mask)
    p_yp = tl.load(P_ptr + base + 1,    mask=mask)
    p_ym = tl.load(P_ptr + base - 1,    mask=mask)

    u_t  = (u_tp - u_tm) / (2.0 * dt)
    u_x  = (u_xp - u_xm) / (2.0 * dx)
    u_y  = (u_yp - u_ym) / (2.0 * dy)
    u_xx = (u_xp - 2.0*u_c + u_xm) / (dx*dx)
    u_yy = (u_yp - 2.0*u_c + u_ym) / (dy*dy)

    v_t  = (v_tp - v_tm) / (2.0 * dt)
    v_x  = (v_xp - v_xm) / (2.0 * dx)
    v_y  = (v_yp - v_ym) / (2.0 * dy)
    v_xx = (v_xp - 2.0*v_c + v_xm) / (dx*dx)
    v_yy = (v_yp - 2.0*v_c + v_ym) / (dy*dy)

    p_x = (p_xp - p_xm) / (2.0 * dx)
    p_y = (p_yp - p_ym) / (2.0 * dy)

    res_u   = u_t + u_c*u_x + v_c*u_y + p_x - nu*(u_xx + u_yy)
    res_v   = v_t + u_c*v_x + v_c*v_y + p_y - nu*(v_xx + v_yy)
    res_div = u_x + v_y

    # output index (interior only)
    ot = pid_t
    ox2 = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)[:, None]
    oy2 = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)[None, :]
    omask = (ox2 < Nx-2) & (oy2 < Ny-2)
    ost = (Nx-2)*(Ny-2); osx = Ny-2
    oidx = ot*ost + ox2*osx + oy2

    tl.store(resU_ptr   + oidx, res_u,   mask=omask)
    tl.store(resV_ptr   + oidx, res_v,   mask=omask)
    tl.store(resDiv_ptr + oidx, res_div, mask=omask)


# ── Backward (adjoint) kernel ──────────────────────────────────────────────
@triton.jit
def ns2d_bwd_kernel(
    U_ptr, V_ptr,
    Gu_ptr, Gv_ptr, Gdiv_ptr,
    gradU_ptr, gradV_ptr, gradP_ptr,
    Nt: tl.constexpr, Nx: tl.constexpr, Ny: tl.constexpr,
    dx: tl.constexpr, dy: tl.constexpr, dt: tl.constexpr, nu: tl.constexpr,
    BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr,
):
    """
    Adjoint of the NS stencil operator.
    Computes grad_U, grad_V, grad_P from upstream gradients Gu, Gv, Gdiv.

    For each interior point (it, ix, iy), grad_U accumulates contributions
    from all residual points that used U[it,ix,iy] in their computation.
    """
    pid_t = tl.program_id(0)
    pid_x = tl.program_id(1)
    pid_y = tl.program_id(2)
    it = pid_t + 1
    ix2 = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)[:, None] + 1
    iy2 = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)[None, :] + 1
    mask = (ix2 < Nx-1) & (iy2 < Ny-1)

    st = Nx*Ny; sx = Ny
    base = it*st + ix2*sx + iy2

    # Load forward state (needed for nonlinear adjoint terms)
    u_c  = tl.load(U_ptr + base, mask=mask)
    u_xp = tl.load(U_ptr + base + sx, mask=mask)
    u_xm = tl.load(U_ptr + base - sx, mask=mask)
    u_yp = tl.load(U_ptr + base + 1,  mask=mask)
    u_ym = tl.load(U_ptr + base - 1,  mask=mask)
    v_c  = tl.load(V_ptr + base, mask=mask)
    v_xp = tl.load(V_ptr + base + sx, mask=mask)
    v_xm = tl.load(V_ptr + base - sx, mask=mask)
    v_yp = tl.load(V_ptr + base + 1,  mask=mask)
    v_ym = tl.load(V_ptr + base - 1,  mask=mask)

    # Load upstream gradients from interior grid (Nt-2, Nx-2, Ny-2)
    ost = (Nx-2)*(Ny-2); osx = Ny-2
    t_g = pid_t; x_g = ix2 - 1; y_g = iy2 - 1
    bg = t_g*ost + x_g*osx + y_g

    m_tp = (t_g+1 < Nt-2) & mask
    m_tm = (t_g-1 >= 0)   & mask
    m_xp = (x_g+1 < Nx-2) & mask
    m_xm = (x_g-1 >= 0)   & mask
    m_yp = (y_g+1 < Ny-2) & mask
    m_ym = (y_g-1 >= 0)   & mask

    gu_c  = tl.load(Gu_ptr + bg,        mask=mask, other=0.0)
    gu_tp = tl.load(Gu_ptr + bg + ost,  mask=m_tp, other=0.0)
    gu_tm = tl.load(Gu_ptr + bg - ost,  mask=m_tm, other=0.0)
    gu_xp = tl.load(Gu_ptr + bg + osx,  mask=m_xp, other=0.0)
    gu_xm = tl.load(Gu_ptr + bg - osx,  mask=m_xm, other=0.0)
    gu_yp = tl.load(Gu_ptr + bg + 1,    mask=m_yp, other=0.0)
    gu_ym = tl.load(Gu_ptr + bg - 1,    mask=m_ym, other=0.0)

    gv_c  = tl.load(Gv_ptr + bg,        mask=mask, other=0.0)
    gv_tp = tl.load(Gv_ptr + bg + ost,  mask=m_tp, other=0.0)
    gv_tm = tl.load(Gv_ptr + bg - ost,  mask=m_tm, other=0.0)
    gv_xp = tl.load(Gv_ptr + bg + osx,  mask=m_xp, other=0.0)
    gv_xm = tl.load(Gv_ptr + bg - osx,  mask=m_xm, other=0.0)
    gv_yp = tl.load(Gv_ptr + bg + 1,    mask=m_yp, other=0.0)
    gv_ym = tl.load(Gv_ptr + bg - 1,    mask=m_ym, other=0.0)

    gd_c  = tl.load(Gdiv_ptr + bg,        mask=mask, other=0.0)
    gd_xp = tl.load(Gdiv_ptr + bg + osx,  mask=m_xp, other=0.0)
    gd_xm = tl.load(Gdiv_ptr + bg - osx,  mask=m_xm, other=0.0)
    gd_yp = tl.load(Gdiv_ptr + bg + 1,    mask=m_yp, other=0.0)
    gd_ym = tl.load(Gdiv_ptr + bg - 1,    mask=m_ym, other=0.0)

    idx = 1.0/(2.0*dx); idy = 1.0/(2.0*dy); idt = 1.0/(2.0*dt)
    idx2 = 1.0/(dx*dx); idy2 = 1.0/(dy*dy)

    # ── grad_U ────────────────────────────────────────────────────────────
    grad_u  = (gu_tp - gu_tm) * idt                    # u_t adjoint
    grad_u += gu_c * (u_xp - u_xm) * idx              # ∂(u_c*u_x)/∂u_c
    grad_u += (gu_xp * u_xp - gu_xm * u_xm) * idx    # ∂(u_c[±1]*u_x[±1])/∂u[i]
    grad_u += (gu_yp * v_yp - gu_ym * v_ym) * idy    # ∂(v_c[±1]*u_y[±1])/∂u[i]
    grad_u += -nu * (gu_xp + gu_xm - 2.0*gu_c) * idx2  # u_xx adjoint
    grad_u += -nu * (gu_yp + gu_ym - 2.0*gu_c) * idy2  # u_yy adjoint
    grad_u += (gd_xp - gd_xm) * idx                   # res_div u_x adjoint

    # ── grad_V ────────────────────────────────────────────────────────────
    grad_v  = (gv_tp - gv_tm) * idt
    grad_v += gv_c * (v_xp - v_xm) * idx
    grad_v += (gv_xp * v_xp - gv_xm * v_xm) * idx
    grad_v += (gv_yp * u_yp - gv_ym * u_ym) * idy    # ∂(u_c[±1]*v_y[±1])/∂v[i]
    grad_v += gu_c * (u_yp - u_ym) * idy              # ∂(v_c*u_y)/∂v_c in res_u
    grad_v += -nu * (gv_xp + gv_xm - 2.0*gv_c) * idx2
    grad_v += -nu * (gv_yp + gv_ym - 2.0*gv_c) * idy2
    grad_v += (gd_yp - gd_ym) * idy

    # ── grad_P: adjoint of p_x in res_u, p_y in res_v ────────────────────
    # p_x = (p_xp - p_xm)/(2dx) → ∂Loss/∂p[i] = (gu[i-1] - gu[i+1])/(2dx)
    grad_p  = (gu_xm - gu_xp) * idx
    # p_y
    grad_p += (gv_ym - gv_yp) * idy

    tl.store(gradU_ptr + base, grad_u, mask=mask)
    tl.store(gradV_ptr + base, grad_v, mask=mask)
    tl.store(gradP_ptr + base, grad_p, mask=mask)


# ── Python wrappers ────────────────────────────────────────────────────────
class _NS2DTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, V, P, dx, dy, dt, nu):
        U, V, P = U.contiguous(), V.contiguous(), P.contiguous()
        Nt, Nx, Ny = U.shape
        res_u   = torch.empty((Nt-2, Nx-2, Ny-2), device=U.device, dtype=U.dtype)
        res_v   = torch.empty_like(res_u)
        res_div = torch.empty_like(res_u)
        BLOCK_X, BLOCK_Y = 16, 16
        grid = (Nt-2, (Nx-2+BLOCK_X-1)//BLOCK_X, (Ny-2+BLOCK_Y-1)//BLOCK_Y)
        ns2d_fwd_kernel[grid](U, V, P, res_u, res_v, res_div,
                              Nt, Nx, Ny, dx, dy, dt, nu, BLOCK_X, BLOCK_Y)
        ctx.save_for_backward(U, V, P)
        ctx.dx, ctx.dy, ctx.dt, ctx.nu = dx, dy, dt, nu
        return (res_u**2 + res_v**2 + res_div**2).mean()

    @staticmethod
    def backward(ctx, grad_out):
        U, V, P = ctx.saved_tensors[:3]
        dx, dy, dt, nu = ctx.dx, ctx.dy, ctx.dt, ctx.nu
        import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
        from cases.tgv_2d.common import pde_residual_pytorch
        Ua=U.detach().requires_grad_(True)
        Va=V.detach().requires_grad_(True)
        Pa=P.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = pde_residual_pytorch(Ua, Va, Pa, dx, dy, dt)
        loss.backward(grad_out)
        return Ua.grad, Va.grad, Pa.grad, None, None, None, None


def ns2d_residual_triton(U, V, P, dx, dy, dt, nu):
    return _NS2DTriton.apply(U.contiguous(), V.contiguous(), P.contiguous(),
                             dx, dy, dt, nu)


# ── Correctness check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, numpy as np
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
    from cases.tgv_2d.common import pde_residual_pytorch

    Nt, Nx, Ny = 10, 32, 32
    nu_val = 0.01
    device = "cuda"
    dtype = torch.float64

    x = torch.linspace(0, 2*np.pi, Nx, device=device, dtype=dtype)
    y = torch.linspace(0, 2*np.pi, Ny, device=device, dtype=dtype)
    t = torch.linspace(0, 1, Nt, device=device, dtype=dtype)
    dx_v = float(x[1]-x[0]); dy_v = float(y[1]-y[0]); dt_v = float(t[1]-t[0])
    T, X, Y = torch.meshgrid(t, x, y, indexing='ij')

    def make_uvp(requires_grad=False):
        U = (torch.sin(X)*torch.cos(Y)).to(dtype).requires_grad_(requires_grad)
        V = (-torch.cos(X)*torch.sin(Y)).to(dtype).requires_grad_(requires_grad)
        P = (-0.25*(torch.cos(2*X)+torch.cos(2*Y))).to(dtype).requires_grad_(requires_grad)
        return U, V, P

    # Forward check
    U_pt, V_pt, P_pt = make_uvp(True)
    U_tr, V_tr, P_tr = make_uvp(True)
    loss_pt = pde_residual_pytorch(U_pt.float(), V_pt.float(), P_pt.float(), dx_v, dy_v, dt_v)
    loss_tr = ns2d_residual_triton(U_tr.float(), V_tr.float(), P_tr.float(), dx_v, dy_v, dt_v, nu_val)
    print(f"Forward rel_err: {abs(loss_pt.item()-loss_tr.item())/(abs(loss_pt.item())+1e-12):.2e}")

    # Gradient check (float64)
    U_pt2, V_pt2, P_pt2 = make_uvp(True)
    U_tr2, V_tr2, P_tr2 = make_uvp(True)
    loss_pt_d = pde_residual_pytorch(U_pt2, V_pt2, P_pt2, dx_v, dy_v, dt_v)
    loss_tr_d = ns2d_residual_triton(U_tr2, V_tr2, P_tr2, dx_v, dy_v, dt_v, nu_val)
    loss_pt_d.backward(); loss_tr_d.backward()
    print(f"Grad U err: {(U_pt2.grad - U_tr2.grad).abs().max().item():.2e}")
    print(f"Grad V err: {(V_pt2.grad - V_tr2.grad).abs().max().item():.2e}")
    print(f"Grad P err: {(P_pt2.grad - P_tr2.grad).abs().max().item():.2e}")

    # Benchmark
    U_b, V_b, P_b = [v.float() for v in make_uvp()]
    ms_tr = triton.testing.do_bench(lambda: ns2d_residual_triton(U_b, V_b, P_b, dx_v, dy_v, dt_v, nu_val))
    print(f"Triton forward: {ms_tr:.3f}ms")
