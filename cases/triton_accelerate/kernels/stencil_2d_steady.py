"""
Steady-state 2D NS Triton kernels for LDC: Forward + Adjoint Backward.
PDE: res_u = u*u_x + v*u_y - nu*(u_xx+u_yy)
     res_v = u*v_x + v*v_y - nu*(v_xx+v_yy)
No time, no pressure. Grid [Nx, Ny].
"""
import torch
import triton
import triton.language as tl


# ── Forward kernel ─────────────────────────────────────────────────────────
@triton.jit
def ldc_fwd_kernel(
    U_ptr, V_ptr,
    resU_ptr, resV_ptr,
    Nx: tl.constexpr, Ny: tl.constexpr,
    dx: tl.constexpr, dy: tl.constexpr, nu: tl.constexpr,
    BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr,
):
    """Each program handles a BLOCK_X x BLOCK_Y tile of interior points."""
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    ix = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)[:, None] + 1
    iy = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)[None, :] + 1
    mask = (ix < Nx - 1) & (iy < Ny - 1)

    sx = Ny  # x-stride
    base = ix * sx + iy

    u_c  = tl.load(U_ptr + base,       mask=mask)
    u_xp = tl.load(U_ptr + base + sx,  mask=mask)
    u_xm = tl.load(U_ptr + base - sx,  mask=mask)
    u_yp = tl.load(U_ptr + base + 1,   mask=mask)
    u_ym = tl.load(U_ptr + base - 1,   mask=mask)

    v_c  = tl.load(V_ptr + base,       mask=mask)
    v_xp = tl.load(V_ptr + base + sx,  mask=mask)
    v_xm = tl.load(V_ptr + base - sx,  mask=mask)
    v_yp = tl.load(V_ptr + base + 1,   mask=mask)
    v_ym = tl.load(V_ptr + base - 1,   mask=mask)

    u_x  = (u_xp - u_xm) / (2.0 * dx)
    u_y  = (u_yp - u_ym) / (2.0 * dy)
    u_xx = (u_xp - 2.0*u_c + u_xm) / (dx*dx)
    u_yy = (u_yp - 2.0*u_c + u_ym) / (dy*dy)

    v_x  = (v_xp - v_xm) / (2.0 * dx)
    v_y  = (v_yp - v_ym) / (2.0 * dy)
    v_xx = (v_xp - 2.0*v_c + v_xm) / (dx*dx)
    v_yy = (v_yp - 2.0*v_c + v_ym) / (dy*dy)

    res_u = u_c*u_x + v_c*u_y - nu*(u_xx + u_yy)
    res_v = u_c*v_x + v_c*v_y - nu*(v_xx + v_yy)

    # output indexing
    ox = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)[:, None]
    oy = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)[None, :]
    omask = (ox < Nx - 2) & (oy < Ny - 2)
    osx = Ny - 2
    oidx = ox * osx + oy

    tl.store(resU_ptr + oidx, res_u, mask=omask)
    tl.store(resV_ptr + oidx, res_v, mask=omask)


# ── Backward kernel ────────────────────────────────────────────────────────
@triton.jit
def ldc_bwd_kernel(
    U_ptr, V_ptr,
    Gu_ptr, Gv_ptr,
    gradU_ptr, gradV_ptr,
    Nx: tl.constexpr, Ny: tl.constexpr,
    dx: tl.constexpr, dy: tl.constexpr, nu: tl.constexpr,
    BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr,
):
    """Adjoint of the steady NS stencil for U and V."""
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    ix2 = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)[:, None] + 1
    iy2 = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)[None, :] + 1
    mask = (ix2 < Nx - 1) & (iy2 < Ny - 1)

    sx = Ny
    base = ix2 * sx + iy2

    # Load U and V for nonlinear adjoint terms
    u_c  = tl.load(U_ptr + base,       mask=mask)
    u_xp = tl.load(U_ptr + base + sx,  mask=mask)
    u_xm = tl.load(U_ptr + base - sx,  mask=mask)
    u_yp = tl.load(U_ptr + base + 1,   mask=mask)
    u_ym = tl.load(U_ptr + base - 1,   mask=mask)

    v_c  = tl.load(V_ptr + base,       mask=mask)
    v_xp = tl.load(V_ptr + base + sx,  mask=mask)
    v_xm = tl.load(V_ptr + base - sx,  mask=mask)
    v_yp = tl.load(V_ptr + base + 1,   mask=mask)
    v_ym = tl.load(V_ptr + base - 1,   mask=mask)

    # Load upstream gradients from interior grid
    osx = Ny - 2
    xg = ix2 - 1; yg = iy2 - 1
    bg = xg * osx + yg

    m_xp = (xg + 1 < Nx - 2) & mask
    m_xm = (xg - 1 >= 0) & mask
    m_yp = (yg + 1 < Ny - 2) & mask
    m_ym = (yg - 1 >= 0) & mask

    gu_c = tl.load(Gu_ptr + bg,       mask=mask, other=0.0)
    gu_xp = tl.load(Gu_ptr + bg + osx, mask=m_xp, other=0.0)
    gu_xm = tl.load(Gu_ptr + bg - osx, mask=m_xm, other=0.0)
    gu_yp = tl.load(Gu_ptr + bg + 1,   mask=m_yp, other=0.0)
    gu_ym = tl.load(Gu_ptr + bg - 1,   mask=m_ym, other=0.0)

    gv_c = tl.load(Gv_ptr + bg,       mask=mask, other=0.0)
    gv_xp = tl.load(Gv_ptr + bg + osx, mask=m_xp, other=0.0)
    gv_xm = tl.load(Gv_ptr + bg - osx, mask=m_xm, other=0.0)
    gv_yp = tl.load(Gv_ptr + bg + 1,   mask=m_yp, other=0.0)
    gv_ym = tl.load(Gv_ptr + bg - 1,   mask=m_ym, other=0.0)

    idx_val = 1.0/(2.0*dx); idy = 1.0/(2.0*dy)
    idx2 = 1.0/(dx*dx); idy2 = 1.0/(dy*dy)

    # ── grad_U ── (from res_u and res_v)
    # self: gu_c * (u_x + nu*(2/dx²+2/dy²)) + gv_c * v_x
    grad_u  = gu_c * ((u_xp - u_xm) * idx_val + 2.0 * nu * (idx2 + idy2))
    grad_u += gv_c * (v_xp - v_xm) * idx_val
    # x-neighbors convective: U[i±1] in u_c*u_x of res_u[i∓1,j]
    grad_u += (gu_xm * u_xm - gu_xp * u_xp) * idx_val
    # y-neighbors convective (v_c*u_y in res_u)
    grad_u += (gu_ym * v_ym - gu_yp * v_yp) * idy
    # Laplacian neighbors: -nu * (gu_xp + gu_xm)/dx² - nu * (gu_yp + gu_ym)/dy²
    grad_u += -nu * (gu_xp + gu_xm) * idx2
    grad_u += -nu * (gu_yp + gu_ym) * idy2

    # ── grad_V ── (from res_u and res_v)
    # self: gu_c * u_y (v_c in res_u) + gv_c * (v_y + nu*(2/dx²+2/dy²))
    grad_v  = gu_c * (u_yp - u_ym) * idy
    grad_v += gv_c * ((v_yp - v_ym) * idy + 2.0 * nu * (idx2 + idy2))
    # x-neighbors convective (u_c*v_x in res_v)
    grad_v += (gv_xm * u_xm - gv_xp * u_xp) * idx_val
    # y-neighbors convective (v_c*v_y in res_v)
    grad_v += (gv_ym * v_ym - gv_yp * v_yp) * idy
    # Laplacian neighbors
    grad_v += -nu * (gv_xp + gv_xm) * idx2
    grad_v += -nu * (gv_yp + gv_ym) * idy2

    tl.store(gradU_ptr + base, grad_u, mask=mask)
    tl.store(gradV_ptr + base, grad_v, mask=mask)


# ── Boundary gradient fix ──────────────────────────────────────────────────
def _add_boundary_gradients_ldc(U, V, Gu, Gv, grad_u, grad_v, dx, dy, nu_val):
    """Add gradient contributions for 4 boundary edges."""
    Nx, Ny = U.shape
    inv_2dx = 1.0/(2.0*dx); inv_2dy = 1.0/(2.0*dy)
    inv_dx2 = 1.0/(dx*dx); inv_dy2 = 1.0/(dy*dy)
    i = slice(1, -1)

    # x=0: U[0,j] is u_l in res_u[0,j-1] and v_l in res_v[0,j-1]
    # dRes_u/du_l = -u_c/(2dx) - nu/dx², where u_c = U[1,j]
    grad_u[0, i] += -Gu[0] * (U[1, i] * inv_2dx + nu_val * inv_dx2)
    grad_v[0, i] += -Gv[0] * (U[1, i] * inv_2dx + nu_val * inv_dx2)  # u_c from U!

    # x=Nx-1: u_r
    grad_u[Nx-1, i] += Gu[Nx-3] * (U[Nx-2, i] * inv_2dx - nu_val * inv_dx2)
    grad_v[Nx-1, i] += Gv[Nx-3] * (U[Nx-2, i] * inv_2dx - nu_val * inv_dx2)

    # y=0: u_ym
    grad_u[i, 0] += -Gu[:, 0] * (V[i, 1] * inv_2dy + nu_val * inv_dy2)
    grad_v[i, 0] += -Gv[:, 0] * (V[i, 1] * inv_2dy + nu_val * inv_dy2)  # v_c from V

    # y=Ny-1: u_yp
    grad_u[i, Ny-1] += Gu[:, Ny-3] * (V[i, Ny-2] * inv_2dy - nu_val * inv_dy2)
    grad_v[i, Ny-1] += Gv[:, Ny-3] * (V[i, Ny-2] * inv_2dy - nu_val * inv_dy2)


# ── Autograd Function ──────────────────────────────────────────────────────
class _LDCTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, V, dx, dy, nu_val):
        U, V = U.contiguous(), V.contiguous()
        Nx, Ny = U.shape
        N_total = (Nx - 2) * (Ny - 2)
        res_u = torch.empty(N_total, device=U.device, dtype=U.dtype)
        res_v = torch.empty_like(res_u)
        BLOCK_X, BLOCK_Y = 16, 16
        grid = ((Nx - 2 + BLOCK_X - 1) // BLOCK_X, (Ny - 2 + BLOCK_Y - 1) // BLOCK_Y)
        ldc_fwd_kernel[grid](U, V, res_u, res_v, Nx, Ny, dx, dy, nu_val, BLOCK_X, BLOCK_Y)
        ctx.save_for_backward(U, V)
        ctx.res_u, ctx.res_v = res_u, res_v
        ctx.dx, ctx.dy, ctx.nu_val = dx, dy, nu_val
        return (res_u**2 + res_v**2).mean()

    @staticmethod
    def backward(ctx, grad_out):
        U, V = ctx.saved_tensors
        res_u, res_v = ctx.res_u, ctx.res_v
        dx, dy, nu_val = ctx.dx, ctx.dy, ctx.nu_val
        Nx, Ny = U.shape
        N_total = res_u.numel()

        scale = (2.0 / N_total) * grad_out
        # Reshape G values to 2D for boundary fix
        Gu_flat = res_u * scale; Gv_flat = res_v * scale

        grad_u = torch.zeros_like(U)
        grad_v = torch.zeros_like(V)
        BLOCK_X, BLOCK_Y = 16, 16
        grid = ((Nx - 2 + BLOCK_X - 1) // BLOCK_X, (Ny - 2 + BLOCK_Y - 1) // BLOCK_Y)
        ldc_bwd_kernel[grid](U, V, Gu_flat, Gv_flat, grad_u, grad_v,
                             Nx, Ny, dx, dy, nu_val, BLOCK_X, BLOCK_Y)

        # Reshape G to 2D for boundary fix
        Gu = Gu_flat.reshape(Nx - 2, Ny - 2)
        Gv = Gv_flat.reshape(Nx - 2, Ny - 2)
        _add_boundary_gradients_ldc(U, V, Gu, Gv, grad_u, grad_v, dx, dy, nu_val)

        return grad_u, grad_v, None, None, None


def ldc_residual_triton(U, V, dx, dy, nu_val):
    return _LDCTriton.apply(U.contiguous(), V.contiguous(), dx, dy, nu_val)


# ── Correctness check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, numpy as np, os
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
    os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/triton_ldc_test')

    from cases.ldc_2d.common import pde_residual_pytorch, make_grid

    Nx, Ny = 64, 64
    nu_val = 0.01
    device = "cuda"
    dtype = torch.float64

    X, Y, dx, dy = make_grid()
    U = (torch.sin(np.pi * X) * torch.sin(np.pi * Y)).to(dtype)
    V = (torch.cos(np.pi * X) * torch.cos(np.pi * Y)).to(dtype)

    # Forward check
    print("1. Forward check...")
    U_pt = U.clone().requires_grad_(True)
    V_pt = V.clone().requires_grad_(True)
    loss_pt = pde_residual_pytorch(U_pt.float(), V_pt.float(), dx, dy)
    U_tr = U.clone().requires_grad_(True)
    V_tr = V.clone().requires_grad_(True)
    loss_tr = ldc_residual_triton(U_tr.float(), V_tr.float(), dx, dy, nu_val)
    print(f"  Forward rel_err: {abs(loss_pt.item()-loss_tr.item())/(abs(loss_pt.item())+1e-12):.2e}")

    # Gradient check
    print("2. Gradient check (float64)...")
    U_pt2 = U.clone().requires_grad_(True)
    V_pt2 = V.clone().requires_grad_(True)
    loss_pt_d = pde_residual_pytorch(U_pt2, V_pt2, dx, dy)
    loss_pt_d.backward()

    U_tr2 = U.clone().requires_grad_(True)
    V_tr2 = V.clone().requires_grad_(True)
    loss_tr_d = ldc_residual_triton(U_tr2, V_tr2, dx, dy, nu_val)
    loss_tr_d.backward()

    eU = (U_pt2.grad - U_tr2.grad).abs().max().item()
    eV = (V_pt2.grad - V_tr2.grad).abs().max().item()
    print(f"  Grad U err: {eU:.2e}")
    print(f"  Grad V err: {eV:.2e}")
    print(f"  {'ALL PASS' if max(eU,eV) < 1e-5 else 'FAIL'}")

    # MLP field test
    print("3. MLP field test...")
    from cases.ldc_2d.common import MLP
    model = MLP().to(device)
    xy = torch.stack([X.flatten(), Y.flatten()], dim=1)
    with torch.no_grad():
        u_f, v_f = model(xy)
    U_mlp = u_f.reshape(Nx, Ny).detach().to(dtype)
    V_mlp = v_f.reshape(Nx, Ny).detach().to(dtype)

    U_pt3 = U_mlp.clone().requires_grad_(True)
    V_pt3 = V_mlp.clone().requires_grad_(True)
    loss_pt3 = pde_residual_pytorch(U_pt3, V_pt3, dx, dy)
    loss_pt3.backward()

    U_tr3 = U_mlp.clone().requires_grad_(True)
    V_tr3 = V_mlp.clone().requires_grad_(True)
    loss_tr3 = ldc_residual_triton(U_tr3, V_tr3, dx, dy, nu_val)
    loss_tr3.backward()

    eU3 = (U_pt3.grad - U_tr3.grad).abs().max().item()
    eV3 = (V_pt3.grad - V_tr3.grad).abs().max().item()
    print(f"  Grad U err: {eU3:.2e}")
    print(f"  Grad V err: {eV3:.2e}")
    print(f"  {'ALL PASS' if max(eU3,eV3) < 1e-5 else 'FAIL'}")

    import triton as _triton
    U_b = U_mlp.float(); V_b = V_mlp.float()
    ms_tr = _triton.testing.do_bench(lambda: ldc_residual_triton(U_b, V_b, dx, dy, nu_val))
    print(f"4. Triton forward: {ms_tr:.3f}ms")
