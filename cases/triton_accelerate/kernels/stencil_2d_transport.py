"""
2D Advection-Diffusion Triton kernels: Forward + Adjoint Backward.
PDE: c_t + u0*c_x + v0*c_y = nu*(c_xx + c_yy)
One scalar field C: [Nt, Nx, Ny]. Constant velocity (u0, v0).
"""
import torch
import triton
import triton.language as tl


# ── Forward kernel ─────────────────────────────────────────────────────────
@triton.jit
def advdiff_fwd_kernel(
    C_ptr, res_ptr,
    Nt: tl.constexpr, Nx: tl.constexpr, Ny: tl.constexpr,
    dx: tl.constexpr, dy: tl.constexpr, dt: tl.constexpr,
    u0: tl.constexpr, v0: tl.constexpr, nu: tl.constexpr,
    BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr,
):
    """Each program handles BLOCK_X x BLOCK_Y interior points at one t-slice."""
    pid_t = tl.program_id(0)
    pid_x = tl.program_id(1)
    pid_y = tl.program_id(2)
    it = pid_t + 1
    ix = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)[:, None] + 1
    iy = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)[None, :] + 1
    mask = (ix < Nx - 1) & (iy < Ny - 1)

    st = Nx * Ny; sx = Ny
    base = it * st + ix * sx + iy

    c_c  = tl.load(C_ptr + base,        mask=mask)
    c_xp = tl.load(C_ptr + base + sx,   mask=mask)
    c_xm = tl.load(C_ptr + base - sx,   mask=mask)
    c_yp = tl.load(C_ptr + base + 1,    mask=mask)
    c_ym = tl.load(C_ptr + base - 1,    mask=mask)
    c_tp = tl.load(C_ptr + base + st,   mask=mask)
    c_tm = tl.load(C_ptr + base - st,   mask=mask)

    c_t  = (c_tp - c_tm) / (2.0 * dt)
    c_x  = (c_xp - c_xm) / (2.0 * dx)
    c_y  = (c_yp - c_ym) / (2.0 * dy)
    c_xx = (c_xp - 2.0*c_c + c_xm) / (dx*dx)
    c_yy = (c_yp - 2.0*c_c + c_ym) / (dy*dy)

    res = c_t + u0*c_x + v0*c_y - nu*(c_xx + c_yy)

    ot = pid_t
    ox = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)[:, None]
    oy = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)[None, :]
    omask = (ox < Nx - 2) & (oy < Ny - 2)
    osx = Ny - 2
    oidx = ot * (Nx - 2) * (Ny - 2) + ox * osx + oy

    tl.store(res_ptr + oidx, res, mask=omask)


# ── Backward (adjoint) kernel ──────────────────────────────────────────────
@triton.jit
def advdiff_bwd_kernel(
    C_ptr, G_ptr, grad_ptr,
    Nt: tl.constexpr, Nx: tl.constexpr, Ny: tl.constexpr,
    dx: tl.constexpr, dy: tl.constexpr, dt: tl.constexpr,
    u0: tl.constexpr, v0: tl.constexpr, nu: tl.constexpr,
    BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr,
):
    """
    Adjoint of adv-diff stencil.
    For each interior point, accumulate contributions from all residuals
    that used C[it,ix,iy] in their computation.
    """
    pid_t = tl.program_id(0)
    pid_x = tl.program_id(1)
    pid_y = tl.program_id(2)
    it = pid_t + 1
    ix2 = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)[:, None] + 1
    iy2 = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)[None, :] + 1
    mask = (ix2 < Nx - 1) & (iy2 < Ny - 1)

    st = Nx * Ny; sx = Ny
    base = it * st + ix2 * sx + iy2

    # Load C stencil (not needed for linear adjoint, but needed for debugging)
    c_xp = tl.load(C_ptr + base + sx,  mask=mask)
    c_xm = tl.load(C_ptr + base - sx,  mask=mask)
    c_yp = tl.load(C_ptr + base + 1,   mask=mask)
    c_ym = tl.load(C_ptr + base - 1,   mask=mask)

    # Load upstream gradient G from interior grid
    osx = Ny - 2; ost = (Nx - 2) * (Ny - 2)
    tg = pid_t; xg = ix2 - 1; yg = iy2 - 1
    bg = tg * ost + xg * osx + yg

    m_tp = (tg + 1 < Nt - 2) & mask
    m_tm = (tg - 1 >= 0) & mask
    m_xp = (xg + 1 < Nx - 2) & mask
    m_xm = (xg - 1 >= 0) & mask
    m_yp = (yg + 1 < Ny - 2) & mask
    m_ym = (yg - 1 >= 0) & mask

    g_c  = tl.load(G_ptr + bg,           mask=mask, other=0.0)
    g_tp = tl.load(G_ptr + bg + ost,      mask=m_tp, other=0.0)
    g_tm = tl.load(G_ptr + bg - ost,      mask=m_tm, other=0.0)
    g_xp = tl.load(G_ptr + bg + osx,      mask=m_xp, other=0.0)
    g_xm = tl.load(G_ptr + bg - osx,      mask=m_xm, other=0.0)
    g_yp = tl.load(G_ptr + bg + 1,        mask=m_yp, other=0.0)
    g_ym = tl.load(G_ptr + bg - 1,        mask=m_ym, other=0.0)

    idx_val = 1.0/(2.0*dx); idy = 1.0/(2.0*dy); idt = 1.0/(2.0*dt)
    idx2 = 1.0/(dx*dx); idy2 = 1.0/(dy*dy)

    # Center: g_c * (u0*c_x term non-existent for center? No, c_x uses neighbors)
    # Forward: res = c_t + u0*c_x + v0*c_y - nu*(c_xx+c_yy)
    # d(res)/d(c_c) = u0*0 + v0*0 - nu*(-2/dx² - 2/dy²) = 2*nu*(1/dx² + 1/dy²)
    # Actually: c_x = (c_xp - c_xm)/(2dx) doesn't use c_c
    # c_xx = (c_xp - 2*c_c + c_xm)/dx², d(c_xx)/d(c_c) = -2/dx²
    # d(res)/d(c_c) = -nu*(-2/dx² - 2/dy²) = 2*nu*(1/dx² + 1/dy²)
    grad_val  = g_c * (2.0*nu*(idx2 + idy2))

    # Convection neighbors: d(u0*c_x)/d(c_xp) = u0/(2dx), d(u0*c_x)/d(c_xm) = -u0/(2dx)
    # C[it,ix,iy] as c_xp of res at (tg, xg-1, yg): +u0*g_xm/(2dx) + nu*g_xm/dx²
    # C[it,ix,iy] as c_xm of res at (tg, xg+1, yg): -u0*g_xp/(2dx) + nu*g_xp/dx²
    # Wait: d(c_xx)/d(c_xp) = 1/dx², d(-nu*c_xx)/d(c_xp) = -nu/dx²
    # So: +u0*g_xm/(2dx) - nu*g_xm/dx² (as c_xp neighbor)
    #     -u0*g_xp/(2dx) - nu*g_xp/dx² (as c_xm neighbor)

    # Combine: convection + diffusion neighbor contributions
    grad_val += (g_xm * (u0 * idx_val - nu * idx2)    # C is c_xp of res at tg,xg-1,yg
                 - g_xp * (u0 * idx_val + nu * idx2))   # C is c_xm of res at tg,xg+1,yg
    grad_val += (g_ym * (v0 * idy - nu * idy2)         # C is c_yp of res at tg,xg,yg-1
                 - g_yp * (v0 * idy + nu * idy2))       # C is c_ym of res at tg,xg,yg+1

    # Time neighbors
    grad_val += (g_tm - g_tp) * idt

    tl.store(grad_ptr + base, grad_val, mask=mask)


# ── Boundary gradient fix ──────────────────────────────────────────────────
def _add_boundary_gradients_advdiff(C, G, grad_c, dx, dy, dt, nu_val, u0_val, v0_val):
    """Add gradient contributions for boundary faces."""
    Nt, Nx, Ny = C.shape
    inv_2dx = 1.0/(2.0*dx); inv_2dy = 1.0/(2.0*dy); inv_2dt = 1.0/(2.0*dt)
    inv_dx2 = 1.0/(dx*dx); inv_dy2 = 1.0/(dy*dy)
    i = slice(1, -1)
    G4 = G.reshape(Nt - 2, Nx - 2, Ny - 2)

    # t=0 / t=Nt-1
    grad_c[0, i, i] += -G4[0] * inv_2dt
    grad_c[Nt-1, i, i] += G4[Nt-3] * inv_2dt

    # x=0: c is c_xm of res at xg=0 -> -u0*G/(2dx) - nu*G/dx²
    grad_c[i, 0, i] += -G4[:, 0] * (u0_val * inv_2dx + nu_val * inv_dx2)

    # x=Nx-1: c is c_xp of res at xg=Nx-3 -> +u0*G/(2dx) - nu*G/dx²
    grad_c[i, Nx-1, i] += G4[:, Nx-3] * (u0_val * inv_2dx - nu_val * inv_dx2)

    # y=0
    grad_c[i, i, 0] += -G4[:, :, 0] * (v0_val * inv_2dy + nu_val * inv_dy2)

    # y=Ny-1
    grad_c[i, i, Ny-1] += G4[:, :, Ny-3] * (v0_val * inv_2dy - nu_val * inv_dy2)


# ── Autograd Function ──────────────────────────────────────────────────────
class _AdvDiffTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C, dx, dy, dt, u0_val, v0_val, nu_val):
        C = C.contiguous()
        Nt, Nx, Ny = C.shape
        N_total = (Nt - 2) * (Nx - 2) * (Ny - 2)
        res = torch.empty(N_total, device=C.device, dtype=C.dtype)
        BLOCK_X, BLOCK_Y = 16, 16
        grid = (Nt - 2, (Nx - 2 + BLOCK_X - 1) // BLOCK_X, (Ny - 2 + BLOCK_Y - 1) // BLOCK_Y)
        advdiff_fwd_kernel[grid](C, res, Nt, Nx, Ny, dx, dy, dt, u0_val, v0_val, nu_val, BLOCK_X, BLOCK_Y)
        ctx.save_for_backward(C)
        ctx.res = res
        ctx.dx, ctx.dy, ctx.dt = dx, dy, dt
        ctx.u0_val, ctx.v0_val, ctx.nu_val = u0_val, v0_val, nu_val
        return res.pow(2).mean()

    @staticmethod
    def backward(ctx, grad_out):
        C, = ctx.saved_tensors
        res = ctx.res
        dx, dy, dt = ctx.dx, ctx.dy, ctx.dt
        u0_val, v0_val, nu_val = ctx.u0_val, ctx.v0_val, ctx.nu_val
        Nt, Nx, Ny = C.shape
        N_total = res.numel()

        scale = (2.0 / N_total) * grad_out
        G = res * scale

        grad_c = torch.zeros_like(C)
        BLOCK_X, BLOCK_Y = 16, 16
        grid = (Nt - 2, (Nx - 2 + BLOCK_X - 1) // BLOCK_X, (Ny - 2 + BLOCK_Y - 1) // BLOCK_Y)
        advdiff_bwd_kernel[grid](C, G, grad_c, Nt, Nx, Ny, dx, dy, dt,
                                 u0_val, v0_val, nu_val, BLOCK_X, BLOCK_Y)

        _add_boundary_gradients_advdiff(C, G, grad_c, dx, dy, dt, nu_val, u0_val, v0_val)

        return grad_c, None, None, None, None, None, None


def advdiff_residual_triton(C, dx, dy, dt, u0_val, v0_val, nu_val):
    return _AdvDiffTriton.apply(C.contiguous(), dx, dy, dt, u0_val, v0_val, nu_val)


# ── Correctness check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, numpy as np, os
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
    os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/triton_advdiff_test')

    from cases.transport_2d.physics import pde_residual_pytorch, make_grid, exact_c, MLP

    Nt, Nx, Ny = 20, 64, 64
    nu_val = 0.01; u0_val = 1.0; v0_val = 1.0
    device = "cuda"; dtype = torch.float64

    X, Y, T, dx, dy, dt = make_grid()
    C = (torch.sin(X - u0_val*T) * torch.sin(Y - v0_val*T) * torch.exp(-2*nu_val*T)).to(dtype)

    # Forward
    print("1. Forward check...")
    C_pt = C.clone().requires_grad_(True)
    loss_pt = pde_residual_pytorch(C_pt.float(), dx, dy, dt)
    C_tr = C.clone().requires_grad_(True)
    loss_tr = advdiff_residual_triton(C_tr.float(), dx, dy, dt, u0_val, v0_val, nu_val)
    print(f"  Forward rel_err: {abs(loss_pt.item()-loss_tr.item())/(abs(loss_pt.item())+1e-12):.2e}")

    # Gradient
    print("2. Gradient check (float64)...")
    C_pt2 = C.clone().requires_grad_(True)
    loss_pt2 = pde_residual_pytorch(C_pt2, dx, dy, dt)
    loss_pt2.backward()

    C_tr2 = C.clone().requires_grad_(True)
    loss_tr2 = advdiff_residual_triton(C_tr2, dx, dy, dt, u0_val, v0_val, nu_val)
    loss_tr2.backward()

    eC = (C_pt2.grad - C_tr2.grad).abs().max().item()
    print(f"  Grad C err (ALL): {eC:.2e}  {'PASS' if eC < 1e-5 else 'FAIL'}")

    # MLP field test
    print("3. MLP field test...")
    model = MLP().to(device)
    xyt = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)
    with torch.no_grad():
        c_f = model(xyt)
    C_mlp = c_f.reshape(Nt, Nx, Ny).detach().to(dtype)

    C_pt3 = C_mlp.clone().requires_grad_(True)
    loss_pt3 = pde_residual_pytorch(C_pt3, dx, dy, dt)
    loss_pt3.backward()

    C_tr3 = C_mlp.clone().requires_grad_(True)
    loss_tr3 = advdiff_residual_triton(C_tr3, dx, dy, dt, u0_val, v0_val, nu_val)
    loss_tr3.backward()

    eC3 = (C_pt3.grad - C_tr3.grad).abs().max().item()
    print(f"  Grad C err (ALL): {eC3:.2e}  {'PASS' if eC3 < 1e-5 else 'FAIL'}")

    import triton as _triton
    C_b = C_mlp.float()
    ms = _triton.testing.do_bench(lambda: advdiff_residual_triton(C_b, dx, dy, dt, u0_val, v0_val, nu_val))
    print(f"4. Triton forward: {ms:.3f}ms")
