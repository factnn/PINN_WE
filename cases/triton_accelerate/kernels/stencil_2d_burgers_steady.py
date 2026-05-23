"""
2D Steady-State NS Triton kernels for LDC: Forward + Adjoint Backward.
PDE: res_u = u*u_x + v*u_y + p_x - nu*(u_xx+u_yy)
     res_v = u*v_x + v*v_y + p_y - nu*(v_xx+v_yy)
     res_div = u_x + v_y
No time dimension. Grid [Nx, Ny].
"""
import torch
import triton
import triton.language as tl


# ── Forward kernel ─────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_X': 8, 'BLOCK_Y': 8}),
        triton.Config({'BLOCK_X': 16, 'BLOCK_Y': 16}),
        triton.Config({'BLOCK_X': 32, 'BLOCK_Y': 32}),
        triton.Config({'BLOCK_X': 16, 'BLOCK_Y': 32}),
    ],
    key=['Nx', 'Ny'],
)
@triton.jit
def ldc_fwd_kernel(
    U_ptr, V_ptr, P_ptr,
    resU_ptr, resV_ptr, resDiv_ptr,
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

    p_xp = tl.load(P_ptr + base + sx,  mask=mask)
    p_xm = tl.load(P_ptr + base - sx,  mask=mask)
    p_yp = tl.load(P_ptr + base + 1,   mask=mask)
    p_ym = tl.load(P_ptr + base - 1,   mask=mask)

    u_x  = (u_xp - u_xm) / (2.0 * dx)
    u_y  = (u_yp - u_ym) / (2.0 * dy)
    u_xx = (u_xp - 2.0*u_c + u_xm) / (dx*dx)
    u_yy = (u_yp - 2.0*u_c + u_ym) / (dy*dy)

    v_x  = (v_xp - v_xm) / (2.0 * dx)
    v_y  = (v_yp - v_ym) / (2.0 * dy)
    v_xx = (v_xp - 2.0*v_c + v_xm) / (dx*dx)
    v_yy = (v_yp - 2.0*v_c + v_ym) / (dy*dy)

    p_x = (p_xp - p_xm) / (2.0 * dx)
    p_y = (p_yp - p_ym) / (2.0 * dy)

    res_u   = u_c*u_x + v_c*u_y + p_x - nu*(u_xx + u_yy)
    res_v   = u_c*v_x + v_c*v_y + p_y - nu*(v_xx + v_yy)
    res_div = u_x + v_y

    # output indexing (interior only)
    ox = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)[:, None]
    oy = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)[None, :]
    omask = (ox < Nx - 2) & (oy < Ny - 2)
    osx = Ny - 2
    oidx = ox * osx + oy

    tl.store(resU_ptr   + oidx, res_u,   mask=omask)
    tl.store(resV_ptr   + oidx, res_v,   mask=omask)
    tl.store(resDiv_ptr + oidx, res_div, mask=omask)


# ── Backward kernel ────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_X': 8, 'BLOCK_Y': 8}),
        triton.Config({'BLOCK_X': 16, 'BLOCK_Y': 16}),
        triton.Config({'BLOCK_X': 32, 'BLOCK_Y': 32}),
        triton.Config({'BLOCK_X': 16, 'BLOCK_Y': 32}),
    ],
    key=['Nx', 'Ny'],
)
@triton.jit
def ldc_bwd_kernel(
    U_ptr, V_ptr,
    Gu_ptr, Gv_ptr, Gdiv_ptr,
    gradU_ptr, gradV_ptr, gradP_ptr,
    Nx: tl.constexpr, Ny: tl.constexpr,
    dx: tl.constexpr, dy: tl.constexpr, nu: tl.constexpr,
    BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr,
):
    """Adjoint of the steady NS stencil for U, V, P."""
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

    # Load upstream gradients from interior grid (Nx-2, Ny-2)
    osx = Ny - 2
    xg = ix2 - 1; yg = iy2 - 1
    bg = xg * osx + yg

    m_xp = (xg + 1 < Nx - 2) & mask
    m_xm = (xg - 1 >= 0) & mask
    m_yp = (yg + 1 < Ny - 2) & mask
    m_ym = (yg - 1 >= 0) & mask

    gu_c  = tl.load(Gu_ptr + bg,        mask=mask, other=0.0)
    gu_xp = tl.load(Gu_ptr + bg + osx,  mask=m_xp, other=0.0)
    gu_xm = tl.load(Gu_ptr + bg - osx,  mask=m_xm, other=0.0)
    gu_yp = tl.load(Gu_ptr + bg + 1,    mask=m_yp, other=0.0)
    gu_ym = tl.load(Gu_ptr + bg - 1,    mask=m_ym, other=0.0)

    gv_c  = tl.load(Gv_ptr + bg,        mask=mask, other=0.0)
    gv_xp = tl.load(Gv_ptr + bg + osx,  mask=m_xp, other=0.0)
    gv_xm = tl.load(Gv_ptr + bg - osx,  mask=m_xm, other=0.0)
    gv_yp = tl.load(Gv_ptr + bg + 1,    mask=m_yp, other=0.0)
    gv_ym = tl.load(Gv_ptr + bg - 1,    mask=m_ym, other=0.0)

    gd_c  = tl.load(Gdiv_ptr + bg,       mask=mask, other=0.0)
    gd_xp = tl.load(Gdiv_ptr + bg + osx, mask=m_xp, other=0.0)
    gd_xm = tl.load(Gdiv_ptr + bg - osx, mask=m_xm, other=0.0)
    gd_yp = tl.load(Gdiv_ptr + bg + 1,   mask=m_yp, other=0.0)
    gd_ym = tl.load(Gdiv_ptr + bg - 1,   mask=m_ym, other=0.0)

    idx_val = 1.0/(2.0*dx); idy = 1.0/(2.0*dy)
    idx2 = 1.0/(dx*dx); idy2 = 1.0/(dy*dy)

    # ── grad_U ────────────────────────────────────────────────────────────
    grad_u  = gu_c * (u_xp - u_xm) * idx_val           # ∂(u_c*u_x)/∂u_c
    grad_u += gv_c * (v_xp - v_xm) * idx_val           # ∂(u_c*v_x)/∂u_c in res_v
    grad_u += (gu_xm * u_xm - gu_xp * u_xp) * idx_val  # x-neighbor convective
    grad_u += (gu_ym * v_ym - gu_yp * v_yp) * idy      # y-neighbor (v_c*u_y)
    grad_u += -nu * (gu_xp + gu_xm - 2.0*gu_c) * idx2  # u_xx adjoint
    grad_u += -nu * (gu_yp + gu_ym - 2.0*gu_c) * idy2   # u_yy adjoint
    grad_u += (gd_xm - gd_xp) * idx_val                 # div u_x adjoint

    # ── grad_V ────────────────────────────────────────────────────────────
    grad_v  = gu_c * (u_yp - u_ym) * idy               # ∂(v_c*u_y)/∂v_c in res_u
    grad_v += gv_c * (v_yp - v_ym) * idy               # ∂(v_c*v_y)/∂v_c in res_v
    grad_v += (gv_xm * u_xm - gv_xp * u_xp) * idx_val  # x-neighbor (u_c*v_x, u_c from U)
    grad_v += (gv_ym * v_ym - gv_yp * v_yp) * idy      # y-neighbor convective
    grad_v += -nu * (gv_xp + gv_xm - 2.0*gv_c) * idx2  # v_xx adjoint
    grad_v += -nu * (gv_yp + gv_ym - 2.0*gv_c) * idy2   # v_yy adjoint
    grad_v += (gd_ym - gd_yp) * idy                     # div v_y adjoint

    # ── grad_P: adjoint of p_x in res_u, p_y in res_v ────────────────────
    grad_p  = (gu_xm - gu_xp) * idx_val                 # p_x adjoint
    grad_p += (gv_ym - gv_yp) * idy                     # p_y adjoint

    tl.store(gradU_ptr + base, grad_u, mask=mask)
    tl.store(gradV_ptr + base, grad_v, mask=mask)
    tl.store(gradP_ptr + base, grad_p, mask=mask)


# ── Boundary gradient fix ──────────────────────────────────────────────────
def _add_boundary_gradients_ldc(U, V, Gu, Gv, Gdiv, grad_u, grad_v, grad_p, dx, dy, nu_val):
    """Add gradient contributions for 4 boundary edges + P boundaries."""
    Nx, Ny = U.shape
    inv_2dx = 1.0/(2.0*dx); inv_2dy = 1.0/(2.0*dy)
    inv_dx2 = 1.0/(dx*dx); inv_dy2 = 1.0/(dy*dy)
    i = slice(1, -1)

    # x=0: U[0,j] is u_l in res_u[0,j-1] and v_l in res_v[0,j-1]
    # dRes_u/du_l = -u_c/(2dx) - nu/dx², where u_c = U[1,j]
    grad_u[0, i] += -Gu[0] * (U[1, i] * inv_2dx + nu_val * inv_dx2)
    grad_v[0, i] += -Gv[0] * (U[1, i] * inv_2dx + nu_val * inv_dx2)  # u_c from U!
    grad_u[0, i] += -Gdiv[0] * inv_2dx  # div adjoint

    # x=Nx-1: u_r
    grad_u[Nx-1, i] += Gu[Nx-3] * (U[Nx-2, i] * inv_2dx - nu_val * inv_dx2)
    grad_v[Nx-1, i] += Gv[Nx-3] * (U[Nx-2, i] * inv_2dx - nu_val * inv_dx2)
    grad_u[Nx-1, i] += Gdiv[Nx-3] * inv_2dx

    # y=0: u_ym
    grad_u[i, 0] += -Gu[:, 0] * (V[i, 1] * inv_2dy + nu_val * inv_dy2)
    grad_v[i, 0] += -Gv[:, 0] * (V[i, 1] * inv_2dy + nu_val * inv_dy2)
    grad_v[i, 0] += -Gdiv[:, 0] * inv_2dy

    # y=Ny-1: u_yp
    grad_u[i, Ny-1] += Gu[:, Ny-3] * (V[i, Ny-2] * inv_2dy - nu_val * inv_dy2)
    grad_v[i, Ny-1] += Gv[:, Ny-3] * (V[i, Ny-2] * inv_2dy - nu_val * inv_dy2)
    grad_v[i, Ny-1] += Gdiv[:, Ny-3] * inv_2dy

    # P boundaries: p_x/p_y adjoint
    grad_p[0, i] += -Gu[0] * inv_2dx
    grad_p[Nx-1, i] += Gu[Nx-3] * inv_2dx
    grad_p[i, 0] += -Gv[:, 0] * inv_2dy
    grad_p[i, Ny-1] += Gv[:, Ny-3] * inv_2dy


# ── Autograd Function ──────────────────────────────────────────────────────
class _LDCTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, V, P, dx, dy, nu_val):
        U, V, P = U.contiguous(), V.contiguous(), P.contiguous()
        Nx, Ny = U.shape
        res_u   = torch.empty((Nx-2, Ny-2), device=U.device, dtype=U.dtype)
        res_v   = torch.empty_like(res_u)
        res_div = torch.empty_like(res_u)
        grid = lambda meta: ((Nx-2+meta['BLOCK_X']-1)//meta['BLOCK_X'], (Ny-2+meta['BLOCK_Y']-1)//meta['BLOCK_Y'])
        ldc_fwd_kernel[grid](U, V, P, res_u, res_v, res_div,
                             Nx, Ny, dx, dy, nu_val)
        ctx.save_for_backward(U, V)
        ctx.res_u, ctx.res_v, ctx.res_div = res_u, res_v, res_div
        ctx.dx, ctx.dy, ctx.nu_val = dx, dy, nu_val
        return (res_u**2 + res_v**2 + res_div**2).mean()

    @staticmethod
    def backward(ctx, grad_out):
        U, V = ctx.saved_tensors
        res_u, res_v, res_div = ctx.res_u, ctx.res_v, ctx.res_div
        dx, dy, nu_val = ctx.dx, ctx.dy, ctx.nu_val
        Nx, Ny = U.shape
        N_total = res_u.numel()

        scale = (2.0 / N_total) * grad_out
        Gu = res_u * scale
        Gv = res_v * scale
        Gdiv = res_div * scale

        grad_u = torch.zeros_like(U)
        grad_v = torch.zeros_like(V)
        grad_p = torch.zeros_like(U)
        grid = lambda meta: ((Nx-2+meta['BLOCK_X']-1)//meta['BLOCK_X'], (Ny-2+meta['BLOCK_Y']-1)//meta['BLOCK_Y'])
        ldc_bwd_kernel[grid](U, V, Gu, Gv, Gdiv, grad_u, grad_v, grad_p,
                             Nx, Ny, dx, dy, nu_val)

        _add_boundary_gradients_ldc(U, V, Gu, Gv, Gdiv, grad_u, grad_v, grad_p, dx, dy, nu_val)

        return grad_u, grad_v, grad_p, None, None, None


def ldc_residual_triton(U, V, P, dx, dy, nu_val):
    return _LDCTriton.apply(U.contiguous(), V.contiguous(), P.contiguous(), dx, dy, nu_val)


# ── Correctness check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, numpy as np, os, importlib.util
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
    os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/triton_ldc_test')

    # Load self as proper module
    spec = importlib.util.spec_from_file_location('kernels.stencil_2d_burgers_steady', __file__)
    _self = importlib.util.module_from_spec(spec)
    sys.modules['kernels.stencil_2d_burgers_steady'] = _self
    spec.loader.exec_module(_self)
    ldc_residual_triton = _self.ldc_residual_triton

    from cases.ldc_2d.physics import pde_residual_pytorch, make_grid, MLP

    Nx, Ny = 64, 64
    nu_val = 0.01
    device = "cuda"
    dtype = torch.float64

    X, Y, dx, dy = make_grid()
    X, Y = X.to(dtype), Y.to(dtype)
    U = (torch.sin(2*np.pi*X) * torch.cos(2*np.pi*Y))
    V = (-torch.cos(2*np.pi*X) * torch.sin(2*np.pi*Y))
    P = (-0.25*(torch.cos(4*np.pi*X) + torch.cos(4*np.pi*Y)))

    # ── Test 1: Forward ──
    print("=== Test 1: Forward (sin/cos) ===")
    U_pt = U.clone().requires_grad_(True)
    V_pt = V.clone().requires_grad_(True)
    P_pt = P.clone().requires_grad_(True)
    U_tr = U.clone().requires_grad_(True)
    V_tr = V.clone().requires_grad_(True)
    P_tr = P.clone().requires_grad_(True)
    loss_pt = pde_residual_pytorch(U_pt, V_pt, P_pt, dx, dy)
    loss_tr = ldc_residual_triton(U_tr, V_tr, P_tr, dx, dy, nu_val)
    rel_err = abs(loss_pt.item()-loss_tr.item())/(abs(loss_pt.item())+1e-12)
    print(f"  PyTorch loss: {loss_pt.item():.12e}")
    print(f"  Triton  loss: {loss_tr.item():.12e}")
    print(f"  Rel err: {rel_err:.2e}")
    assert rel_err < 1e-12, f"Forward mismatch: {rel_err:.2e}"

    # ── Test 2: Gradient (sin/cos) ──
    print("\n=== Test 2: Gradient (sin/cos, float64) ===")
    U_pt2 = U.clone().requires_grad_(True)
    V_pt2 = V.clone().requires_grad_(True)
    P_pt2 = P.clone().requires_grad_(True)
    U_tr2 = U.clone().requires_grad_(True)
    V_tr2 = V.clone().requires_grad_(True)
    P_tr2 = P.clone().requires_grad_(True)
    loss_pt_d = pde_residual_pytorch(U_pt2, V_pt2, P_pt2, dx, dy)
    loss_tr_d = ldc_residual_triton(U_tr2, V_tr2, P_tr2, dx, dy, nu_val)
    loss_pt_d.backward(); loss_tr_d.backward()
    gu_err = (U_pt2.grad - U_tr2.grad).abs().max().item()
    gv_err = (V_pt2.grad - V_tr2.grad).abs().max().item()
    gp_err = (P_pt2.grad - P_tr2.grad).abs().max().item()
    print(f"  grad_U err: {gu_err:.2e}")
    print(f"  grad_V err: {gv_err:.2e}")
    print(f"  grad_P err: {gp_err:.2e}")

    # ── Test 3: MLP field ──
    print("\n=== Test 3: MLP output field ===")
    torch.manual_seed(42)
    xy = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)
    with torch.no_grad():
        model = MLP().to(device).to(dtype)
        u_f, v_f, p_f = model(xy)
    U_mlp = u_f.reshape(Nx, Ny).detach()
    V_mlp = v_f.reshape(Nx, Ny).detach()
    P_mlp = p_f.reshape(Nx, Ny).detach()

    U_pt3 = U_mlp.clone().requires_grad_(True)
    V_pt3 = V_mlp.clone().requires_grad_(True)
    P_pt3 = P_mlp.clone().requires_grad_(True)
    U_tr3 = U_mlp.clone().requires_grad_(True)
    V_tr3 = V_mlp.clone().requires_grad_(True)
    P_tr3 = P_mlp.clone().requires_grad_(True)
    loss_pt3 = pde_residual_pytorch(U_pt3, V_pt3, P_pt3, dx, dy)
    loss_tr3 = ldc_residual_triton(U_tr3, V_tr3, P_tr3, dx, dy, nu_val)
    loss_pt3.backward(); loss_tr3.backward()
    gu_mlp = (U_pt3.grad - U_tr3.grad).abs().max().item()
    gv_mlp = (V_pt3.grad - V_tr3.grad).abs().max().item()
    gp_mlp = (P_pt3.grad - P_tr3.grad).abs().max().item()
    print(f"  grad_U err: {gu_mlp:.2e}")
    print(f"  grad_V err: {gv_mlp:.2e}")
    print(f"  grad_P err: {gp_mlp:.2e}")

    # ── Summary ──
    print("\n=== Verification Summary ===")
    checks = [
        ("Forward rel diff < 1e-12", rel_err < 1e-12),
        ("grad_U err (sin/cos) < 1e-5", gu_err < 1e-5),
        ("grad_V err (sin/cos) < 1e-5", gv_err < 1e-5),
        ("grad_P err (sin/cos) < 1e-5", gp_err < 1e-5),
        ("grad_U err (MLP) < 1e-5", gu_mlp < 1e-5),
        ("grad_V err (MLP) < 1e-5", gv_mlp < 1e-5),
        ("grad_P err (MLP) < 1e-5", gp_mlp < 1e-5),
    ]
    all_pass = True
    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
        if not ok: all_pass = False

    # Benchmark
    print("\n=== Benchmark ===")
    U_b, V_b, P_b = U_mlp.float(), V_mlp.float(), P_mlp.float()
    ms_tr = triton.testing.do_bench(lambda: ldc_residual_triton(U_b, V_b, P_b, dx, dy, nu_val))
    ms_pt = triton.testing.do_bench(lambda: pde_residual_pytorch(U_b, V_b, P_b, dx, dy))
    print(f"  PyTorch FD: {ms_pt:.3f}ms")
    print(f"  Triton:     {ms_tr:.3f}ms")

    if all_pass:
        print("\nALL CHECKS PASSED")
    else:
        print("\nSOME CHECKS FAILED")
        sys.exit(1)
