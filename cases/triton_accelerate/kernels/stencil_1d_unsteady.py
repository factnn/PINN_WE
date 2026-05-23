"""
1D Triton stencil kernel for Burgers PDE residual.
Fuses: u_x (central diff) + u_xx (2nd diff) + nonlinear term into one kernel.
"""
import torch
import triton
import triton.language as tl

nu = 0.01 / 3.14159265358979


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 64}),
        triton.Config({'BLOCK': 128}),
        triton.Config({'BLOCK': 256}),
        triton.Config({'BLOCK': 512}),
    ],
    key=['N'],
)
@triton.jit
def burgers_residual_1d_kernel(
    u_ptr, res_ptr,
    dx: tl.constexpr,
    nu: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK) + 1
    mask = i < N - 1

    u_c = tl.load(u_ptr + i,     mask=mask)
    u_l = tl.load(u_ptr + i - 1, mask=mask)
    u_r = tl.load(u_ptr + i + 1, mask=mask)

    u_x  = (u_r - u_l) / (2.0 * dx)
    u_xx = (u_r - 2.0 * u_c + u_l) / (dx * dx)
    res  = u_c * u_x - nu * u_xx

    tl.store(res_ptr + (i - 1), res, mask=mask)


def burgers_residual_triton(u: torch.Tensor, dx: float, nu: float = 0.01 / 3.14159265358979) -> torch.Tensor:
    N = u.shape[0]
    res = torch.empty(N - 2, device=u.device, dtype=u.dtype)
    grid = lambda meta: ((N - 2 + meta['BLOCK'] - 1) // meta['BLOCK'],)
    burgers_residual_1d_kernel[grid](u, res, dx, nu, N)
    return res


# ── Backward kernel ────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 64}),
        triton.Config({'BLOCK': 128}),
        triton.Config({'BLOCK': 256}),
        triton.Config({'BLOCK': 512}),
    ],
    key=['N'],
)
@triton.jit
def burgers_bwd_kernel(
    u_ptr, G_ptr, grad_ptr,
    dx: tl.constexpr,
    nu: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Adjoint backward for 1D Burgers stencil.
    Computes grad_U[i] from U and upstream gradient G[i] = dLoss/dRes[i].
    Boundary: grad_U[0] = grad_U[N-1] = 0 (Dirichlet BC).
    """
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK) + 1  # interior points [1, N-2]
    mask = i < N - 1

    # Load U stencil
    u_c = tl.load(u_ptr + i,     mask=mask)
    u_r = tl.load(u_ptr + i + 1, mask=mask)
    u_l = tl.load(u_ptr + i - 1, mask=mask)

    # Load G: G[j] corresponds to res[j] = dLoss/dRes at interior point j
    # G is stored at offset (j-1) since res has N-2 elements
    # So G[i] is at G_ptr + (i-1), G[i-1] at G_ptr + (i-2), G[i+1] at G_ptr + i
    g_c = tl.load(G_ptr + (i - 1), mask=mask)  # G[i]

    # G[i-1] and G[i+1] need boundary masks
    mask_l = (i - 1 >= 1) & mask  # i>1 so G[i-1] exists
    mask_r = (i + 1 < N - 1) & mask  # i<N-2 so G[i+1] exists
    g_l = tl.load(G_ptr + (i - 2), mask=mask_l, other=0.0)  # G[i-1]
    g_r = tl.load(G_ptr + i,     mask=mask_r, other=0.0)  # G[i+1]

    # Combined formula:
    # grad_U[i] = G[i]*(u[i+1]-u[i-1])/(2dx)     convection center
    #           + 2*nu*G[i]/dx²                    diffusion center
    #           + G[i-1]*u[i-1]/(2dx) - nu*G[i-1]/dx²   contribution from res[i-1]
    #           - G[i+1]*u[i+1]/(2dx) - nu*G[i+1]/dx²   contribution from res[i+1]
    inv_2dx = 1.0 / (2.0 * dx)
    inv_dx2 = 1.0 / (dx * dx)

    grad_val = (g_c * (u_r - u_l) * inv_2dx
                + 2.0 * nu * g_c * inv_dx2
                + g_l * u_l * inv_2dx - nu * g_l * inv_dx2
                - g_r * u_r * inv_2dx - nu * g_r * inv_dx2)

    tl.store(grad_ptr + i, grad_val, mask=mask)


def burgers_backward_triton(u: torch.Tensor, res: torch.Tensor, dx: float, nu: float = 0.01 / 3.14159265358979) -> torch.Tensor:
    """Compute d(Loss)/dU given U and residual = res(u). Loss = mean(res²)."""
    N = u.shape[0]
    G = 2.0 * res / (N - 2)
    grad_u = torch.zeros(N, device=u.device, dtype=u.dtype)
    grid = lambda meta: ((N - 2 + meta['BLOCK'] - 1) // meta['BLOCK'],)
    burgers_bwd_kernel[grid](u, G, grad_u, dx, nu, N)

    # --- Boundary gradient contributions ---
    inv_2dx = 1.0 / (2.0 * dx)
    inv_dx2 = 1.0 / (dx * dx)
    grad_u[0] += G[0] * (-u[1] * inv_2dx - nu * inv_dx2)
    grad_u[-1] += G[-1] * (u[-2] * inv_2dx - nu * inv_dx2)

    return grad_u


# ── Autograd Function ──────────────────────────────────────────────────────
class _Burgers1DTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, dx, nu_val):
        u_c = u.contiguous()
        res = burgers_residual_triton(u_c, dx, nu_val)
        ctx.save_for_backward(u_c, res)
        ctx.dx, ctx.nu_val = dx, nu_val
        return res.pow(2).mean()

    @staticmethod
    def backward(ctx, grad_output):
        u, res = ctx.saved_tensors
        dx, nu_val = ctx.dx, ctx.nu_val
        grad_u = burgers_backward_triton(u, res, dx, nu_val)
        return grad_output * grad_u, None, None


def burgers_loss_triton_autograd(u: torch.Tensor, dx: float, nu_val: float = 0.01 / 3.14159265358979) -> torch.Tensor:
    """Triton-based loss with both forward and backward in Triton. Autograd-compatible."""
    return _Burgers1DTriton.apply(u, dx, nu_val)


# ── 2D time-dependent Burgers forward kernel (stores res, not res²) ───────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_X': 64}),
        triton.Config({'BLOCK_X': 128}),
        triton.Config({'BLOCK_X': 256}),
        triton.Config({'BLOCK_X': 512}),
    ],
    key=['Nx'],
)
@triton.jit
def burgers_residual_2d_raw_kernel(
    U_ptr, res_ptr,
    Nt: tl.constexpr, Nx: tl.constexpr,
    dx: tl.constexpr, dt: tl.constexpr, nu: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    """Forward kernel storing raw residual (not squared), for backward compat."""
    pid_t = tl.program_id(0)
    pid_x = tl.program_id(1)
    it = pid_t + 1
    ix = pid_x * BLOCK_X + tl.arange(0, BLOCK_X) + 1
    mask = (it < Nt - 1) & (ix < Nx - 1)

    stride = Nx
    u_c  = tl.load(U_ptr + it * stride + ix,       mask=mask)
    u_xp = tl.load(U_ptr + it * stride + ix + 1,   mask=mask)
    u_xm = tl.load(U_ptr + it * stride + ix - 1,   mask=mask)
    u_tp = tl.load(U_ptr + (it + 1) * stride + ix, mask=mask)
    u_tm = tl.load(U_ptr + (it - 1) * stride + ix, mask=mask)

    u_x  = (u_xp - u_xm) / (2.0 * dx)
    u_xx = (u_xp - 2.0 * u_c + u_xm) / (dx * dx)
    u_t  = (u_tp - u_tm) / (2.0 * dt)

    res = u_t + u_c * u_x - nu * u_xx

    out_t = pid_t
    out_x = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    out_mask = out_x < Nx - 2
    tl.store(res_ptr + out_t * (Nx - 2) + out_x, res, mask=out_mask)


# ── 2D time-dependent Burgers backward ────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_X': 64}),
        triton.Config({'BLOCK_X': 128}),
        triton.Config({'BLOCK_X': 256}),
        triton.Config({'BLOCK_X': 512}),
    ],
    key=['Nx'],
)
@triton.jit
def burgers_2d_bwd_kernel(
    u_ptr, G_ptr, grad_ptr,
    Nt: tl.constexpr, Nx: tl.constexpr,
    dx: tl.constexpr, dt: tl.constexpr, nu: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    """
    Backward for 2D Burgers PDE: res = u_t + u*u_x - nu*u_xx.
    Stencil: center (i,j), left/right (i,j±1), up/down (i±1,j).
    """
    pid_t = tl.program_id(0)
    pid_x = tl.program_id(1)

    i = pid_t + 1           # t index
    j = pid_x * BLOCK_X + tl.arange(0, BLOCK_X) + 1  # x index
    mask = (i < Nt - 1) & (j < Nx - 1)

    # Load U stencil
    u_c  = tl.load(u_ptr + i * Nx + j,       mask=mask)
    u_l  = tl.load(u_ptr + i * Nx + (j - 1), mask=mask)
    u_r  = tl.load(u_ptr + i * Nx + (j + 1), mask=mask)
    u_tp = tl.load(u_ptr + (i + 1) * Nx + j, mask=mask)
    u_tm = tl.load(u_ptr + (i - 1) * Nx + j, mask=mask)

    # Load G from res (stored at [i-1, j-1])
    g_idx = (i - 1) * (Nx - 2) + (j - 1)
    g_c = tl.load(G_ptr + g_idx, mask=mask)

    # Spatial neighbors
    g_l = tl.load(G_ptr + g_idx - 1,     mask=(j > 1) & mask, other=0.0)
    g_r = tl.load(G_ptr + g_idx + 1,     mask=(j < Nx - 2) & mask, other=0.0)
    g_tp = tl.load(G_ptr + g_idx + (Nx - 2), mask=(i < Nt - 2) & mask, other=0.0)
    g_tm = tl.load(G_ptr + g_idx - (Nx - 2), mask=(i > 1) & mask, other=0.0)

    inv_2dx = 1.0 / (2.0 * dx)
    inv_dx2 = 1.0 / (dx * dx)
    inv_2dt = 1.0 / (2.0 * dt)

    # CENTER: from res[i,j] center + others contributing to u[i,j]
    grad_val = g_c * ((u_r - u_l) * inv_2dx + 2.0 * nu * inv_dx2)
    # LEFT: from res[i,j-1] as u_right
    grad_val += g_l * (u_l * inv_2dx - nu * inv_dx2)
    # RIGHT: from res[i,j+1] as u_left
    grad_val += g_r * (-u_r * inv_2dx - nu * inv_dx2)
    # TIME_UP: from res[i+1,j] as u_tm
    grad_val += g_tp * (-inv_2dt)
    # TIME_DOWN: from res[i-1,j] as u_tp
    grad_val += g_tm * (inv_2dt)

    tl.store(grad_ptr + i * Nx + j, grad_val, mask=mask)


def burgers_2d_backward_triton(U: torch.Tensor, res: torch.Tensor, dx: float, dt: float, nu_val: float = 0.01 / 3.14159265358979) -> torch.Tensor:
    """Compute d(Loss)/dU for 2D (t,x) Burgers PDE. Loss = mean(res²)."""
    Nt, Nx = U.shape
    N_res = res.numel()
    G = 2.0 * res / N_res  # upstream gradient
    grad_U = torch.zeros(Nt, Nx, device=U.device, dtype=U.dtype)
    grid = lambda meta: (Nt - 2, (Nx - 2 + meta['BLOCK_X'] - 1) // meta['BLOCK_X'])
    burgers_2d_bwd_kernel[grid](U, G, grad_U, Nt, Nx, dx, dt, nu_val)

    # --- Boundary gradient contributions ---
    # The kernel only covers interior i in [1,Nt-2], j in [1,Nx-2].
    # Boundary points (t=0, t=Nt-1, x=0, x=Nx-1) receive PDE gradient from
    # being neighbors in the stencil. We compute these analytically.
    inv_2dx = 1.0 / (2.0 * dx)
    inv_dx2 = 1.0 / (dx * dx)
    inv_2dt = 1.0 / (2.0 * dt)

    # t=0 boundary: U[0, 1:Nx-1] appears as u_tm in res[0, :]
    # dLoss/dU[0,j] from res[0,j-1] as u_tm = -G[0,j-1] / (2*dt)
    grad_U[0, 1:-1] += -G[0, :] * inv_2dt

    # t=Nt-1 boundary: U[Nt-1, 1:Nx-1] appears as u_tp in res[Nt-3, :]
    # dLoss/dU[Nt-1,j] from res[Nt-3,j-1] as u_tp = G[Nt-3,:] / (2*dt)
    grad_U[Nt-1, 1:-1] += G[Nt-3, :] * inv_2dt

    # x=0 boundary: U[1:Nt-1, 0] appears as u_l/u_xm in res[:, 0]
    # dLoss/dU[i,0]:
    #   from res[i-1,0] as u_l (u_x term): -G[i-1,0] * U[i,1]/(2dx)  (wait: res uses u_c*u_x)
    #   from res[i-1,0] as u_xx left: -nu * G[i-1,0] / dx²
    # More precisely: res[t,x] uses u_c at (t+1,x+1), u_l at (t+1,x), u_r at (t+1,x+2)
    #   dRes/du_l = -u_c/(2dx) - nu/dx²
    # U[i,0] is u_l of res[i-1,0]: -G[i-1,0] * (U[i,1]/(2dx) + nu/dx²)
    if Nx > 2:
        grad_U[1:-1, 0] += -G[:, 0] * (U[1:-1, 1] * inv_2dx + nu_val * inv_dx2)

    # x=Nx-1 boundary: U[1:Nt-1, Nx-1] appears as u_r/u_xp in res[:, Nx-3]
    #   dRes/du_r = u_c/(2dx) - nu/dx²
    # U[i,Nx-1] is u_r of res[i-1,Nx-3]: G[i-1,Nx-3] * (U[i,Nx-2]/(2dx) - nu/dx²)
    if Nx > 2:
        grad_U[1:-1, Nx-1] += G[:, Nx-3] * (U[1:-1, Nx-2] * inv_2dx - nu_val * inv_dx2)

    return grad_U


# ── 2D Autograd Function ─────────────────────────────────────────────────
class _Burgers2DTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, dx, dt, nu_val):
        U_c = U.contiguous()
        Nt_, Nx_ = U_c.shape
        res = torch.empty((Nt_ - 2, Nx_ - 2), device=U.device, dtype=U.dtype)
        grid = lambda meta: (Nt_ - 2, (Nx_ - 2 + meta['BLOCK_X'] - 1) // meta['BLOCK_X'])
        burgers_residual_2d_raw_kernel[grid](U_c, res, Nt_, Nx_, dx, dt, nu_val)
        ctx.save_for_backward(U_c, res)
        ctx.dx, ctx.dt, ctx.nu_val = dx, dt, nu_val
        return res.pow(2).mean()

    @staticmethod
    def backward(ctx, grad_output):
        U, res = ctx.saved_tensors
        dx, dt, nu_val = ctx.dx, ctx.dt, ctx.nu_val
        grad_U = burgers_2d_backward_triton(U, res, dx, dt, nu_val)
        return grad_output * grad_U, None, None, None


def burgers_2d_loss_triton_autograd(U: torch.Tensor, dx: float, dt: float, nu_val: float = 0.01 / 3.14159265358979) -> torch.Tensor:
    """Triton 2D Burgers loss with full Triton forward+backward. Autograd-compatible."""
    return _Burgers2DTriton.apply(U, dx, dt, nu_val)


# ── Correctness check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    N = 1024
    device = "cuda"
    dtype = torch.float64
    x = torch.linspace(-1, 1, N, device=device, dtype=dtype)
    dx = float(x[1] - x[0])
    u = torch.sin(np.pi * x).to(dtype)

    # ── 1D Forward check ──
    print("=" * 60)
    print("1. 1D Forward check")
    u_x_ref  = (u[2:] - u[:-2]) / (2 * dx)
    u_xx_ref = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    res_ref  = u[1:-1] * u_x_ref - nu * u_xx_ref
    res_tri = burgers_residual_triton(u, dx)
    err = (res_tri - res_ref).abs().max().item()
    print(f"   Max error vs PyTorch: {err:.2e}  {'PASS' if err < 1e-6 else 'FAIL'}")

    # ── 1D Gradient check ──
    print("=" * 60)
    print("2. 1D Gradient check")
    u_pt = u.detach().clone().requires_grad_(True)
    u_x_pt  = (u_pt[2:] - u_pt[:-2]) / (2 * dx)
    u_xx_pt = (u_pt[2:] - 2 * u_pt[1:-1] + u_pt[:-2]) / dx**2
    res_pt = u_pt[1:-1] * u_x_pt - nu * u_xx_pt
    loss_pt = res_pt.pow(2).mean()
    loss_pt.backward()

    u_tr = u.detach().clone().requires_grad_(True)
    loss_tr = burgers_loss_triton_autograd(u_tr, dx, nu)
    loss_tr.backward()

    grad_err = (u_pt.grad[1:-1] - u_tr.grad[1:-1]).abs().max().item()
    boundary_err = (u_tr.grad[0].abs() + u_tr.grad[-1].abs()).item()
    print(f"   Max grad err (interior): {grad_err:.2e}  {'PASS' if grad_err < 1e-5 else 'FAIL'}")
    print(f"   Boundary grad = 0:       {boundary_err:.2e}  {'PASS' if boundary_err < 1e-10 else 'FAIL'}")
    print(f"   Loss match: {abs(loss_pt.item() - loss_tr.item()):.2e}")

    # ── 2D Gradient check ──
    print("=" * 60)
    print("3. 2D (t,x) Gradient check")
    Nt_, Nx_ = 100, 256
    t_vals = torch.linspace(0, 1, Nt_, device=device, dtype=dtype)
    x_vals = torch.linspace(-1, 1, Nx_, device=device, dtype=dtype)
    dt_ = float(t_vals[1] - t_vals[0])
    dx_ = float(x_vals[1] - x_vals[0])
    T, X = torch.meshgrid(t_vals, x_vals, indexing='ij')
    U_field = (torch.sin(np.pi * X) * torch.cos(0.5 * np.pi * T)).to(dtype)

    # PyTorch reference for 2D
    U_pt2 = U_field.detach().clone().requires_grad_(True)
    u_x_2 = (U_pt2[:, 2:] - U_pt2[:, :-2]) / (2 * dx_)
    u_xx_2 = (U_pt2[:, 2:] - 2 * U_pt2[:, 1:-1] + U_pt2[:, :-2]) / dx_**2
    u_t_2 = (U_pt2[2:, 1:-1] - U_pt2[:-2, 1:-1]) / (2 * dt_)
    res_2d = u_t_2 + U_pt2[1:-1, 1:-1] * u_x_2[1:-1] - nu * u_xx_2[1:-1]
    loss_2d_pt = res_2d.pow(2).mean()
    loss_2d_pt.backward()

    # Triton 2D
    U_tr2 = U_field.detach().clone().requires_grad_(True)
    loss_2d_tr = burgers_2d_loss_triton_autograd(U_tr2, dx_, dt_, nu)
    loss_2d_tr.backward()

    grad_err_2d = (U_pt2.grad[1:-1, 1:-1] - U_tr2.grad[1:-1, 1:-1]).abs().max().item()
    print(f"   Max grad err (interior): {grad_err_2d:.2e}  {'PASS' if grad_err_2d < 1e-5 else 'FAIL'}")
    print(f"   Loss match: {abs(loss_2d_pt.item() - loss_2d_tr.item()):.2e}")

    # ── Benchmark ──
    print("=" * 60)
    print("4. Benchmark (1D)")
    u_f32 = u.float()
    ms_torch = triton.testing.do_bench(lambda: (
        (u_f32[2:] - u_f32[:-2]) / (2*dx) * u_f32[1:-1] - nu * (u_f32[2:] - 2*u_f32[1:-1] + u_f32[:-2]) / dx**2
    ).pow(2).mean())
    ms_triton = triton.testing.do_bench(lambda: burgers_loss_triton_autograd(u_f32, dx, nu))
    print(f"   PyTorch forward : {ms_torch:.3f} ms")
    print(f"   Triton  forward : {ms_triton:.3f} ms")
    print(f"   Speedup         : {ms_torch/ms_triton:.2f}x")

    def pt_fwd_bwd():
        u_tmp = u_f32.detach().clone().requires_grad_(True)
        u_x_t  = (u_tmp[2:] - u_tmp[:-2]) / (2*dx)
        u_xx_t = (u_tmp[2:] - 2*u_tmp[1:-1] + u_tmp[:-2]) / dx**2
        res_t = u_tmp[1:-1] * u_x_t - nu * u_xx_t
        l = res_t.pow(2).mean()
        l.backward()

    def triton_fwd_bwd():
        u_tmp = u_f32.detach().clone().requires_grad_(True)
        l = burgers_loss_triton_autograd(u_tmp, dx, nu)
        l.backward()

    ms_pt_fb = triton.testing.do_bench(pt_fwd_bwd)
    ms_tr_fb = triton.testing.do_bench(triton_fwd_bwd)
    print(f"   PyTorch fwd+bwd : {ms_pt_fb:.3f} ms")
    print(f"   Triton  fwd+bwd : {ms_tr_fb:.3f} ms")
    print(f"   Fwd+Bwd speedup : {ms_pt_fb/ms_tr_fb:.2f}x")

    print("=" * 60)
    print("5. Benchmark (2D t,x)")
    U_f32 = U_field.float()

    def pt_2d_fwd_bwd():
        U_tmp = U_f32.detach().clone().requires_grad_(True)
        ux_t = (U_tmp[:, 2:] - U_tmp[:, :-2]) / (2*dx_)
        uxx_t = (U_tmp[:, 2:] - 2*U_tmp[:, 1:-1] + U_tmp[:, :-2]) / dx_**2
        ut_t = (U_tmp[2:, 1:-1] - U_tmp[:-2, 1:-1]) / (2*dt_)
        r_t = ut_t + U_tmp[1:-1, 1:-1] * ux_t[1:-1] - nu * uxx_t[1:-1]
        l = r_t.pow(2).mean()
        l.backward()

    def triton_2d_fwd_bwd():
        U_tmp = U_f32.detach().clone().requires_grad_(True)
        l = burgers_2d_loss_triton_autograd(U_tmp, dx_, dt_, nu)
        l.backward()

    ms_pt_2d = triton.testing.do_bench(pt_2d_fwd_bwd)
    ms_tr_2d = triton.testing.do_bench(triton_2d_fwd_bwd)
    print(f"   PyTorch fwd+bwd : {ms_pt_2d:.3f} ms")
    print(f"   Triton  fwd+bwd : {ms_tr_2d:.3f} ms")
    print(f"   Fwd+Bwd speedup : {ms_pt_2d/ms_tr_2d:.2f}x")



#     """
# 2D Triton stencil kernel for Navier-Stokes PDE (Forward + Adjoint Backward).
# Fuses all derivatives, advection, and diffusion into single passes.
# """
# import torch
# import triton
# import triton.language as tl
# import numpy as np

# # ==========================================
# # 1. 前向计算 Kernel (Forward)
# # ==========================================
# @triton.jit
# def ns2d_fwd_kernel(
#     U_ptr, V_ptr, P_ptr,
#     resU_ptr, resV_ptr, resDiv_ptr,
#     Nt: tl.constexpr, Nx: tl.constexpr, Ny: tl.constexpr,
#     dx: tl.constexpr, dy: tl.constexpr, dt: tl.constexpr, nu: tl.constexpr,
#     BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr,
# ):
#     pid_t = tl.program_id(0)
#     pid_x = tl.program_id(1)
#     pid_y = tl.program_id(2)

#     it = pid_t + 1
#     ix = pid_x * BLOCK_X + tl.arange(0, BLOCK_X) + 1
#     iy = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y) + 1
    
#     ix2 = ix[:, None]
#     iy2 = iy[None, :]
#     mask = (ix2 < Nx - 1) & (iy2 < Ny - 1)

#     stride_t = Nx * Ny
#     stride_x = Ny
#     base = it * stride_t + ix2 * stride_x + iy2

#     # Load 7-point stencil for U
#     u_c  = tl.load(U_ptr + base, mask=mask)
#     u_xp = tl.load(U_ptr + base + stride_x, mask=mask)
#     u_xm = tl.load(U_ptr + base - stride_x, mask=mask)
#     u_yp = tl.load(U_ptr + base + 1, mask=mask)
#     u_ym = tl.load(U_ptr + base - 1, mask=mask)
#     u_tp = tl.load(U_ptr + base + stride_t, mask=mask)
#     u_tm = tl.load(U_ptr + base - stride_t, mask=mask)

#     # Load 7-point stencil for V
#     v_c  = tl.load(V_ptr + base, mask=mask)
#     v_xp = tl.load(V_ptr + base + stride_x, mask=mask)
#     v_xm = tl.load(V_ptr + base - stride_x, mask=mask)
#     v_yp = tl.load(V_ptr + base + 1, mask=mask)
#     v_ym = tl.load(V_ptr + base - 1, mask=mask)
#     v_tp = tl.load(V_ptr + base + stride_t, mask=mask)
#     v_tm = tl.load(V_ptr + base - stride_t, mask=mask)

#     # Load 5-point stencil for P
#     p_xp = tl.load(P_ptr + base + stride_x, mask=mask)
#     p_xm = tl.load(P_ptr + base - stride_x, mask=mask)
#     p_yp = tl.load(P_ptr + base + 1, mask=mask)
#     p_ym = tl.load(P_ptr + base - 1, mask=mask)

#     # Derivatives
#     u_t = (u_tp - u_tm) / (2.0 * dt)
#     u_x = (u_xp - u_xm) / (2.0 * dx)
#     u_y = (u_yp - u_ym) / (2.0 * dy)
#     u_xx = (u_xp - 2.0 * u_c + u_xm) / (dx * dx)
#     u_yy = (u_yp - 2.0 * u_c + u_ym) / (dy * dy)

#     v_t = (v_tp - v_tm) / (2.0 * dt)
#     v_x = (v_xp - v_xm) / (2.0 * dx)
#     v_y = (v_yp - v_ym) / (2.0 * dy)
#     v_xx = (v_xp - 2.0 * v_c + v_xm) / (dx * dx)
#     v_yy = (v_yp - 2.0 * v_c + v_ym) / (dy * dy)

#     p_x = (p_xp - p_xm) / (2.0 * dx)
#     p_y = (p_yp - p_ym) / (2.0 * dy)

#     # Residuals
#     res_u = u_t + u_c * u_x + v_c * u_y + p_x - nu * (u_xx + u_yy)
#     res_v = v_t + u_c * v_x + v_c * v_y + p_y - nu * (v_xx + v_yy)
#     res_div = u_x + v_y

#     # Store to HBM (Only save the bare minimum for backward)
#     out_stride_t = (Nx - 2) * (Ny - 2)
#     out_stride_x = Ny - 2
#     out_idx = pid_t * out_stride_t + (ix2 - 1) * out_stride_x + (iy2 - 1)
#     out_mask = (ix2 - 1 >= 0) & (ix2 - 1 < Nx - 2) & (iy2 - 1 >= 0) & (iy2 - 1 < Ny - 2)

#     tl.store(resU_ptr + out_idx, res_u, mask=out_mask)
#     tl.store(resV_ptr + out_idx, res_v, mask=out_mask)
#     tl.store(resDiv_ptr + out_idx, res_div, mask=out_mask)

# # ==========================================
# # 2. 伴随反向 Kernel (Backward Adjoint)
# # ==========================================
# @triton.jit
# def ns2d_bwd_kernel(
#     U_ptr, V_ptr,
#     Gu_ptr, Gv_ptr, Gdiv_ptr,
#     gradU_ptr, gradV_ptr, gradP_ptr,
#     Nt: tl.constexpr, Nx: tl.constexpr, Ny: tl.constexpr,
#     dx: tl.constexpr, dy: tl.constexpr, dt: tl.constexpr, nu: tl.constexpr,
#     BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr
# ):
#     pid_t = tl.program_id(0)
#     pid_x = tl.program_id(1)
#     pid_y = tl.program_id(2)

#     it = pid_t + 1
#     ix = pid_x * BLOCK_X + tl.arange(0, BLOCK_X) + 1
#     iy = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y) + 1
#     ix2, iy2 = ix[:, None], iy[None, :]
#     mask = (ix2 < Nx - 1) & (iy2 < Ny - 1)

#     stride_t, stride_x = Nx * Ny, Ny
#     base = it * stride_t + ix2 * stride_x + iy2

#     # 1. 读取前向原始状态 (只需要空间 5 点模板，无需时间邻居)
#     u_xp = tl.load(U_ptr + base + stride_x, mask=mask)
#     u_xm = tl.load(U_ptr + base - stride_x, mask=mask)
#     u_yp = tl.load(U_ptr + base + 1, mask=mask)
#     u_ym = tl.load(U_ptr + base - 1, mask=mask)
    
#     v_xp = tl.load(V_ptr + base + stride_x, mask=mask)
#     v_xm = tl.load(V_ptr + base - stride_x, mask=mask)
#     v_yp = tl.load(V_ptr + base + 1, mask=mask)
#     v_ym = tl.load(V_ptr + base - 1, mask=mask)

#     # 2. 读取上游梯度场 G (注意边界溢出保护 other=0.0)
#     stride_g_t, stride_g_x = (Nx - 2) * (Ny - 2), Ny - 2
#     t_g, x_g, y_g = it - 1, ix2 - 1, iy2 - 1
#     base_g = t_g * stride_g_t + x_g * stride_g_x + y_g

#     m_tp = (t_g + 1 < Nt - 2) & mask
#     m_tm = (t_g - 1 >= 0) & mask
#     m_xp = (x_g + 1 < Nx - 2) & mask
#     m_xm = (x_g - 1 >= 0) & mask
#     m_yp = (y_g + 1 < Ny - 2) & mask
#     m_ym = (y_g - 1 >= 0) & mask

#     gu_c  = tl.load(Gu_ptr + base_g, mask=mask)
#     gu_tp = tl.load(Gu_ptr + base_g + stride_g_t, mask=m_tp, other=0.0)
#     gu_tm = tl.load(Gu_ptr + base_g - stride_g_t, mask=m_tm, other=0.0)
#     gu_xp = tl.load(Gu_ptr + base_g + stride_g_x, mask=m_xp, other=0.0)
#     gu_xm = tl.load(Gu_ptr + base_g - stride_g_x, mask=m_xm, other=0.0)
#     gu_yp = tl.load(Gu_ptr + base_g + 1, mask=m_yp, other=0.0)
#     gu_ym = tl.load(Gu_ptr + base_g - 1, mask=m_ym, other=0.0)

#     gv_c  = tl.load(Gv_ptr + base_g, mask=mask)
#     gv_tp = tl.load(Gv_ptr + base_g + stride_g_t, mask=m_tp, other=0.0)
#     gv_tm = tl.load(Gv_ptr + base_g - stride_g_t, mask=m_tm, other=0.0)
#     gv_xp = tl.load(Gv_ptr + base_g + stride_g_x, mask=m_xp, other=0.0)
#     gv_xm = tl.load(Gv_ptr + base_g - stride_g_x, mask=m_xm, other=0.0)
#     gv_yp = tl.load(Gv_ptr + base_g + 1, mask=m_yp, other=0.0)
#     gv_ym = tl.load(Gv_ptr + base_g - 1, mask=m_ym, other=0.0)

#     gdiv_xp = tl.load(Gdiv_ptr + base_g + stride_g_x, mask=m_xp, other=0.0)
#     gdiv_xm = tl.load(Gdiv_ptr + base_g - stride_g_x, mask=m_xm, other=0.0)
#     gdiv_yp = tl.load(Gdiv_ptr + base_g + 1, mask=m_yp, other=0.0)
#     gdiv_ym = tl.load(Gdiv_ptr + base_g - 1, mask=m_ym, other=0.0)

#     # 3. 伴随算子矩阵向量乘 (Adjoint Math)
#     idx2, idy2 = 1.0 / (dx * dx), 1.0 / (dy * dy)
#     idx, idy, idt = 1.0 / (2.0 * dx), 1.0 / (2.0 * dy), 1.0 / (2.0 * dt)

#     # Pressure adjoint (纯梯度转置)
#     grad_p = (gu_xm - gu_xp) * idx + (gv_ym - gv_yp) * idy

#     # U velocity adjoint (融合时间、对流与扩散)
#     grad_u = gv_c * (v_xp - v_xm) * idx \
#            + gu_c * ((u_xp - u_xm) * idx + 2.0 * nu * (idx2 + idy2)) \
#            - gu_tp * idt + gu_tm * idt \
#            - gu_xp * (u_xp * idx + nu * idx2) + gu_xm * (u_xm * idx - nu * idx2) \
#            - gu_yp * (v_yp * idy + nu * idy2) + gu_ym * (v_ym * idy - nu * idy2) \
#            - gdiv_xp * idx + gdiv_xm * idx

#     # V velocity adjoint
#     grad_v = gu_c * (u_yp - u_ym) * idy \
#            + gv_c * ((v_yp - v_ym) * idy + 2.0 * nu * (idx2 + idy2)) \
#            - gv_tp * idt + gv_tm * idt \
#            - gv_xp * (u_xp * idx + nu * idx2) + gv_xm * (u_xm * idx - nu * idx2) \
#            - gv_yp * (v_yp * idy + nu * idy2) + gv_ym * (v_ym * idy - nu * idy2) \
#            - gdiv_yp * idy + gdiv_ym * idy

#     # Store gradients
#     tl.store(gradU_ptr + base, grad_u, mask=mask)
#     tl.store(gradV_ptr + base, grad_v, mask=mask)
#     tl.store(gradP_ptr + base, grad_p, mask=mask)


# # ==========================================
# # 3. PyTorch Autograd 封装盒子
# # ==========================================
# class _NS2DTriton(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, U, V, P, dx, dy, dt, nu):
#         U, V, P = U.contiguous(), V.contiguous(), P.contiguous()
#         Nt, Nx, Ny = U.shape
        
#         res_u = torch.empty((Nt-2, Nx-2, Ny-2), device=U.device, dtype=U.dtype)
#         res_v = torch.empty_like(res_u)
#         res_div = torch.empty_like(res_u)

#         BLOCK_X, BLOCK_Y = 16, 16
#         grid = (Nt - 2, (Nx - 2 + BLOCK_X - 1) // BLOCK_X, (Ny - 2 + BLOCK_Y - 1) // BLOCK_Y)

#         ns2d_fwd_kernel[grid](
#             U, V, P, res_u, res_v, res_div,
#             Nt, Nx, Ny, dx, dy, dt, nu, BLOCK_X, BLOCK_Y
#         )

#         ctx.save_for_backward(U, V) # P 场不参与梯度收集，无需保存
#         ctx.res_u, ctx.res_v, ctx.res_div = res_u, res_v, res_div
#         ctx.dx, ctx.dy, ctx.dt, ctx.nu = dx, dy, dt, nu

#         return (res_u**2 + res_v**2 + res_div**2).mean()

#     @staticmethod
#     def backward(ctx, grad_output):
#         U, V = ctx.saved_tensors
#         res_u, res_v, res_div = ctx.res_u, ctx.res_v, ctx.res_div
#         dx, dy, dt, nu = ctx.dx, ctx.dy, ctx.dt, ctx.nu
#         Nt, Nx, Ny = U.shape

#         # 解析标量 grad_output 并将其扩展为上游残差的梯度 (Gu, Gv, Gdiv)
#         scale = (2.0 / res_u.numel()) * grad_output
#         Gu, Gv, Gdiv = res_u * scale, res_v * scale, res_div * scale

#         grad_u = torch.zeros_like(U)
#         grad_v = torch.zeros_like(V)
#         grad_p = torch.zeros_like(U)

#         BLOCK_X, BLOCK_Y = 16, 16
#         grid = (Nt - 2, (Nx - 2 + BLOCK_X - 1) // BLOCK_X, (Ny - 2 + BLOCK_Y - 1) // BLOCK_Y)

#         ns2d_bwd_kernel[grid](
#             U, V, Gu, Gv, Gdiv, grad_u, grad_v, grad_p,
#             Nt, Nx, Ny, dx, dy, dt, nu, BLOCK_X, BLOCK_Y
#         )

#         return grad_u, grad_v, grad_p, None, None, None, None


# def pde_residual_triton(U, V, P, dx, dy, dt, nu=0.01):
#     return _NS2DTriton.apply(U, V, P, dx, dy, dt, nu)


# # ── 绝对严谨的数值校验 ──────────────────────────────────────────────────────
# if __name__ == "__main__":
#     Nt, Nx, Ny = 10, 32, 32
#     nu = 0.01
#     device = "cuda"
    
#     # 用双精度做梯度 Check 以避免舍入误差
#     x = torch.linspace(0, 2 * np.pi, Nx, device=device, dtype=torch.float64)
#     y = torch.linspace(0, 2 * np.pi, Ny, device=device, dtype=torch.float64)
#     t = torch.linspace(0, 1, Nt, device=device, dtype=torch.float64)
#     dx, dy, dt = float(x[1] - x[0]), float(y[1] - y[0]), float(t[1] - t[0])
#     T, X, Y = torch.meshgrid(t, x, y, indexing='ij')

#     U_pt = (torch.sin(X) * torch.cos(Y)).requires_grad_(True)
#     V_pt = (-torch.cos(X) * torch.sin(Y)).requires_grad_(True)
#     P_pt = (-0.25 * (torch.cos(2*X) + torch.cos(2*Y))).requires_grad_(True)

#     U_tr = U_pt.clone().detach().requires_grad_(True)
#     V_tr = V_pt.clone().detach().requires_grad_(True)
#     P_tr = P_pt.clone().detach().requires_grad_(True)

#     # --- 1. PyTorch 原始差分计算 ---
#     def ns_pytorch(U, V, P, dx, dy, dt, nu):
#         u_t = (U[2:,1:-1,1:-1] - U[:-2,1:-1,1:-1]) / (2*dt)
#         u_x = (U[1:-1,2:,1:-1] - U[1:-1,:-2,1:-1]) / (2*dx)
#         u_y = (U[1:-1,1:-1,2:] - U[1:-1,1:-1,:-2]) / (2*dy)
#         u_xx = (U[1:-1,2:,1:-1] - 2*U[1:-1,1:-1,1:-1] + U[1:-1,:-2,1:-1]) / dx**2
#         u_yy = (U[1:-1,1:-1,2:] - 2*U[1:-1,1:-1,1:-1] + U[1:-1,1:-1,:-2]) / dy**2
        
#         v_t = (V[2:,1:-1,1:-1] - V[:-2,1:-1,1:-1]) / (2*dt)
#         v_x = (V[1:-1,2:,1:-1] - V[1:-1,:-2,1:-1]) / (2*dx)
#         v_y = (V[1:-1,1:-1,2:] - V[1:-1,1:-1,:-2]) / (2*dy)
#         v_xx = (V[1:-1,2:,1:-1] - 2*V[1:-1,1:-1,1:-1] + V[1:-1,:-2,1:-1]) / dx**2
#         v_yy = (V[1:-1,1:-1,2:] - 2*V[1:-1,1:-1,1:-1] + V[1:-1,1:-1,:-2]) / dy**2

#         p_x = (P[1:-1,2:,1:-1] - P[1:-1,:-2,1:-1]) / (2*dx)
#         p_y = (P[1:-1,1:-1,2:] - P[1:-1,1:-1,:-2]) / (2*dy)

#         res_u = u_t + U[1:-1,1:-1,1:-1]*u_x + V[1:-1,1:-1,1:-1]*u_y + p_x - nu*(u_xx+u_yy)
#         res_v = v_t + U[1:-1,1:-1,1:-1]*v_x + V[1:-1,1:-1,1:-1]*v_y + p_y - nu*(v_xx+v_yy)
#         res_div = u_x + v_y

#         return (res_u**2 + res_v**2 + res_div**2).mean()

#     # --- 2. 启动测试 ---
#     loss_pt = ns_pytorch(U_pt, V_pt, P_pt, dx, dy, dt, nu)
#     loss_tr = pde_residual_triton(U_tr, V_tr, P_tr, dx, dy, dt, nu)
    
#     print(f"Forward L1 Error:  {abs(loss_pt.item() - loss_tr.item()):.2e} (Expected < 1e-10)")

#     loss_pt.backward()
#     loss_tr.backward()

#     print(f"Backward U Error:  {(U_pt.grad - U_tr.grad).abs().max().item():.2e}")
#     print(f"Backward V Error:  {(V_pt.grad - V_tr.grad).abs().max().item():.2e}")
#     print(f"Backward P Error:  {(P_pt.grad - P_tr.grad).abs().max().item():.2e}")
