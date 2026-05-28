"""
1D Heat Equation Triton Kernel: u_t = nu * u_xx (pure diffusion, no convection).
Data: U: [Nt, Nx], contiguous row-major.
Stencil: 5-point (center + t±1 + x±1).
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 64}),
        triton.Config({'BLOCK': 128}),
        triton.Config({'BLOCK': 256}),
        triton.Config({'BLOCK': 512}),
    ],
    key=['N_total'],
)
@triton.jit
def heat1d_fwd_kernel(
    U_ptr, res_ptr,
    Nt: tl.constexpr, Nx: tl.constexpr,
    dx: tl.constexpr, dt: tl.constexpr, nu: tl.constexpr,
    N_total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N_total

    Ni_nx = Nx - 2
    it = idx // Ni_nx + 1
    ix = idx % Ni_nx + 1

    stride = Nx
    base = it * stride + ix

    u_c  = tl.load(U_ptr + base, mask=mask)
    u_tp = tl.load(U_ptr + base + stride, mask=mask)
    u_tm = tl.load(U_ptr + base - stride, mask=mask)
    u_xp = tl.load(U_ptr + base + 1, mask=mask)
    u_xm = tl.load(U_ptr + base - 1, mask=mask)

    u_t = (u_tp - u_tm) / (2.0 * dt)
    u_xx = (u_xp - 2.0 * u_c + u_xm) / (dx * dx)

    res = u_t - nu * u_xx

    tl.store(res_ptr + pid * BLOCK + tl.arange(0, BLOCK), res, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 64}),
        triton.Config({'BLOCK': 128}),
        triton.Config({'BLOCK': 256}),
        triton.Config({'BLOCK': 512}),
    ],
    key=['N_total'],
)
@triton.jit
def heat1d_bwd_kernel(
    G_ptr, grad_U_ptr,
    Nt: tl.constexpr, Nx: tl.constexpr,
    dx: tl.constexpr, dt: tl.constexpr, nu: tl.constexpr,
    N_total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N_total

    Ni_nx = Nx - 2
    it = idx // Ni_nx + 1
    ix = idx % Ni_nx + 1

    gt = it - 1; gx = ix - 1
    bg = gt * Ni_nx + gx

    m_tp = (gt + 1 < Nt - 2) & mask
    m_tm = (gt - 1 >= 0) & mask
    m_xp = (gx + 1 < Nx - 2) & mask
    m_xm = (gx - 1 >= 0) & mask

    g_c  = tl.load(G_ptr + bg, mask=mask, other=0.0)
    g_tp = tl.load(G_ptr + bg + Ni_nx, mask=m_tp, other=0.0)
    g_tm = tl.load(G_ptr + bg - Ni_nx, mask=m_tm, other=0.0)
    g_xp = tl.load(G_ptr + bg + 1, mask=m_xp, other=0.0)
    g_xm = tl.load(G_ptr + bg - 1, mask=m_xm, other=0.0)

    idt = 1.0 / (2.0 * dt)
    idx2 = 1.0 / (dx * dx)

    # Adjoint of linear heat equation (no nonlinear terms)
    grad_u = g_c * (2.0 * nu * idx2)    # center: -2*nu/dx² → adjoint same
    grad_u += (g_tm - g_tp) * idt        # time neighbors
    grad_u -= nu * (g_xp + g_xm) * idx2  # x neighbors

    stride = Nx
    base = it * stride + ix
    tl.store(grad_U_ptr + base, grad_u, mask=mask)


class _Heat1DTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, dx, dt, nu_val):
        U = U.contiguous()
        Nt, Nx = U.shape
        N_total = (Nt - 2) * (Nx - 2)

        res = torch.empty(N_total, device=U.device, dtype=U.dtype)
        grid = lambda meta: ((N_total + meta['BLOCK'] - 1) // meta['BLOCK'],)
        heat1d_fwd_kernel[grid](U, res, Nt, Nx, dx, dt, nu_val, N_total)

        ctx.res = res
        ctx.dx, ctx.dt, ctx.nu_val = dx, dt, nu_val
        ctx.shape = (Nt, Nx)
        return (res**2).mean()

    @staticmethod
    def backward(ctx, grad_out):
        Nt, Nx = ctx.shape
        res = ctx.res
        dx, dt, nu_val = ctx.dx, ctx.dt, ctx.nu_val
        N_total = res.numel()

        scale = (2.0 / N_total) * grad_out
        G = res * scale

        grad_u = torch.zeros(Nt, Nx, device=res.device, dtype=res.dtype)
        grid = lambda meta: ((N_total + meta['BLOCK'] - 1) // meta['BLOCK'],)
        heat1d_bwd_kernel[grid](G, grad_u, Nt, Nx, dx, dt, nu_val, N_total)

        # Boundary contributions
        Ni_nx = Nx - 2
        G_2d = G.reshape(Nt - 2, Ni_nx)
        idt = 1.0 / (2.0 * dt)
        idx2 = 1.0 / (dx * dx)
        # Time boundaries
        grad_u[0, 1:-1] += -G_2d[0] * idt
        grad_u[Nt-1, 1:-1] += G_2d[-1] * idt
        # X boundaries
        grad_u[1:-1, 0] += -G_2d[:, 0] * nu_val * idx2
        grad_u[1:-1, Nx-1] += -G_2d[:, -1] * nu_val * idx2

        return grad_u, None, None, None


def heat1d_residual_triton(U, dx, dt, nu_val):
    return _Heat1DTriton.apply(U.contiguous(), dx, dt, nu_val)


if __name__ == "__main__":
    import numpy as np
    device = "cuda"
    dtype = torch.float64
    Nt, Nx = 50, 128
    nu_val = 0.5
    dx = 2.0 / (Nx - 1)
    dt = 1.0 / (Nt - 1)

    x = torch.linspace(-1, 1, Nx, device=device, dtype=dtype)
    t = torch.linspace(0, 1, Nt, device=device, dtype=dtype)
    T, X = torch.meshgrid(t, x, indexing='ij')

    # Exact: u = sin(πx) * exp(-ν*π²*t)
    U_exact = torch.sin(np.pi * X) * torch.exp(-nu_val * np.pi**2 * T)

    # PyTorch FD
    def pde_pt(U, dx, dt):
        u_t = (U[2:, 1:-1] - U[:-2, 1:-1]) / (2*dt)
        u_xx = (U[1:-1, 2:] - 2*U[1:-1, 1:-1] + U[1:-1, :-2]) / dx**2
        res = u_t - nu_val * u_xx
        return (res**2).mean()

    U1 = U_exact.clone().requires_grad_(True)
    U2 = U_exact.clone().requires_grad_(True)
    loss_pt = pde_pt(U1, dx, dt)
    loss_tr = heat1d_residual_triton(U2, dx, dt, nu_val)
    print(f"Forward rel_err: {abs(loss_pt.item()-loss_tr.item())/(abs(loss_pt.item())+1e-12):.2e}")

    loss_pt.backward(); loss_tr.backward()
    err = (U1.grad - U2.grad).abs().max().item()
    print(f"Grad max err: {err:.2e}")
    print(f"{'PASS' if err < 1e-5 else 'FAIL'}")
