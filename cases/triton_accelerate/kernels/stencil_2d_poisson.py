"""
2D Poisson Triton Kernel: Δu = f (Laplacian stencil).
Simplest possible PDE kernel. Laplacian is self-adjoint.
Data: U, F: [Nx, Ny], contiguous row-major.
Stencil: 5-point (center + 4 neighbors).
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
def poisson2d_fwd_kernel(
    U_ptr, F_ptr, res_ptr,
    Nx: tl.constexpr, Ny: tl.constexpr,
    dx: tl.constexpr, dy: tl.constexpr,
    N_total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Forward: res = u_xx + u_yy - f on interior points."""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N_total

    # Convert flat index to 2D (interior points only)
    ix = idx // (Ny - 2) + 1
    iy = idx % (Ny - 2) + 1

    sx = Ny  # stride in x direction
    base = ix * sx + iy

    # Load 5-point stencil
    u_c = tl.load(U_ptr + base, mask=mask)
    u_xp = tl.load(U_ptr + base + sx, mask=mask)
    u_xm = tl.load(U_ptr + base - sx, mask=mask)
    u_yp = tl.load(U_ptr + base + 1, mask=mask)
    u_ym = tl.load(U_ptr + base - 1, mask=mask)

    # Load source term
    f_c = tl.load(F_ptr + base, mask=mask)

    # Laplacian
    u_xx = (u_xp - 2.0 * u_c + u_xm) / (dx * dx)
    u_yy = (u_yp - 2.0 * u_c + u_ym) / (dy * dy)

    # Residual: Δu - f = 0
    res = u_xx + u_yy - f_c

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
def poisson2d_bwd_kernel(
    G_ptr, grad_U_ptr,
    Nx: tl.constexpr, Ny: tl.constexpr,
    dx: tl.constexpr, dy: tl.constexpr,
    N_total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Backward: adjoint of Laplacian (self-adjoint, same stencil)."""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N_total

    ix = idx // (Ny - 2) + 1
    iy = idx % (Ny - 2) + 1

    # Interior grid index in G (flat)
    gx = ix - 1
    gy = iy - 1
    Ni_ny = Ny - 2
    bg = gx * Ni_ny + gy

    # Neighbor masks
    m_xp = (gx + 1 < Nx - 2) & mask
    m_xm = (gx - 1 >= 0) & mask
    m_yp = (gy + 1 < Ny - 2) & mask
    m_ym = (gy - 1 >= 0) & mask

    # Load upstream gradient neighbors
    g_c = tl.load(G_ptr + bg, mask=mask, other=0.0)
    g_xp = tl.load(G_ptr + bg + Ni_ny, mask=m_xp, other=0.0)
    g_xm = tl.load(G_ptr + bg - Ni_ny, mask=m_xm, other=0.0)
    g_yp = tl.load(G_ptr + bg + 1, mask=m_yp, other=0.0)
    g_ym = tl.load(G_ptr + bg - 1, mask=m_ym, other=0.0)

    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)

    # Adjoint of Laplacian (self-adjoint: same stencil applied to G)
    grad_u = g_c * (-2.0 * idx2 - 2.0 * idy2)  # center contribution
    grad_u += (g_xp + g_xm) * idx2              # x neighbors
    grad_u += (g_yp + g_ym) * idy2              # y neighbors
    # Diffusion neighbor adjoint (from being a neighbor of adjacent residuals)
    grad_u -= (g_xp + g_xm) * idx2 + (g_yp + g_ym) * idy2
    # Wait - let me redo this properly.
    # Actually for Laplacian, adjoint is simpler:
    # d(res_j)/d(U_i) contributes to grad_U_i
    # Center: d(res_here)/d(U_here) = -2/dx² - 2/dy²  → g_c * (-2*idx2 - 2*idy2)
    # As x-neighbor: d(res_{i-1})/d(U_i) = 1/dx²      → g_xm * idx2
    # As x+neighbor: d(res_{i+1})/d(U_i) = 1/dx²      → g_xp * idx2
    # Similarly for y
    grad_u = g_c * (-2.0 * idx2 - 2.0 * idy2)
    grad_u += g_xp * idx2 + g_xm * idx2
    grad_u += g_yp * idy2 + g_ym * idy2

    sx = Ny
    base = ix * sx + iy
    tl.store(grad_U_ptr + base, grad_u, mask=mask)


# ── Boundary gradient fix ──────────────────────────────────────────────────
def _add_boundary_gradients_poisson(G, grad_u, dx, dy):
    """Add gradient contributions for boundary points."""
    Nx, Ny = grad_u.shape
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)
    i = slice(1, -1)

    # x=0 boundary
    grad_u[0, i] += G[0, :] * idx2
    # x=Nx-1 boundary
    grad_u[Nx-1, i] += G[Nx-3, :] * idx2
    # y=0 boundary
    grad_u[i, 0] += G[:, 0] * idy2
    # y=Ny-1 boundary
    grad_u[i, Ny-1] += G[:, Ny-3] * idy2


# ── Autograd Function ──────────────────────────────────────────────────────
class _Poisson2DTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, F, dx, dy):
        U = U.contiguous()
        Nx, Ny = U.shape
        N_total = (Nx - 2) * (Ny - 2)

        res = torch.empty(N_total, device=U.device, dtype=U.dtype)
        grid = lambda meta: ((N_total + meta['BLOCK'] - 1) // meta['BLOCK'],)
        poisson2d_fwd_kernel[grid](U, F, res, Nx, Ny, dx, dy, N_total)

        ctx.save_for_backward(torch.tensor([Nx, Ny], device=U.device))
        ctx.res = res
        ctx.dx, ctx.dy = dx, dy
        ctx.shape = (Nx, Ny)
        return (res**2).mean()

    @staticmethod
    def backward(ctx, grad_out):
        Nx, Ny = ctx.shape
        res = ctx.res
        dx, dy = ctx.dx, ctx.dy
        N_total = res.numel()

        scale = (2.0 / N_total) * grad_out
        G = res * scale

        grad_u = torch.zeros(Nx, Ny, device=res.device, dtype=res.dtype)

        grid = lambda meta: ((N_total + meta['BLOCK'] - 1) // meta['BLOCK'],)
        poisson2d_bwd_kernel[grid](G, grad_u, Nx, Ny, dx, dy, N_total)

        G_2d = G.reshape(Nx - 2, Ny - 2)
        _add_boundary_gradients_poisson(G_2d, grad_u, dx, dy)

        return grad_u, None, None, None


def poisson2d_residual_triton(U, F, dx, dy):
    """Compute mean(residual²) for 2D Poisson: Δu - f = 0."""
    return _Poisson2DTriton.apply(U.contiguous(), F.contiguous(), dx, dy)


# ── Verification ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    device = "cuda"
    dtype = torch.float64
    Nx, Ny = 64, 64
    dx = 1.0 / (Nx - 1)
    dy = 1.0 / (Ny - 1)

    x = torch.linspace(0, 1, Nx, device=device, dtype=dtype)
    y = torch.linspace(0, 1, Ny, device=device, dtype=dtype)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Manufactured solution: u = sin(pi*x)*sin(pi*y), f = -2*pi²*sin(pi*x)*sin(pi*y)
    U_exact = torch.sin(np.pi * X) * torch.sin(np.pi * Y)
    F = -2 * np.pi**2 * torch.sin(np.pi * X) * torch.sin(np.pi * Y)

    # PyTorch FD reference
    def pde_pytorch(U, F, dx, dy):
        u_xx = (U[2:, 1:-1] - 2*U[1:-1, 1:-1] + U[:-2, 1:-1]) / dx**2
        u_yy = (U[1:-1, 2:] - 2*U[1:-1, 1:-1] + U[1:-1, :-2]) / dy**2
        res = u_xx + u_yy - F[1:-1, 1:-1]
        return (res**2).mean()

    # Forward check
    U_t = U_exact.clone().requires_grad_(True)
    loss_pt = pde_pytorch(U_t, F, dx, dy)
    U_tr = U_exact.clone().requires_grad_(True)
    loss_tr = poisson2d_residual_triton(U_tr, F, dx, dy)
    print(f"Forward rel_err: {abs(loss_pt.item()-loss_tr.item())/(abs(loss_pt.item())+1e-12):.2e}")

    # Gradient check
    loss_pt.backward()
    loss_tr.backward()
    err = (U_t.grad - U_tr.grad).abs().max().item()
    print(f"Grad max err: {err:.2e}")
    print(f"{'PASS' if err < 1e-5 else 'FAIL'}")
