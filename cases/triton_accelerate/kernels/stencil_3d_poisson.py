"""
3D Poisson Triton Kernel: Δu = f (Laplacian stencil).
Data: U, F: [Nx, Ny, Nz], contiguous row-major.
Stencil: 7-point (center + 6 neighbors).
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
def poisson3d_fwd_kernel(
    U_ptr, F_ptr, res_ptr,
    Nx: tl.constexpr, Ny: tl.constexpr, Nz: tl.constexpr,
    dx: tl.constexpr, dy: tl.constexpr, dz: tl.constexpr,
    N_total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Forward: res = u_xx + u_yy + u_zz - f on interior points."""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N_total

    Ni_ny = Ny - 2
    Ni_nz = Nz - 2

    ix = idx // (Ni_ny * Ni_nz) + 1
    rem = idx % (Ni_ny * Ni_nz)
    iy = rem // Ni_nz + 1
    iz = rem % Ni_nz + 1

    sx = Ny * Nz  # x stride
    sy = Nz       # y stride

    base = ix * sx + iy * sy + iz

    # Load 7-point stencil
    u_c  = tl.load(U_ptr + base, mask=mask)
    u_xp = tl.load(U_ptr + base + sx, mask=mask)
    u_xm = tl.load(U_ptr + base - sx, mask=mask)
    u_yp = tl.load(U_ptr + base + sy, mask=mask)
    u_ym = tl.load(U_ptr + base - sy, mask=mask)
    u_zp = tl.load(U_ptr + base + 1, mask=mask)
    u_zm = tl.load(U_ptr + base - 1, mask=mask)

    f_c = tl.load(F_ptr + base, mask=mask)

    # Laplacian
    u_xx = (u_xp - 2.0 * u_c + u_xm) / (dx * dx)
    u_yy = (u_yp - 2.0 * u_c + u_ym) / (dy * dy)
    u_zz = (u_zp - 2.0 * u_c + u_zm) / (dz * dz)

    res = u_xx + u_yy + u_zz - f_c

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
def poisson3d_bwd_kernel(
    G_ptr, grad_U_ptr,
    Nx: tl.constexpr, Ny: tl.constexpr, Nz: tl.constexpr,
    dx: tl.constexpr, dy: tl.constexpr, dz: tl.constexpr,
    N_total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Backward: adjoint of 3D Laplacian (self-adjoint)."""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N_total

    Ni_ny = Ny - 2
    Ni_nz = Nz - 2

    ix = idx // (Ni_ny * Ni_nz) + 1
    iy = (idx % (Ni_ny * Ni_nz)) // Ni_nz + 1
    iz = (idx % (Ni_ny * Ni_nz)) % Ni_nz + 1

    # Interior grid flat index
    gx = ix - 1; gy = iy - 1; gz = iz - 1
    bg = gx * Ni_ny * Ni_nz + gy * Ni_nz + gz

    # Neighbor masks
    m_xp = (gx + 1 < Nx - 2) & mask
    m_xm = (gx - 1 >= 0) & mask
    m_yp = (gy + 1 < Ny - 2) & mask
    m_ym = (gy - 1 >= 0) & mask
    m_zp = (gz + 1 < Nz - 2) & mask
    m_zm = (gz - 1 >= 0) & mask

    g_c  = tl.load(G_ptr + bg, mask=mask, other=0.0)
    g_xp = tl.load(G_ptr + bg + Ni_ny * Ni_nz, mask=m_xp, other=0.0)
    g_xm = tl.load(G_ptr + bg - Ni_ny * Ni_nz, mask=m_xm, other=0.0)
    g_yp = tl.load(G_ptr + bg + Ni_nz, mask=m_yp, other=0.0)
    g_ym = tl.load(G_ptr + bg - Ni_nz, mask=m_ym, other=0.0)
    g_zp = tl.load(G_ptr + bg + 1, mask=m_zp, other=0.0)
    g_zm = tl.load(G_ptr + bg - 1, mask=m_zm, other=0.0)

    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)
    idz2 = 1.0 / (dz * dz)

    # Self-adjoint Laplacian: same stencil weights
    grad_u = g_c * (-2.0 * idx2 - 2.0 * idy2 - 2.0 * idz2)
    grad_u += (g_xp + g_xm) * idx2
    grad_u += (g_yp + g_ym) * idy2
    grad_u += (g_zp + g_zm) * idz2

    sx = Ny * Nz; sy = Nz
    base = ix * sx + iy * sy + iz
    tl.store(grad_U_ptr + base, grad_u, mask=mask)


# ── Boundary gradient fix ──────────────────────────────────────────────────
def _add_boundary_gradients_poisson3d(G, grad_u, dx, dy, dz):
    """Boundary contributions for 6 faces."""
    Nx, Ny, Nz = grad_u.shape
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)
    idz2 = 1.0 / (dz * dz)
    i = slice(1, -1)

    grad_u[0, i, i] += G[0, :, :] * idx2
    grad_u[Nx-1, i, i] += G[Nx-3, :, :] * idx2
    grad_u[i, 0, i] += G[:, 0, :] * idy2
    grad_u[i, Ny-1, i] += G[:, Ny-3, :] * idy2
    grad_u[i, i, 0] += G[:, :, 0] * idz2
    grad_u[i, i, Nz-1] += G[:, :, Nz-3] * idz2


# ── Autograd Function ──────────────────────────────────────────────────────
class _Poisson3DTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, F, dx, dy, dz):
        U = U.contiguous()
        Nx, Ny, Nz = U.shape
        N_total = (Nx - 2) * (Ny - 2) * (Nz - 2)

        res = torch.empty(N_total, device=U.device, dtype=U.dtype)
        grid = lambda meta: ((N_total + meta['BLOCK'] - 1) // meta['BLOCK'],)
        poisson3d_fwd_kernel[grid](U, F, res, Nx, Ny, Nz, dx, dy, dz, N_total)

        ctx.res = res
        ctx.dx, ctx.dy, ctx.dz = dx, dy, dz
        ctx.shape = (Nx, Ny, Nz)
        return (res**2).mean()

    @staticmethod
    def backward(ctx, grad_out):
        Nx, Ny, Nz = ctx.shape
        res = ctx.res
        dx, dy, dz = ctx.dx, ctx.dy, ctx.dz
        N_total = res.numel()

        scale = (2.0 / N_total) * grad_out
        G = res * scale

        grad_u = torch.zeros(Nx, Ny, Nz, device=res.device, dtype=res.dtype)

        grid = lambda meta: ((N_total + meta['BLOCK'] - 1) // meta['BLOCK'],)
        poisson3d_bwd_kernel[grid](G, grad_u, Nx, Ny, Nz, dx, dy, dz, N_total)

        G_3d = G.reshape(Nx - 2, Ny - 2, Nz - 2)
        _add_boundary_gradients_poisson3d(G_3d, grad_u, dx, dy, dz)

        return grad_u, None, None, None, None


def poisson3d_residual_triton(U, F, dx, dy, dz):
    """Compute mean(residual²) for 3D Poisson: Δu - f = 0."""
    return _Poisson3DTriton.apply(U.contiguous(), F.contiguous(), dx, dy, dz)


# ── Verification ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    device = "cuda"
    dtype = torch.float64
    Nx, Ny, Nz = 32, 32, 32
    dx = 1.0 / (Nx - 1)
    dy = 1.0 / (Ny - 1)
    dz = 1.0 / (Nz - 1)

    x = torch.linspace(0, 1, Nx, device=device, dtype=dtype)
    y = torch.linspace(0, 1, Ny, device=device, dtype=dtype)
    z = torch.linspace(0, 1, Nz, device=device, dtype=dtype)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # Manufactured: u = sin(πx)sin(πy)sin(πz), f = -3π²sin(πx)sin(πy)sin(πz)
    U_exact = torch.sin(np.pi * X) * torch.sin(np.pi * Y) * torch.sin(np.pi * Z)
    F = -3 * np.pi**2 * torch.sin(np.pi * X) * torch.sin(np.pi * Y) * torch.sin(np.pi * Z)

    # PyTorch FD reference
    def pde_pytorch(U, F, dx, dy, dz):
        i = slice(1, -1)
        u_xx = (U[2:, i, i] - 2*U[i, i, i] + U[:-2, i, i]) / dx**2
        u_yy = (U[i, 2:, i] - 2*U[i, i, i] + U[i, :-2, i]) / dy**2
        u_zz = (U[i, i, 2:] - 2*U[i, i, i] + U[i, i, :-2]) / dz**2
        res = u_xx + u_yy + u_zz - F[i, i, i]
        return (res**2).mean()

    # Forward check
    U_t = U_exact.clone().requires_grad_(True)
    loss_pt = pde_pytorch(U_t, F, dx, dy, dz)
    U_tr = U_exact.clone().requires_grad_(True)
    loss_tr = poisson3d_residual_triton(U_tr, F, dx, dy, dz)
    print(f"Forward rel_err: {abs(loss_pt.item()-loss_tr.item())/(abs(loss_pt.item())+1e-12):.2e}")

    # Gradient check
    loss_pt.backward()
    loss_tr.backward()
    err = (U_t.grad - U_tr.grad).abs().max().item()
    print(f"Grad max err: {err:.2e}")
    print(f"{'PASS' if err < 1e-5 else 'FAIL'}")
