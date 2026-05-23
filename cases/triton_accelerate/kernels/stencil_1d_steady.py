"""
1D Steady Burgers Triton kernels: Forward + Adjoint Backward.
PDE: u*u_x = nu*u_xx. Grid [Nx], no time dimension.
"""
import torch
import triton
import triton.language as tl


# ── Forward kernel ─────────────────────────────────────────────────────────
@triton.jit
def burgers_steady_fwd_kernel(
    u_ptr, res_ptr,
    Nx: tl.constexpr,
    dx: tl.constexpr, nu: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK) + 1
    mask = i < Nx - 1

    u_c = tl.load(u_ptr + i,     mask=mask)
    u_l = tl.load(u_ptr + i - 1, mask=mask)
    u_r = tl.load(u_ptr + i + 1, mask=mask)

    u_x  = (u_r - u_l) / (2.0 * dx)
    u_xx = (u_r - 2.0 * u_c + u_l) / (dx * dx)
    res  = u_c * u_x - nu * u_xx

    tl.store(res_ptr + (i - 1), res, mask=mask)


# ── Backward (adjoint) kernel ──────────────────────────────────────────────
@triton.jit
def burgers_steady_bwd_kernel(
    u_ptr, G_ptr, grad_ptr,
    Nx: tl.constexpr,
    dx: tl.constexpr, nu: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK) + 1
    mask = i < Nx - 1

    u_c = tl.load(u_ptr + i,     mask=mask)
    u_l = tl.load(u_ptr + i - 1, mask=mask)
    u_r = tl.load(u_ptr + i + 1, mask=mask)

    mask_l = (i - 1 >= 1) & mask
    mask_r = (i + 1 < Nx - 1) & mask
    g_c = tl.load(G_ptr + (i - 1), mask=mask)
    g_l = tl.load(G_ptr + (i - 2), mask=mask_l, other=0.0)
    g_r = tl.load(G_ptr + i,     mask=mask_r, other=0.0)

    inv_2dx = 1.0 / (2.0 * dx)
    inv_dx2 = 1.0 / (dx * dx)

    # Center + convection/diffusion neighbors
    grad_val = (g_c * (u_r - u_l) * inv_2dx
                + 2.0 * nu * g_c * inv_dx2
                + g_l * u_l * inv_2dx - nu * g_l * inv_dx2
                - g_r * u_r * inv_2dx - nu * g_r * inv_dx2)

    tl.store(grad_ptr + i, grad_val, mask=mask)


# ── Boundary gradient fix ──────────────────────────────────────────────────
def burgers_steady_backward(u, res, dx, nu_val):
    Nx = u.shape[0]
    G = 2.0 * res / (Nx - 2)
    grad_u = torch.zeros(Nx, device=u.device, dtype=u.dtype)
    BLOCK = 256
    grid = ((Nx - 2 + BLOCK - 1) // BLOCK,)
    burgers_steady_bwd_kernel[grid](u, G, grad_u, Nx, dx, nu_val, BLOCK)

    # Boundary contributions
    inv_2dx = 1.0 / (2.0 * dx)
    inv_dx2 = 1.0 / (dx * dx)
    grad_u[0] += G[0] * (-u[1] * inv_2dx - nu_val * inv_dx2)
    grad_u[-1] += G[-1] * (u[-2] * inv_2dx - nu_val * inv_dx2)

    return grad_u


# ── Autograd Function ──────────────────────────────────────────────────────
class _BurgersSteadyTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, dx, nu_val):
        u = u.contiguous()
        Nx = u.shape[0]
        res = torch.empty(Nx - 2, device=u.device, dtype=u.dtype)
        BLOCK = 256
        grid = ((Nx - 2 + BLOCK - 1) // BLOCK,)
        burgers_steady_fwd_kernel[grid](u, res, Nx, dx, nu_val, BLOCK)
        ctx.save_for_backward(u, res)
        ctx.dx, ctx.nu_val = dx, nu_val
        return res.pow(2).mean()

    @staticmethod
    def backward(ctx, grad_out):
        u, res = ctx.saved_tensors
        dx, nu_val = ctx.dx, ctx.nu_val
        grad_u = burgers_steady_backward(u, res, dx, nu_val)
        return grad_out * grad_u, None, None


def burgers_steady_loss_triton(u, dx, nu_val):
    return _BurgersSteadyTriton.apply(u.contiguous(), dx, nu_val)


# ── Correctness check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, numpy as np, os
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
    os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/triton_burgers_steady_test')

    from cases.burgers_1d_steady.physics import pde_residual_pytorch, make_grid, MLP

    Nx = 256
    nu_val = 0.01 / np.pi
    device = "cuda"; dtype = torch.float64

    X, dx = make_grid()
    u = exact_u(X).to(dtype) if False else (torch.sin(2*np.pi*X)*X*(1-X)).to(dtype)

    # Forward
    print("1. Forward check...")
    u_pt = u.clone().requires_grad_(True)
    loss_pt = pde_residual_pytorch(u_pt.float(), dx)
    u_tr = u.clone().requires_grad_(True)
    loss_tr = burgers_steady_loss_triton(u_tr.float(), dx, nu_val)
    print(f"  Forward rel_err: {abs(loss_pt.item()-loss_tr.item())/(abs(loss_pt.item())+1e-12):.2e}")

    # Gradient
    print("2. Gradient check (float64)...")
    from cases.burgers_1d_steady.physics import pde_residual_pytorch
    u_pt2 = u.clone().requires_grad_(True)
    loss_pt2 = pde_residual_pytorch(u_pt2, dx)
    loss_pt2.backward()

    u_tr2 = u.clone().requires_grad_(True)
    loss_tr2 = burgers_steady_loss_triton(u_tr2, dx, nu_val)
    loss_tr2.backward()

    eU = (u_pt2.grad - u_tr2.grad).abs().max().item()
    print(f"  Grad U err (ALL): {eU:.1e}  {'PASS' if eU < 1e-5 else 'FAIL'}")

    # MLP field test
    print("3. MLP field test...")
    model = MLP().to(device)
    x_inp = X.unsqueeze(-1)
    with torch.no_grad():
        u_mlp = model(x_inp).detach().to(dtype)

    u_pt3 = u_mlp.clone().requires_grad_(True)
    loss_pt3 = pde_residual_pytorch(u_pt3, dx)
    loss_pt3.backward()

    u_tr3 = u_mlp.clone().requires_grad_(True)
    loss_tr3 = burgers_steady_loss_triton(u_tr3, dx, nu_val)
    loss_tr3.backward()

    eU3 = (u_pt3.grad - u_tr3.grad).abs().max().item()
    print(f"  Grad U err (ALL): {eU3:.1e}  {'PASS' if eU3 < 1e-5 else 'FAIL'}")

    import triton as _triton
    u_b = u_mlp.float()
    ms = _triton.testing.do_bench(lambda: burgers_steady_loss_triton(u_b, dx, nu_val))
    print(f"4. Triton forward: {ms:.3f}ms")

    def exact_u(x):
        return torch.sin(2 * np.pi * x) * x * (1 - x)
