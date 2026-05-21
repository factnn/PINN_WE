"""
1D Viscous Burgers: three implementations of PDE residual loss.
PDE: u_t + u*u_x = nu*u_xx,  x in [-1,1], t in [0,1]
IC:  u(x,0) = -sin(pi*x)
BC:  u(-1,t) = u(1,t) = 0

Three backends:
  1. Vanilla PINN  — pure autograd
  2. CAN-PINN      — PyTorch native finite difference
  3. Ours (Triton) — fused stencil kernel

Deliverable: correctness check + micro-benchmark (no full training needed).
"""
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from kernels.stencil_1d_unsteady import burgers_2d_backward_triton

nu = 0.01 / np.pi
Nx, Nt = 1024, 100
device = "cuda"
dtype = torch.float32


# ── Shared MLP ────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, width=50, depth=4):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xt):
        return self.net(xt).squeeze(-1)


# ── Grid setup ────────────────────────────────────────────────────────────────
def make_grid():
    x = torch.linspace(-1, 1, Nx, device=device, dtype=dtype)
    t = torch.linspace(0, 1, Nt, device=device, dtype=dtype)
    T, X = torch.meshgrid(t, x, indexing='ij')   # [Nt, Nx]
    return X, T, float(x[1] - x[0]), float(t[1] - t[0])


# ── 1. Vanilla PINN (autograd) ────────────────────────────────────────────────
def loss_vanilla(model, X, T):
    xt = torch.stack([X.flatten(), T.flatten()], dim=1)
    xt.requires_grad_(True)
    u = model(xt).reshape(Nt, Nx)

    # autograd derivatives
    grads = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]
    u_x = grads[:, 0].reshape(Nt, Nx)
    u_t = grads[:, 1].reshape(Nt, Nx)
    u_xx = torch.autograd.grad(u_x.sum(), xt, create_graph=True)[0][:, 0].reshape(Nt, Nx)

    res = u_t + u * u_x - nu * u_xx
    return res.pow(2).mean()


# ── 2. CAN-PINN (PyTorch native FD) ──────────────────────────────────────────
def loss_canpinn(model, X, T, dx, dt):
    xt = torch.stack([X.flatten(), T.flatten()], dim=1)
    U = model(xt).reshape(Nt, Nx)

    # spatial: central diff on interior
    u_x  = (U[:, 2:] - U[:, :-2]) / (2 * dx)
    u_xx = (U[:, 2:] - 2 * U[:, 1:-1] + U[:, :-2]) / dx**2
    # temporal: central diff on interior (forward diff at t=0)
    u_t  = (U[2:, 1:-1] - U[:-2, 1:-1]) / (2 * dt)

    res = u_t + U[1:-1, 1:-1] * u_x[1:-1] - nu * u_xx[1:-1]
    return res.pow(2).mean()


# ── 3. Triton fused kernel ────────────────────────────────────────────────────
import triton
import triton.language as tl


@triton.jit
def burgers_residual_2d_kernel(
    U_ptr, res_ptr,
    Nt: tl.constexpr, Nx: tl.constexpr,
    dx: tl.constexpr, dt: tl.constexpr, nu: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    """Each program handles one row of t (interior), BLOCK_X interior x points."""
    pid_t = tl.program_id(0)   # t index in [1, Nt-2]
    pid_x = tl.program_id(1)   # x block

    it = pid_t + 1             # actual t index
    ix = pid_x * BLOCK_X + tl.arange(0, BLOCK_X) + 1  # actual x indices
    mask = ix < Nx - 1

    stride = Nx  # row stride

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


def loss_triton(U: torch.Tensor, dx: float, dt: float) -> torch.Tensor:
    """U: [Nt, Nx] — supports autograd via custom Function."""
    return _BurgersTriton.apply(U, dx, dt)


class _BurgersTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, dx, dt):
        U_c = U.contiguous()
        Nt_, Nx_ = U_c.shape
        res = torch.empty((Nt_ - 2, Nx_ - 2), device=U.device, dtype=U.dtype)
        BLOCK_X = 128
        grid = (Nt_ - 2, (Nx_ - 2 + BLOCK_X - 1) // BLOCK_X)
        burgers_residual_2d_kernel[grid](U_c, res, Nt_, Nx_, dx, dt, nu, BLOCK_X)
        ctx.save_for_backward(U_c, res)
        ctx.dx, ctx.dt = dx, dt
        return res.pow(2).mean()

    @staticmethod
    def backward(ctx, grad_out):
        U, res = ctx.saved_tensors
        dx, dt = ctx.dx, ctx.dt
        grad_U = burgers_2d_backward_triton(U, res, dx, dt, nu)
        return grad_out * grad_U, None, None
def _loss_triton_forward(U, dx, dt):
    """Pure PyTorch forward for backward pass."""
    u_x  = (U[:, 2:] - U[:, :-2]) / (2 * dx)
    u_xx = (U[:, 2:] - 2 * U[:, 1:-1] + U[:, :-2]) / dx**2
    u_t  = (U[2:, 1:-1] - U[:-2, 1:-1]) / (2 * dt)
    res  = u_t + U[1:-1, 1:-1] * u_x[1:-1] - nu * u_xx[1:-1]
    return res.pow(2).mean()


# ── Benchmark ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from profiler import profile_fn

    X, T, dx, dt = make_grid()
    model = MLP().to(device)

    # --- get U once (detached) for FD methods ---
    with torch.no_grad():
        xt = torch.stack([X.flatten(), T.flatten()], dim=1)
        U = model(xt).reshape(Nt, Nx)

    # --- correctness check: forward ---
    print("Computing reference (CAN-PINN)...")
    loss_ref = loss_canpinn(model, X, T, dx, dt).item()
    loss_tri = loss_triton(U, dx, dt).item()
    print(f"CAN-PINN loss : {loss_ref:.6e}")
    print(f"Triton   loss : {loss_tri:.6e}")
    print(f"Rel diff      : {abs(loss_ref - loss_tri) / (abs(loss_ref) + 1e-12):.2e}")

    # --- correctness check: gradient ---
    print("\n--- Gradient check (Triton backward vs PyTorch autograd) ---")
    U_grad = U.detach().clone().requires_grad_(True)
    loss_tri_grad = loss_triton(U_grad, dx, dt)
    loss_tri_grad.backward()
    grad_triton = U_grad.grad.clone()

    U_ref = U.detach().clone().requires_grad_(True)
    u_x_r = (U_ref[:, 2:] - U_ref[:, :-2]) / (2 * dx)
    u_xx_r = (U_ref[:, 2:] - 2 * U_ref[:, 1:-1] + U_ref[:, :-2]) / dx**2
    u_t_r = (U_ref[2:, 1:-1] - U_ref[:-2, 1:-1]) / (2 * dt)
    res_r = u_t_r + U_ref[1:-1, 1:-1] * u_x_r[1:-1] - nu * u_xx_r[1:-1]
    loss_ref_grad = res_r.pow(2).mean()
    loss_ref_grad.backward()
    grad_ref = U_ref.grad.clone()

    grad_err = (grad_ref[1:-1, 1:-1] - grad_triton[1:-1, 1:-1]).abs().max().item()
    print(f"Max grad err  : {grad_err:.2e}  {'PASS' if grad_err < 1e-5 else 'FAIL'}")

    # --- micro-benchmark ---
    print("\n--- Micro-benchmark (single residual call) ---")

    # vanilla autograd
    ms_vanilla = triton.testing.do_bench(lambda: loss_vanilla(model, X, T))

    # CAN-PINN
    ms_can = triton.testing.do_bench(lambda: loss_canpinn(model, X, T, dx, dt))

    # Triton
    ms_triton = triton.testing.do_bench(lambda: loss_triton(U, dx, dt))

    print(f"Vanilla PINN (autograd) : {ms_vanilla:.3f} ms")
    print(f"CAN-PINN (PyTorch FD)   : {ms_can:.3f} ms")
    print(f"Ours (Triton)           : {ms_triton:.3f} ms")
    print(f"Speedup vs Vanilla      : {ms_vanilla/ms_triton:.2f}x")
    print(f"Speedup vs CAN-PINN     : {ms_can/ms_triton:.2f}x")

    # --- peak memory ---
    for name, fn in [("Vanilla", lambda: loss_vanilla(model, X, T)),
                     ("CAN-PINN", lambda: loss_canpinn(model, X, T, dx, dt)),
                     ("Triton", lambda: loss_triton(U, dx, dt))]:
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"Peak mem [{name:10s}]: {mem:.1f} MB")

    # --- profiler traces ---
    profile_fn(lambda: loss_canpinn(model, X, T, dx, dt), name="burgers_1d_canpinn")
    profile_fn(lambda: loss_triton(U, dx, dt),            name="burgers_1d_triton")
