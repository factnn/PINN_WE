# autoresearch-TritonBackward-1D

自动实现 1D Burgers 方程的 Triton adjoint backward kernel。

## Setup (do once)

```bash
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
cd /share/project/zpy/PINN_WE/cases/triton_accelerate
git checkout -b autoresearch/triton-bwd-1d-$(date +%Y%m%d) 2>/dev/null || true
```

Read `kernels/stencil_1d.py` once at the start.

## Task

**Goal**: Add a Triton backward kernel to `kernels/stencil_1d.py` that correctly computes `∂Loss/∂U`.

**Current state**:
- Forward: `burgers_residual_triton(u, dx)` — Triton kernel, correct
- Backward: PyTorch fallback in `_BurgersTriton.backward()` in `baseline/burgers_1d_compare.py`
- Need: replace fallback with Triton kernel

**Success criterion**: `Max grad err < 1e-5` vs PyTorch autograd reference.

Test by running:
```bash
CUDA_VISIBLE_DEVICES=0 python kernels/stencil_1d.py
```

## Physics: 1D Burgers Adjoint

Residual: `res[i] = u[i] * (u[i+1]-u[i-1])/(2dx) - ν*(u[i+1]-2u[i]+u[i-1])/dx²`

Loss = `mean(res²)`, so upstream gradient: `G[i] = 2*res[i]/N`

**grad_U[i]** — contributions from all residual points that used U[i]:

```
# From res[i] directly (u_c * u_x, u_xx center):
grad_U[i] += G[i] * (u[i+1]-u[i-1])/(2dx)      # ∂(u_c*u_x)/∂u_c
grad_U[i] += -ν * (-2) * G[i] / dx²              # ∂(-ν*u_xx)/∂u_c = +2ν*G[i]/dx²

# From res[i-1] (u[i] appears as u_xp = u[(i-1)+1]):
grad_U[i] += G[i-1] * u[i-1] / (2dx)            # ∂(u_c[i-1]*u_x[i-1])/∂u[i]
grad_U[i] += -ν * G[i-1] / dx²                   # ∂(-ν*u_xx[i-1])/∂u_xp

# From res[i+1] (u[i] appears as u_xm = u[(i+1)-1]):
grad_U[i] -= G[i+1] * u[i+1] / (2dx)            # ∂(u_c[i+1]*u_x[i+1])/∂u[i]
grad_U[i] += -ν * G[i+1] / dx²                   # ∂(-ν*u_xx[i+1])/∂u_xm
```

Combined:
```
grad_U[i] = G[i] * (u[i+1]-u[i-1])/(2dx) + 2ν*G[i]/dx²
          + G[i-1] * u[i-1]/(2dx) - ν*G[i-1]/dx²
          - G[i+1] * u[i+1]/(2dx) - ν*G[i+1]/dx²
```

Boundary: grad_U[0] = grad_U[N-1] = 0 (Dirichlet BC).

## What to implement

Add to `kernels/stencil_1d.py`:
1. `burgers_bwd_kernel`: Triton kernel computing grad_U from G and U
2. `burgers_backward_triton(u, res, dx)`: Python wrapper
3. Update `_BurgersTriton.backward()` in `baseline/burgers_1d_compare.py` to use it

## Test code to add to stencil_1d.py __main__

```python
# Gradient check
import numpy as np
N = 1024; device = "cuda"
x = torch.linspace(-1, 1, N, device=device)
dx = float(x[1]-x[0])
u = torch.sin(np.pi * x).requires_grad_(True)
u2 = u.detach().clone().requires_grad_(True)

# PyTorch reference
from baseline.burgers_1d_compare import loss_canpinn_raw  # u*u_x - nu*u_xx
res_pt = (u[2:]-u[:-2])/(2*dx)*u[1:-1] - (0.01/np.pi)*(u[2:]-2*u[1:-1]+u[:-2])/dx**2
loss_pt = res_pt.pow(2).mean()
loss_pt.backward()

# Triton
loss_tr = burgers_residual_triton(u2, dx)  # must support backward
loss_tr.backward()

print(f"Grad err: {(u.grad[1:-1] - u2.grad[1:-1]).abs().max().item():.2e}")
```

## Experiment loop

LOOP FOREVER:
1. Read current `ns2d_bwd_kernel` / `burgers_bwd_kernel` implementation
2. Run test, check grad error
3. If error > 1e-5: identify wrong term from physics above, fix it, `git commit`
4. If error < 1e-5: update `_BurgersTriton.backward()` to use Triton kernel, verify end-to-end training still converges
5. **DONE** when grad err < 1e-5 AND training converges

**Context management**: `/compact` when near limit, re-read `kernels/stencil_1d.py` to recover.

**NEVER STOP** until done.
