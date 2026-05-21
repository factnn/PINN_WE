"""Shared utilities for 1D unsteady Burgers experiments.
u_t + u*u_x = nu*u_xx on [-1, 1] x [0, 1].
IC: u(x,0) = -sin(pi*x). BC: u(-1,t) = u(1,t) = 0.
Reference solution via RK45 (high-accuracy numerical).
"""
import sys, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import diags

nu = 0.01 / np.pi
Nx, Nt = 1024, 100
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(__file__).parent.parent.parent / "output" / "burgers_1d"
OUT.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. Network architectures
# ==========================================
class MLP(nn.Module):
    """Point-wise MLP: (x, t) -> u"""
    def __init__(self, width=128, depth=6):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.out_u = nn.Linear(width, 1)

    def forward(self, xt):
        return self.out_u(self.net(xt)).squeeze(-1)


class PhyCNN(nn.Module):
    """Grid-based 1D CNN: (X, T) -> U on [Nt, Nx]"""
    def __init__(self, channels=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(2, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv1d(channels, 1, kernel_size=5, padding=2),
        )

    def forward(self, X, T):
        # X, T: [Nt, Nx]
        inputs = torch.stack([X, T], dim=1)  # [Nt, 2, Nx]
        out = self.enc(inputs)  # [Nt, 1, Nx]
        return out[:, 0, :]  # [Nt, Nx]


# ==========================================
# 2. Grid & reference solution
# ==========================================
def make_grid():
    x = torch.linspace(-1, 1, Nx, device=device)
    t = torch.linspace(0, 1, Nt, device=device)
    T, X = torch.meshgrid(t, x, indexing='ij')
    dx = float(x[1]-x[0]); dt = float(t[1]-t[0])
    return X, T, dx, dt


def ic_loss_from_U(U, X):
    return (U[0] - (-torch.sin(np.pi * X[0]))).pow(2).mean()


def bc_loss_from_U(U):
    return U[:, 0].pow(2).mean() + U[:, -1].pow(2).mean()


def exact_burgers(x_np, t_np):
    dx = x_np[1] - x_np[0]
    diag_ = -2*np.ones(Nx-2); off = np.ones(Nx-3)
    D2 = diags([off, diag_, off], [-1,0,1]).toarray() / dx**2
    def rhs(t, u_int):
        u = np.concatenate([[0.], u_int, [0.]])
        u_x = (u[2:]-u[:-2])/(2*dx)
        return -u_int*u_x + nu*D2@u_int
    u0 = -np.sin(np.pi*x_np[1:-1])
    t_eval = t_np[t_np > 0]
    sol = solve_ivp(rhs, [0, t_np[-1]], u0, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)
    U = np.zeros((len(t_np), Nx))
    U[0] = -np.sin(np.pi*x_np)
    U[t_np > 0, 1:-1] = sol.y.T
    return U


# ==========================================
# 3. PDE residual (PyTorch FD)
# ==========================================
def pde_residual_pytorch(U, dx, dt):
    """u_t + u*u_x - nu*u_xx = 0 via central differences. U: [Nt, Nx]."""
    u_t  = (U[2:, 1:-1] - U[:-2, 1:-1]) / (2*dt)
    u_x  = (U[1:-1, 2:] - U[1:-1, :-2]) / (2*dx)
    u_xx = (U[1:-1, 2:] - 2*U[1:-1, 1:-1] + U[1:-1, :-2]) / dx**2
    uc = U[1:-1, 1:-1]
    res = u_t + uc * u_x - nu * u_xx
    return (res**2).mean()


def unified_loss_fn(model, xt, X, T, dx, dt):
    U = infer(model, xt, X, T)
    return pde_residual_pytorch(U, dx, dt) + 10*ic_loss_from_U(U, X) + 10*bc_loss_from_U(U)


# ==========================================
# 4. Inference helper
# ==========================================
def infer(model, xt, X, T):
    """Infer U from model (MLP or PhyCNN)."""
    inner = getattr(model, '_orig_mod', model)
    if isinstance(inner, PhyCNN):
        U = model(X, T)
    else:
        U = model(xt).reshape(Nt, Nx)
    return U


# ==========================================
# 5. Training & visualization
# ==========================================
def train_and_save(backend_name, model_fn, loss_fn=None,
                   max_epochs=200000, lr=1e-3, loss_threshold=1e-4, runs=1):
    if loss_fn is None:
        loss_fn = unified_loss_fn

    X, T, dx, dt = make_grid()
    xt = torch.stack([X.flatten(), T.flatten()], dim=1)
    x_np = X[0].cpu().numpy(); t_np = T[:,0].cpu().numpy()

    BYTES_PER_STEP = Nt * Nx * 4 * 6
    A100_BW_GBS = 1555.0

    all_elapsed = []; all_t2s = []

    for run_i in range(runs):
        model = model_fn().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)

        history = []; t2s = None; t2s_ep = None; step_times = []; t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        print(f"[{backend_name}] run {run_i+1}/{runs} | Grid: {Nx}x{Nt}")

        for ep in range(1, max_epochs+1):
            t_step = time.time()
            opt.zero_grad()
            loss = loss_fn(model, xt, X, T, dx, dt)
            loss.backward(); opt.step(); sch.step()
            torch.cuda.synchronize()
            step_times.append(time.time()-t_step)

            wall = time.time()-t0; lv = loss.item()
            history.append((wall, ep, lv))
            if t2s is None and lv < loss_threshold:
                t2s = wall; t2s_ep = ep
                print(f"  [{backend_name}] run {run_i+1}: T2S={wall:.1f}s at ep {ep}")
                break
            if ep % 10000 == 0:
                print(f"[{backend_name}] run {run_i+1} ep {ep:6d} loss={lv:.3e} t={wall:.1f}s")

        elapsed = time.time()-t0
        mem = torch.cuda.max_memory_allocated()/1e9
        avg_step_ms = np.mean(step_times)*1000
        bw_gbs = BYTES_PER_STEP/(avg_step_ms*1e-3)/1e9
        all_elapsed.append(elapsed); all_t2s.append(t2s)
        print(f"[{backend_name}] run {run_i+1}: {elapsed:.1f}s  mem={mem:.3f}GB  avg_step={avg_step_ms:.2f}ms  epochs={t2s_ep or ep}")

    ckpt = OUT/f"model_{backend_name}.pt"
    torch.save(getattr(model,'_orig_mod',model).state_dict(), ckpt)

    metrics = dict(
        history=history, elapsed=elapsed, mem_gb=mem,
        t2s=t2s, t2s_ep=t2s_ep, avg_step_ms=avg_step_ms,
        bw_gbs=bw_gbs, all_elapsed=all_elapsed, all_t2s=all_t2s,
    )
    np.save(OUT/f"meta_{backend_name}.npy", metrics)

    U_exact = exact_burgers(x_np, t_np)
    with torch.no_grad():
        U_pred = infer(model, xt, X, T).cpu().numpy()
    l2 = np.linalg.norm(U_pred-U_exact)/(np.linalg.norm(U_exact)+1e-12)
    _plot_solution(U_pred, U_exact, x_np, t_np, backend_name, l2)

    print(f"\n[{backend_name}] === Summary ===")
    print(f"  T2S          : {t2s:.1f}s" if t2s else "  T2S          : N/A")
    print(f"  Total_Epochs : {t2s_ep or ep}")
    print(f"  Avg_Step_ms  : {avg_step_ms:.2f}")
    print(f"  Peak_Mem_GB  : {mem:.3f}")
    print(f"  Mem_BW_GBs   : {bw_gbs:.1f}")
    print(f"  L2_err       : {l2:.4e}")
    return dict(elapsed=np.median(all_elapsed), mem_gb=mem, t2s=t2s,
                t2s_ep=t2s_ep, avg_step_ms=avg_step_ms,
                bw_gbs=bw_gbs, l2=l2, history=history)


def _plot_solution(U, U_exact, x_np, t_np, name, l2):
    slices = [(0,'t=0'),(25,'t=0.25'),(50,'t=0.5'),(75,'t=0.75'),(99,'t=1.0')]
    fig, axes = plt.subplots(1,3,figsize=(16,4))
    im = axes[0].contourf(x_np,t_np,U,levels=50,cmap='RdBu_r')
    plt.colorbar(im,ax=axes[0]); axes[0].set_title(f'PINN - {name}')
    im2 = axes[1].contourf(x_np,t_np,U_exact,levels=50,cmap='RdBu_r')
    plt.colorbar(im2,ax=axes[1]); axes[1].set_title('Exact')
    colors = plt.cm.viridis(np.linspace(0,1,len(slices)))
    for (ti,label),c in zip(slices,colors):
        axes[2].plot(x_np,U[ti],color=c,lw=2,label=f'PINN {label}')
        axes[2].plot(x_np,U_exact[ti],color=c,lw=1.5,ls='--')
    axes[2].set_title(f'Slices (L2={l2:.2e})'); axes[2].legend(fontsize=6,ncol=2); axes[2].grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT/f"solution_{name}.png",dpi=150); plt.close()


def base_argparser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--runs', type=int, default=1)
    p.add_argument('--max-epochs', type=int, default=200000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--threshold', type=float, default=1e-4)
    p.add_argument('--gpu', type=int, default=0)
    return p
