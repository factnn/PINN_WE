"""
1D Viscous Burgers end-to-end training comparison.
Three backends: Vanilla PINN, CAN-PINN, Triton.
Trains to convergence, saves loss curves and u(x,t) plots.
"""
import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from baseline.burgers_1d_compare import MLP, loss_vanilla, loss_canpinn, loss_triton, make_grid

nu = 0.01 / np.pi
device = "cuda"


def exact_burgers(x_np, t_np, nu=0.01/np.pi):
    """
    Reference solution via method of lines (scipy ODE solver).
    Solves viscous Burgers on [-1,1] with IC=-sin(pi*x), BC=u(±1)=0.
    """
    from scipy.integrate import solve_ivp
    from scipy.sparse import diags

    Nx = len(x_np)
    dx = x_np[1] - x_np[0]

    # FD matrix for u_xx (interior only, Dirichlet BC=0)
    diag = -2 * np.ones(Nx - 2)
    off  =  1 * np.ones(Nx - 3)
    D2 = (diags([off, diag, off], [-1, 0, 1]).toarray()) / dx**2

    def rhs(t, u_int):
        u = np.concatenate([[0.0], u_int, [0.0]])
        u_x = (u[2:] - u[:-2]) / (2 * dx)
        u_xx = D2 @ u_int
        return -u_int * u_x + nu * u_xx

    u0 = -np.sin(np.pi * x_np[1:-1])  # interior IC
    t_eval_inner = t_np[t_np > 0]
    sol = solve_ivp(rhs, [0, t_np[-1]], u0, t_eval=t_eval_inner,
                    method='RK45', rtol=1e-8, atol=1e-10)

    U = np.zeros((len(t_np), Nx))
    U[0, :] = -np.sin(np.pi * x_np)   # IC at t=0
    U[t_np > 0, 1:-1] = sol.y.T
    return U
OUT = Path(__file__).parent.parent / "output" / "burgers_1d"
OUT.mkdir(parents=True, exist_ok=True)


def ic_loss(model, X, T):
    """u(x,0) = -sin(pi*x)"""
    x0 = X[0]  # [Nx]
    t0 = T[0]
    xt0 = torch.stack([x0, t0], dim=1)
    u0 = model(xt0)
    u0_exact = -torch.sin(np.pi * x0)
    return (u0 - u0_exact).pow(2).mean()


def bc_loss(model, X, T):
    """u(-1,t) = u(1,t) = 0"""
    tvals = T[:, 0]  # [Nt]
    xl = torch.full_like(tvals, -1.0)
    xr = torch.full_like(tvals, 1.0)
    xt_l = torch.stack([xl, tvals], dim=1)
    xt_r = torch.stack([xr, tvals], dim=1)
    return model(xt_l).pow(2).mean() + model(xt_r).pow(2).mean()


def ic_loss_from_U(U, X):
    """u(x,0) = -sin(pi*x), from precomputed U [Nt,Nx]"""
    u0_exact = -torch.sin(np.pi * X[0])
    return (U[0] - u0_exact).pow(2).mean()


def bc_loss_from_U(U):
    """u(-1,t) = u(1,t) = 0, from precomputed U [Nt,Nx]"""
    return U[:, 0].pow(2).mean() + U[:, -1].pow(2).mean()


def train_throughput(backend: str, warmup=50, measure=450, lr=1e-3):
    """Track 1: System throughput. Fixed steps, measure stable per-step time."""
    X, T, dx, dt = make_grid()
    model = MLP(width=50, depth=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt_all = torch.stack([X.flatten(), T.flatten()], dim=1)

    # warmup
    for _ in range(warmup):
        opt.zero_grad()
        U = model(xt_all).reshape(*X.shape)
        loss = _compute_loss(backend, model, U, X, T, dx, dt)
        loss.backward(); opt.step()
    torch.cuda.synchronize()

    # measure
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(measure):
        opt.zero_grad()
        U = model(xt_all).reshape(*X.shape)
        loss = _compute_loss(backend, model, U, X, T, dx, dt)
        loss.backward(); opt.step()
    torch.cuda.synchronize()

    elapsed = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 1e9
    avg_step_ms = elapsed / measure * 1000
    print(f"[throughput/{backend}] avg_step={avg_step_ms:.2f}ms  peak_mem={mem:.3f}GB")
    return avg_step_ms, mem


def train_convergence(backend: str, loss_threshold=1e-4, max_epochs=30000, lr=1e-3):
    """Track 2: Time-to-solution. Early stop when loss < threshold."""
    X, T, dx, dt = make_grid()
    model = MLP(width=50, depth=4).to(device)
    if backend == "compile":
        model = torch.compile(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.5)
    xt_all = torch.stack([X.flatten(), T.flatten()], dim=1)

    history = []
    time_to_solution = None
    t0 = time.time()

    for ep in range(1, max_epochs + 1):
        opt.zero_grad()
        U = model(xt_all).reshape(*X.shape)
        loss = _compute_loss(backend, model, U, X, T, dx, dt)
        loss.backward(); opt.step(); scheduler.step()

        wall = time.time() - t0
        loss_val = loss.item()
        history.append((wall, ep, loss_val))

        if time_to_solution is None and loss_val < loss_threshold:
            time_to_solution = wall
            print(f"  [{backend}] reached {loss_threshold:.0e} at ep {ep}, t={wall:.1f}s")
            break

        if ep % 2000 == 0:
            print(f"[convergence/{backend}] ep {ep:5d}  loss={loss_val:.3e}  t={wall:.1f}s")

    elapsed = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"[convergence/{backend}] Done: {elapsed:.1f}s  peak_mem={mem:.3f}GB")
    return model, history, elapsed, mem, time_to_solution


def _compute_loss(backend, model, U, X, T, dx, dt):
    if backend == "vanilla":
        loss_pde = loss_vanilla(model, X, T)
    elif backend in ("canpinn", "compile"):
        loss_pde = loss_canpinn(model, X, T, dx, dt)
    else:
        loss_pde = loss_triton(U, dx, dt)
    return loss_pde + 10 * ic_loss_from_U(U, X) + 10 * bc_loss_from_U(U)


def plot_solution(model, X, T, name, U_exact=None):
    with torch.no_grad():
        xt = torch.stack([X.flatten(), T.flatten()], dim=1)
        U = model(xt).reshape(*X.shape).cpu().numpy()
    x_np = X[0].cpu().numpy()
    t_np = T[:, 0].cpu().numpy()

    if U_exact is None:
        U_exact = exact_burgers(x_np, t_np)

    l2_err = np.linalg.norm(U - U_exact) / (np.linalg.norm(U_exact) + 1e-12)
    print(f"[{name}] L2 relative error vs exact: {l2_err:.4e}")

    slices = [(0, 't=0'), (25, 't=0.25'), (50, 't=0.5'), (75, 't=0.75'), (99, 't=1.0')]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # contourf: PINN prediction
    im = axes[0].contourf(x_np, t_np, U, levels=50, cmap='RdBu_r')
    plt.colorbar(im, ax=axes[0])
    axes[0].set_xlabel('x'); axes[0].set_ylabel('t')
    axes[0].set_title(f'PINN u(x,t) — {name}')

    # contourf: exact
    im2 = axes[1].contourf(x_np, t_np, U_exact, levels=50, cmap='RdBu_r')
    plt.colorbar(im2, ax=axes[1])
    axes[1].set_xlabel('x'); axes[1].set_ylabel('t')
    axes[1].set_title('Exact u(x,t)')

    # slices comparison
    colors = plt.cm.viridis(np.linspace(0, 1, len(slices)))
    for (ti, label), c in zip(slices, colors):
        axes[2].plot(x_np, U[ti],       color=c, lw=2,   label=f'PINN {label}')
        axes[2].plot(x_np, U_exact[ti], color=c, lw=1.5, ls='--', label=f'Exact {label}')
    axes[2].set_xlabel('x'); axes[2].set_ylabel('u')
    axes[2].set_title(f'Slices — {name} (L2={l2_err:.2e})')
    axes[2].legend(fontsize=6, ncol=2); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / f"solution_{name}.png", dpi=150)
    plt.close()
    return l2_err, U_exact


def plot_loss_curves(histories, names):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for hist, name in zip(histories, names):
        epochs = [h[1] for h in hist]
        walls  = [h[0] for h in hist]
        losses = [h[2] for h in hist]
        axes[0].semilogy(epochs, losses, label=name)
        axes[1].semilogy(walls,  losses, label=name)

    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss vs Epoch'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Wall time (s)'); axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss vs Wall Time'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "loss_curves.png", dpi=150)
    plt.close()
    print(f"Saved loss curves to {OUT}/loss_curves.png")


if __name__ == "__main__":
    X, T, dx, dt = make_grid()
    x_np = X[0].cpu().numpy()
    t_np = T[:, 0].cpu().numpy()

    print("Computing exact solution...")
    U_exact = exact_burgers(x_np, t_np)

    backends = ["vanilla", "canpinn", "compile", "triton"]
    throughput = {}
    convergence = {}

    # Track 1: throughput (warmup=50, measure=450)
    print("\n" + "="*50)
    print("Track 1: System Throughput (warmup=50, measure=450 steps)")
    for b in backends:
        torch.cuda.reset_peak_memory_stats()
        avg_ms, mem = train_throughput(b)
        throughput[b] = dict(avg_step_ms=avg_ms, peak_mem_gb=mem)

    # Track 2: convergence (early stop at 1e-4, max 30000)
    print("\n" + "="*50)
    print("Track 2: Time-to-Solution (threshold=1e-4, max=30000 epochs)")
    for b in backends:
        ckpt = OUT / f"model_{b}.pt"
        meta = OUT / f"meta_{b}.npy"
        torch.cuda.reset_peak_memory_stats()
        if ckpt.exists() and meta.exists():
            print(f"  [cached] {b}")
            model = MLP(width=50, depth=4).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            m = np.load(meta, allow_pickle=True).item()
            hist, elapsed, mem, t2s = m['history'], m['elapsed'], m['mem'], m['t2s']
        else:
            model, hist, elapsed, mem, t2s = train_convergence(b)
            torch.save(model.state_dict(), ckpt)
            np.save(meta, dict(history=hist, elapsed=elapsed, mem=mem, t2s=t2s))
        l2, _ = plot_solution(model, X, T, b, U_exact=U_exact)
        convergence[b] = dict(history=hist, elapsed=elapsed, mem=mem, t2s=t2s, l2=l2)

    # loss curves
    plot_loss_curves(
        [convergence[b]['history'] for b in backends],
        ["Vanilla PINN", "CAN-PINN", "Triton"]
    )

    # summary
    print("\n=== Track 1: Throughput ===")
    print(f"{'backend':<12} {'avg_step(ms)':>14} {'peak_mem(GB)':>13}")
    for b in backends:
        r = throughput[b]
        print(f"{b:<12} {r['avg_step_ms']:>14.2f} {r['peak_mem_gb']:>13.3f}")

    print("\n=== Track 2: Convergence ===")
    print(f"{'backend':<12} {'total(s)':>9} {'mem(GB)':>8} {'T2S(s)':>8} {'L2':>10}")
    for b in backends:
        r = convergence[b]
        t2s = f"{r['t2s']:.1f}" if r['t2s'] else "N/A"
        print(f"{b:<12} {r['elapsed']:>9.1f} {r['mem']:>8.3f} {t2s:>8} {r['l2']:>10.4e}")
