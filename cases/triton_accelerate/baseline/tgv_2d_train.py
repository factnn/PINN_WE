"""
2D Taylor-Green Vortex: three implementations comparison.
PDE: u_t + u*u_x + v*u_y = -p_x + nu*(u_xx+u_yy)
     v_t + u*v_x + v*v_y = -p_y + nu*(v_xx+v_yy)
Exact: u=sin(x)cos(y)exp(-2*nu*t), v=-cos(x)sin(y)exp(-2*nu*t)
Domain: [0,2pi]^2, t in [0,1], nu=0.01
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kernels.stencil_2d import ns2d_residual_triton, ns2d_fwd_kernel, ns2d_bwd_kernel

nu = 0.01
Nx, Ny, Nt = 64, 64, 20
device = "cuda"
OUT = Path(__file__).parent.parent / "output" / "tgv_2d"
OUT.mkdir(parents=True, exist_ok=True)


class MLP(nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        layers = [nn.Linear(3, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.out_u = nn.Linear(width, 1)
        self.out_v = nn.Linear(width, 1)

    def forward(self, xyt):
        h = self.net(xyt)
        return self.out_u(h).squeeze(-1), self.out_v(h).squeeze(-1)


def make_grid():
    x = torch.linspace(0, 2*np.pi, Nx, device=device)
    y = torch.linspace(0, 2*np.pi, Ny, device=device)
    t = torch.linspace(0, 1, Nt, device=device)
    T, X, Y = torch.meshgrid(t, x, y, indexing='ij')
    dx = float(x[1]-x[0]); dy = float(y[1]-y[0]); dt = float(t[1]-t[0])
    return X, Y, T, dx, dy, dt


def exact_uv(X, Y, T):
    U = torch.sin(X) * torch.cos(Y) * torch.exp(-2*nu*T)
    V = -torch.cos(X) * torch.sin(Y) * torch.exp(-2*nu*T)
    return U, V


def pde_residual_pytorch(U, V, dx, dy, dt):
    u_t  = (U[2:,1:-1,1:-1] - U[:-2,1:-1,1:-1]) / (2*dt)
    u_x  = (U[1:-1,2:,1:-1] - U[1:-1,:-2,1:-1]) / (2*dx)
    u_y  = (U[1:-1,1:-1,2:] - U[1:-1,1:-1,:-2]) / (2*dy)
    u_xx = (U[1:-1,2:,1:-1] - 2*U[1:-1,1:-1,1:-1] + U[1:-1,:-2,1:-1]) / dx**2
    u_yy = (U[1:-1,1:-1,2:] - 2*U[1:-1,1:-1,1:-1] + U[1:-1,1:-1,:-2]) / dy**2
    v_t  = (V[2:,1:-1,1:-1] - V[:-2,1:-1,1:-1]) / (2*dt)
    v_x  = (V[1:-1,2:,1:-1] - V[1:-1,:-2,1:-1]) / (2*dx)
    v_y  = (V[1:-1,1:-1,2:] - V[1:-1,1:-1,:-2]) / (2*dy)
    v_xx = (V[1:-1,2:,1:-1] - 2*V[1:-1,1:-1,1:-1] + V[1:-1,:-2,1:-1]) / dx**2
    v_yy = (V[1:-1,1:-1,2:] - 2*V[1:-1,1:-1,1:-1] + V[1:-1,1:-1,:-2]) / dy**2
    u_c = U[1:-1,1:-1,1:-1]; v_c = V[1:-1,1:-1,1:-1]
    res_u = u_t + u_c*u_x + v_c*u_y - nu*(u_xx+u_yy)
    res_v = v_t + u_c*v_x + v_c*v_y - nu*(v_xx+v_yy)
    return (res_u**2 + res_v**2).mean()


def ic_bc_loss(U, V, U_exact, V_exact):
    # IC at t=0
    ic = ((U[0]-U_exact[0])**2 + (V[0]-V_exact[0])**2).mean()
    # Periodic BC: u at x=0 == u at x=Nx-1
    bc_u = ((U[:,0,:]-U[:,-1,:])**2 + (U[:,:,0]-U[:,:,-1])**2).mean()
    bc_v = ((V[:,0,:]-V[:,-1,:])**2 + (V[:,:,0]-V[:,:,-1])**2).mean()
    return ic + bc_u + bc_v


class _NSTriton(torch.autograd.Function):
    """Triton forward+backward for TGV (no pressure). Uses P=zeros internally."""
    @staticmethod
    def forward(ctx, U, V, dx, dy, dt):
        U, V = U.contiguous(), V.contiguous()
        Nt, Nx, Ny = U.shape
        # P=zeros for the no-pressure TGV formulation
        P = torch.zeros_like(U)
        res_u = torch.empty((Nt-2, Nx-2, Ny-2), device=U.device, dtype=U.dtype)
        res_v = torch.empty_like(res_u)
        res_div = torch.empty_like(res_u)
        BLOCK_X, BLOCK_Y = 16, 16
        grid = (Nt-2, (Nx-2+BLOCK_X-1)//BLOCK_X, (Ny-2+BLOCK_Y-1)//BLOCK_Y)
        ns2d_fwd_kernel[grid](U, V, P, res_u, res_v, res_div,
                              Nt, Nx, Ny, dx, dy, dt, nu, BLOCK_X, BLOCK_Y)
        ctx.save_for_backward(U, V)
        ctx.res_u, ctx.res_v = res_u, res_v
        ctx.dx, ctx.dy, ctx.dt = dx, dy, dt
        return (res_u**2 + res_v**2).mean()

    @staticmethod
    def backward(ctx, grad_out):
        U, V = ctx.saved_tensors
        res_u, res_v = ctx.res_u, ctx.res_v
        dx, dy, dt = ctx.dx, ctx.dy, ctx.dt
        Nt, Nx, Ny = U.shape
        N_total = res_u.numel()  # (res_u + res_v) each have same numel
        scale = (2.0 / N_total) * grad_out
        Gu = res_u * scale
        Gv = res_v * scale
        # P gradient is irrelevant for no-P formulation
        Gdiv = torch.zeros_like(res_u)
        grad_u = torch.zeros_like(U)
        grad_v = torch.zeros_like(V)
        grad_p = torch.zeros_like(U)
        BLOCK_X, BLOCK_Y = 16, 16
        grid = (Nt-2, (Nx-2+BLOCK_X-1)//BLOCK_X, (Ny-2+BLOCK_Y-1)//BLOCK_Y)
        ns2d_bwd_kernel[grid](
            U, V, Gu, Gv, Gdiv, grad_u, grad_v, grad_p,
            Nt, Nx, Ny, dx, dy, dt, nu, BLOCK_X, BLOCK_Y
        )
        return grad_u, grad_v, None, None, None


def train(backend, epochs=5000, lr=1e-3, loss_threshold=1e-3):
    X, Y, T, dx, dy, dt = make_grid()
    U_exact, V_exact = exact_uv(X, Y, T)
    xyt = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)

    model = MLP().to(device)
    if backend == "compile":
        model = torch.compile(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.5)

    history = []; t2s = None; t0 = time.time()

    for ep in range(1, epochs+1):
        opt.zero_grad()
        u_flat, v_flat = model(xyt)
        U = u_flat.reshape(Nt, Nx, Ny)
        V = v_flat.reshape(Nt, Nx, Ny)

        if backend == "vanilla":
            # autograd
            xyt_g = xyt.detach().requires_grad_(True)
            u_g, v_g = model(xyt_g)
            u_g = u_g.reshape(Nt, Nx, Ny); v_g = v_g.reshape(Nt, Nx, Ny)
            grads_u = torch.autograd.grad(u_g.sum(), xyt_g, create_graph=True)[0]
            grads_v = torch.autograd.grad(v_g.sum(), xyt_g, create_graph=True)[0]
            u_t = grads_u[:,2].reshape(Nt,Nx,Ny)
            u_x = grads_u[:,0].reshape(Nt,Nx,Ny)
            u_y = grads_u[:,1].reshape(Nt,Nx,Ny)
            v_t = grads_v[:,2].reshape(Nt,Nx,Ny)
            v_x = grads_v[:,0].reshape(Nt,Nx,Ny)
            v_y = grads_v[:,1].reshape(Nt,Nx,Ny)
            u_xx = torch.autograd.grad(u_x.sum(), xyt_g, create_graph=True)[0][:,0].reshape(Nt,Nx,Ny)
            u_yy = torch.autograd.grad(u_y.sum(), xyt_g, create_graph=True)[0][:,1].reshape(Nt,Nx,Ny)
            v_xx = torch.autograd.grad(v_x.sum(), xyt_g, create_graph=True)[0][:,0].reshape(Nt,Nx,Ny)
            v_yy = torch.autograd.grad(v_y.sum(), xyt_g, create_graph=True)[0][:,1].reshape(Nt,Nx,Ny)
            res_u = u_t + u_g*u_x + v_g*u_y - nu*(u_xx+u_yy)
            res_v = v_t + u_g*v_x + v_g*v_y - nu*(v_xx+v_yy)
            loss_pde = (res_u**2 + res_v**2).mean()
        elif backend in ("canpinn", "compile"):
            loss_pde = pde_residual_pytorch(U, V, dx, dy, dt)
        else:
            loss_pde = _NSTriton.apply(U, V, dx, dy, dt)

        loss = loss_pde + 10 * ic_bc_loss(U, V, U_exact, V_exact)
        loss.backward(); opt.step(); sch.step()

        wall = time.time()-t0; lv = loss.item()
        history.append((wall, ep, lv))
        if t2s is None and lv < loss_threshold:
            t2s = wall
            print(f"  [{backend}] T2S={wall:.1f}s at ep {ep}")
            break
        if ep % 1000 == 0:
            print(f"[{backend}] ep {ep:5d} loss={lv:.3e} t={wall:.1f}s")

    elapsed = time.time()-t0
    mem = torch.cuda.max_memory_allocated()/1e9
    print(f"[{backend}] Done: {elapsed:.1f}s  mem={mem:.3f}GB")
    return model, history, elapsed, mem, t2s, U_exact, V_exact, X, Y, T


def plot_results(model, X, Y, T, U_exact, V_exact, name):
    xyt = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)
    with torch.no_grad():
        u_flat, v_flat = model(xyt)
    U = u_flat.reshape(Nt,Nx,Ny).cpu().numpy()
    V = v_flat.reshape(Nt,Nx,Ny).cpu().numpy()
    Ue = U_exact.cpu().numpy(); Ve = V_exact.cpu().numpy()

    l2_u = np.linalg.norm(U-Ue)/np.linalg.norm(Ue)
    l2_v = np.linalg.norm(V-Ve)/np.linalg.norm(Ve)
    print(f"[{name}] L2_u={l2_u:.4e}  L2_v={l2_v:.4e}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    x_np = X[0,:,0].cpu().numpy(); y_np = Y[0,0,:].cpu().numpy()
    ti = Nt//2
    for row, (field, field_e, label) in enumerate([(U, Ue, 'u'), (V, Ve, 'v')]):
        im = axes[row,0].contourf(x_np, y_np, field[ti].T, levels=20, cmap='RdBu_r')
        plt.colorbar(im, ax=axes[row,0]); axes[row,0].set_title(f'PINN {label}(t=0.5)')
        im2 = axes[row,1].contourf(x_np, y_np, field_e[ti].T, levels=20, cmap='RdBu_r')
        plt.colorbar(im2, ax=axes[row,1]); axes[row,1].set_title(f'Exact {label}(t=0.5)')
        err = np.abs(field[ti]-field_e[ti])
        im3 = axes[row,2].contourf(x_np, y_np, err.T, levels=20, cmap='hot_r')
        plt.colorbar(im3, ax=axes[row,2]); axes[row,2].set_title(f'|error| L2={l2_u if row==0 else l2_v:.2e}')
    plt.suptitle(f'TGV 2D — {name}'); plt.tight_layout()
    plt.savefig(OUT/f"solution_{name}.png", dpi=150); plt.close()
    return (l2_u+l2_v)/2


if __name__ == "__main__":
    import triton
    backends = ["vanilla", "canpinn", "compile", "triton"]
    results = {}

    # kernel benchmark
    X, Y, T, dx, dy, dt = make_grid()
    U_e, V_e = exact_uv(X, Y, T)
    print("=== Kernel benchmark ===")
    ms_pt = triton.testing.do_bench(lambda: pde_residual_pytorch(U_e, V_e, dx, dy, dt))
    ms_tr = triton.testing.do_bench(lambda: ns2d_residual_triton(U_e.contiguous(), V_e.contiguous(), dx, dy, dt, nu))
    print(f"PyTorch FD: {ms_pt:.3f}ms  Triton: {ms_tr:.3f}ms  Speedup: {ms_pt/ms_tr:.2f}x")
    for b in [("vanilla", None), ("canpinn", None), ("triton", None)]:
        torch.cuda.reset_peak_memory_stats()

    # end-to-end training
    histories = {}
    for b in backends:
        print(f"\n{'='*50}\nTraining: {b}")
        ckpt = OUT/f"model_{b}.pt"; meta = OUT/f"meta_{b}.npy"
        torch.cuda.reset_peak_memory_stats()
        if ckpt.exists() and meta.exists():
            print(f"  [cached]")
            model = MLP().to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            m = np.load(meta, allow_pickle=True).item()
            hist, elapsed, mem, t2s = m['history'], m['elapsed'], m['mem'], m['t2s']
        else:
            model, hist, elapsed, mem, t2s, U_e, V_e, X, Y, T = train(b, epochs=50000)
            torch.save(getattr(model, '_orig_mod', model).state_dict(), ckpt)
            np.save(meta, dict(history=hist, elapsed=elapsed, mem=mem, t2s=t2s))
        l2 = plot_results(model, X, Y, T, U_e, V_e, b)
        results[b] = dict(elapsed=elapsed, mem=mem, t2s=t2s, l2=l2)
        histories[b] = hist

    # loss curves
    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    for b, hist in histories.items():
        axes[0].semilogy([h[1] for h in hist], [h[2] for h in hist], label=b)
        axes[1].semilogy([h[0] for h in hist], [h[2] for h in hist], label=b)
    for ax, xl in zip(axes, ['Epoch','Wall time (s)']):
        ax.set_xlabel(xl); ax.set_ylabel('Loss'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT/"loss_curves.png", dpi=150); plt.close()

    print("\n=== Summary ===")
    print(f"{'backend':<12} {'time(s)':>8} {'mem(GB)':>8} {'T2S(s)':>8} {'L2':>10}")
    for b in backends:
        r = results[b]
        t2s = f"{r['t2s']:.1f}" if r['t2s'] else "N/A"
        print(f"{b:<12} {r['elapsed']:>8.1f} {r['mem']:>8.3f} {t2s:>8} {r['l2']:>10.4e}")
