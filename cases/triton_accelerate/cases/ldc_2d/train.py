"""
2D Lid-Driven Cavity (LDC): steady-state NS, Re=100.
BC: top lid u=1, v=0; other walls u=v=0.
Reference: Ghia et al. 1982 centerline velocities.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kernels.stencil_2d_ns_unsteady import ns2d_residual_triton

Re = 100.0
nu = 1.0 / Re
Nx, Ny = 64, 64
device = "cuda"
OUT = Path(__file__).parent.parent.parent / "output" / "ldc_2d"
OUT.mkdir(parents=True, exist_ok=True)

# Ghia et al. 1982 reference data (Re=100, centerline u at x=0.5)
GHIA_Y = [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
           0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000]
GHIA_U = [0.0000,-0.0372,-0.0419,-0.0477,-0.0643,-0.1015,-0.1566,-0.2109,
          -0.2058,-0.1364, 0.0033, 0.2315, 0.6872, 0.7372, 0.7887, 0.8412, 1.0000]


class MLP(nn.Module):
    def __init__(self, width=64, depth=5):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.out_u = nn.Linear(width, 1)
        self.out_v = nn.Linear(width, 1)
        self.out_p = nn.Linear(width, 1)

    def forward(self, xy):
        h = self.net(xy)
        return self.out_u(h).squeeze(-1), self.out_v(h).squeeze(-1), self.out_p(h).squeeze(-1)


def make_grid():
    x = torch.linspace(0, 1, Nx, device=device)
    y = torch.linspace(0, 1, Ny, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    dx = float(x[1]-x[0]); dy = float(y[1]-y[0])
    return X, Y, dx, dy


def pde_residual_pytorch_steady(U, V, P, dx, dy):
    """Steady incompressible NS with P and div."""
    u_x  = (U[2:,1:-1] - U[:-2,1:-1]) / (2*dx)
    u_y  = (U[1:-1,2:] - U[1:-1,:-2]) / (2*dy)
    u_xx = (U[2:,1:-1] - 2*U[1:-1,1:-1] + U[:-2,1:-1]) / dx**2
    u_yy = (U[1:-1,2:] - 2*U[1:-1,1:-1] + U[1:-1,:-2]) / dy**2
    v_x  = (V[2:,1:-1] - V[:-2,1:-1]) / (2*dx)
    v_y  = (V[1:-1,2:] - V[1:-1,:-2]) / (2*dy)
    v_xx = (V[2:,1:-1] - 2*V[1:-1,1:-1] + V[:-2,1:-1]) / dx**2
    v_yy = (V[1:-1,2:] - 2*V[1:-1,1:-1] + V[1:-1,:-2]) / dy**2
    p_x  = (P[2:,1:-1] - P[:-2,1:-1]) / (2*dx)
    p_y  = (P[1:-1,2:] - P[1:-1,:-2]) / (2*dy)
    u_c = U[1:-1,1:-1]; v_c = V[1:-1,1:-1]
    res_u = u_c*u_x + v_c*u_y + p_x - nu*(u_xx+u_yy)
    res_v = u_c*v_x + v_c*v_y + p_y - nu*(v_xx+v_yy)
    res_div = u_x + v_y
    return (res_u**2 + res_v**2 + res_div**2).mean()


def bc_loss(U, V):
    """Lid: top u=1,v=0; walls u=v=0."""
    top  = (U[:,-1]-1)**2 + V[:,-1]**2
    bot  = U[:,0]**2 + V[:,0]**2
    left = U[0,:]**2 + V[0,:]**2
    right= U[-1,:]**2 + V[-1,:]**2
    return (top.mean() + bot.mean() + left.mean() + right.mean())


def train(backend, epochs=5000, lr=1e-3, loss_threshold=1e-3):
    X, Y, dx, dy = make_grid()
    xy = torch.stack([X.flatten(), Y.flatten()], dim=1)

    model = MLP().to(device)
    if backend == "compile":
        model = torch.compile(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.5)

    history=[]; t2s=None; t0=time.time()

    for ep in range(1, epochs+1):
        opt.zero_grad()
        u_flat, v_flat, p_flat = model(xy)
        U = u_flat.reshape(Nx, Ny)
        V = v_flat.reshape(Nx, Ny)
        P = p_flat.reshape(Nx, Ny)

        if backend == "vanilla":
            xy_g = xy.detach().requires_grad_(True)
            u_g, v_g, p_g = model(xy_g)
            u_g = u_g.reshape(Nx,Ny); v_g = v_g.reshape(Nx,Ny); p_g = p_g.reshape(Nx,Ny)
            gu = torch.autograd.grad(u_g.sum(), xy_g, create_graph=True)[0]
            gv = torch.autograd.grad(v_g.sum(), xy_g, create_graph=True)[0]
            gp = torch.autograd.grad(p_g.sum(), xy_g, create_graph=True)[0]
            u_x=gu[:,0].reshape(Nx,Ny); u_y=gu[:,1].reshape(Nx,Ny)
            v_x=gv[:,0].reshape(Nx,Ny); v_y=gv[:,1].reshape(Nx,Ny)
            p_x=gp[:,0].reshape(Nx,Ny); p_y=gp[:,1].reshape(Nx,Ny)
            u_xx=torch.autograd.grad(u_x.sum(),xy_g,create_graph=True)[0][:,0].reshape(Nx,Ny)
            u_yy=torch.autograd.grad(u_y.sum(),xy_g,create_graph=True)[0][:,1].reshape(Nx,Ny)
            v_xx=torch.autograd.grad(v_x.sum(),xy_g,create_graph=True)[0][:,0].reshape(Nx,Ny)
            v_yy=torch.autograd.grad(v_y.sum(),xy_g,create_graph=True)[0][:,1].reshape(Nx,Ny)
            res_u=u_g*u_x+v_g*u_y+p_x-nu*(u_xx+u_yy)
            res_v=u_g*v_x+v_g*v_y+p_y-nu*(v_xx+v_yy)
            res_div=u_x+v_y
            loss_pde=(res_u**2+res_v**2+res_div**2).mean()
        else:
            loss_pde = pde_residual_pytorch_steady(U, V, P, dx, dy)

        loss = loss_pde + 10*bc_loss(U, V)
        loss.backward(); opt.step(); sch.step()

        wall=time.time()-t0; lv=loss.item()
        history.append((wall,ep,lv))
        if t2s is None and lv < loss_threshold:
            t2s=wall; print(f"  [{backend}] T2S={wall:.1f}s at ep {ep}"); break
        if ep%1000==0:
            print(f"[{backend}] ep {ep:5d} loss={lv:.3e} t={wall:.1f}s")

    elapsed=time.time()-t0; mem=torch.cuda.max_memory_allocated()/1e9
    print(f"[{backend}] Done: {elapsed:.1f}s  mem={mem:.3f}GB")
    return model, history, elapsed, mem, t2s, X, Y


def plot_results(model, X, Y, name):
    xy = torch.stack([X.flatten(), Y.flatten()], dim=1)
    with torch.no_grad():
        u_flat, v_flat, _ = model(xy)
    U = u_flat.reshape(Nx,Ny).cpu().numpy()
    V = v_flat.reshape(Nx,Ny).cpu().numpy()
    x_np=X[:,0].cpu().numpy(); y_np=Y[0,:].cpu().numpy()

    # Ghia comparison: u at x=0.5
    xi = Nx//2
    u_center = U[xi,:]

    fig, axes = plt.subplots(1,3,figsize=(15,4))
    speed = np.sqrt(U**2+V**2)
    axes[0].contourf(x_np,y_np,speed.T,levels=20,cmap='viridis')
    axes[0].set_title(f'Speed — {name}')
    axes[1].contourf(x_np,y_np,U.T,levels=20,cmap='RdBu_r')
    axes[1].set_title('u velocity')
    axes[2].plot(u_center, y_np, 'b-', label='PINN', lw=2)
    axes[2].plot(GHIA_U, GHIA_Y, 'ro', label='Ghia Re=100', ms=5)
    axes[2].set_xlabel('u'); axes[2].set_ylabel('y')
    axes[2].set_title('Centerline u (x=0.5)'); axes[2].legend(); axes[2].grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT/f"solution_{name}.png",dpi=150); plt.close()

    # L2 vs Ghia at centerline
    ghia_y = np.array(GHIA_Y); ghia_u = np.array(GHIA_U)
    u_interp = np.interp(ghia_y, y_np, u_center)
    l2 = np.linalg.norm(u_interp-ghia_u)/np.linalg.norm(ghia_u)
    print(f"[{name}] L2 vs Ghia centerline: {l2:.4e}")
    return l2


if __name__ == "__main__":
    import triton
    backends = ["vanilla", "canpinn", "compile", "triton"]
    results = {}; histories = {}

    X, Y, dx, dy = make_grid()
    print("=== Kernel benchmark (steady NS) ===")
    U_t = torch.rand(Nx,Ny,device=device); V_t = torch.rand(Nx,Ny,device=device)
    P_t = torch.rand(Nx,Ny,device=device)
    ms_pt = triton.testing.do_bench(lambda: pde_residual_pytorch_steady(U_t,V_t,P_t,dx,dy))
    print(f"PyTorch FD: {ms_pt:.3f}ms")

    for b in backends:
        print(f"\n{'='*50}\nTraining: {b}")
        ckpt=OUT/f"model_{b}.pt"; meta=OUT/f"meta_{b}.npy"
        torch.cuda.reset_peak_memory_stats()
        if ckpt.exists() and meta.exists():
            print("  [cached]")
            model=MLP().to(device)
            model.load_state_dict(torch.load(ckpt,map_location=device))
            m=np.load(meta,allow_pickle=True).item()
            hist,elapsed,mem,t2s=m['history'],m['elapsed'],m['mem'],m['t2s']
            _,_,_,_,X,Y=make_grid(),make_grid()
            X,Y,_,_=make_grid()
        else:
            model,hist,elapsed,mem,t2s,X,Y=train(b,epochs=5000)
            torch.save(getattr(model,'_orig_mod',model).state_dict(),ckpt)
            np.save(meta,dict(history=hist,elapsed=elapsed,mem=mem,t2s=t2s))
        l2=plot_results(model,X,Y,b)
        results[b]=dict(elapsed=elapsed,mem=mem,t2s=t2s,l2=l2)
        histories[b]=hist

    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for b,hist in histories.items():
        axes[0].semilogy([h[1] for h in hist],[h[2] for h in hist],label=b)
        axes[1].semilogy([h[0] for h in hist],[h[2] for h in hist],label=b)
    for ax,xl in zip(axes,['Epoch','Wall time (s)']):
        ax.set_xlabel(xl);ax.set_ylabel('Loss');ax.legend();ax.grid(True,alpha=0.3)
    plt.tight_layout();plt.savefig(OUT/"loss_curves.png",dpi=150);plt.close()

    print("\n=== Summary ===")
    print(f"{'backend':<12} {'time(s)':>8} {'mem(GB)':>8} {'T2S(s)':>8} {'L2_Ghia':>10}")
    for b in backends:
        r=results[b]; t2s=f"{r['t2s']:.1f}" if r['t2s'] else "N/A"
        print(f"{b:<12} {r['elapsed']:>8.1f} {r['mem']:>8.3f} {t2s:>8} {r['l2']:>10.4e}")
