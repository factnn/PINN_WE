"""Shared utilities and baseline models for 2D TGV experiments."""
import sys, time, argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 物理参数与全局配置 ---
nu = 0.01
Nx, Ny, Nt = 64, 64, 20
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(__file__).parent.parent / "output" / "tgv_2d"
OUT.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. 网络架构 (骨架)
# ==========================================
class MLP(nn.Module):
    """标准的点对点全连接网络 (用于 CAN-PINN)"""
    def __init__(self, width=128, depth=6):
        super().__init__()
        layers = [nn.Linear(3, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        
        # 必须输出 U, V, P 三个场
        self.out_u = nn.Linear(width, 1)
        self.out_v = nn.Linear(width, 1)
        self.out_p = nn.Linear(width, 1)

    def forward(self, xyt):
        h = self.net(xyt)
        return self.out_u(h).squeeze(-1), self.out_v(h).squeeze(-1), self.out_p(h).squeeze(-1)


class PhyCNN(nn.Module):
    """基于网格的 3D 通道卷积网络"""
    def __init__(self, channels=32):
        super().__init__()
        # 输入通道为 3：X坐标, Y坐标, T时间戳
        self.enc = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv2d(channels, 3, kernel_size=5, padding=2), # 输出 U, V, P
        )

    def forward(self, X, Y, T):
        # 组装输入 [Batch(Nt), Channels(3), Nx, Ny]
        inputs = torch.stack([X, Y, T], dim=1) 
        out = self.enc(inputs)
        
        U = out[:, 0, :, :]
        V = out[:, 1, :, :]
        P = out[:, 2, :, :]
        return U, V, P


# ==========================================
# 2. 物理环境与解析解
# ==========================================
def make_grid():
    x = torch.linspace(0, 2*np.pi, Nx, device=device)
    y = torch.linspace(0, 2*np.pi, Ny, device=device)
    t = torch.linspace(0, 1, Nt, device=device)
    T, X, Y = torch.meshgrid(t, x, y, indexing='ij')
    dx = float(x[1] - x[0]); dy = float(y[1] - y[0]); dt = float(t[1] - t[0])
    return X, Y, T, dx, dy, dt


def exact_uvp(X, Y, T):
    """2D TGV 的精确解析解 (包含压力 P)"""
    U = torch.sin(X) * torch.cos(Y) * torch.exp(-2*nu*T)
    V = -torch.cos(X) * torch.sin(Y) * torch.exp(-2*nu*T)
    P = -0.25 * (torch.cos(2*X) + torch.cos(2*Y)) * torch.exp(-4*nu*T)
    return U, V, P


# ==========================================
# 3. 物理约束与损失函数 (你的 Triton 靶点)
# ==========================================
def pde_residual_pytorch(U, V, P, dx, dy, dt):
    """纯 PyTorch 原生切片实现的 2D N-S 方程残差。P=None 时跳过压力项（用于 Triton backward）"""
    u_t = (U[2:,1:-1,1:-1]-U[:-2,1:-1,1:-1])/(2*dt)
    v_t = (V[2:,1:-1,1:-1]-V[:-2,1:-1,1:-1])/(2*dt)
    u_x = (U[1:-1,2:,1:-1]-U[1:-1,:-2,1:-1])/(2*dx)
    u_y = (U[1:-1,1:-1,2:]-U[1:-1,1:-1,:-2])/(2*dy)
    v_x = (V[1:-1,2:,1:-1]-V[1:-1,:-2,1:-1])/(2*dx)
    v_y = (V[1:-1,1:-1,2:]-V[1:-1,1:-1,:-2])/(2*dy)
    u_xx = (U[1:-1,2:,1:-1]-2*U[1:-1,1:-1,1:-1]+U[1:-1,:-2,1:-1])/dx**2
    u_yy = (U[1:-1,1:-1,2:]-2*U[1:-1,1:-1,1:-1]+U[1:-1,1:-1,:-2])/dy**2
    v_xx = (V[1:-1,2:,1:-1]-2*V[1:-1,1:-1,1:-1]+V[1:-1,:-2,1:-1])/dx**2
    v_yy = (V[1:-1,1:-1,2:]-2*V[1:-1,1:-1,1:-1]+V[1:-1,1:-1,:-2])/dy**2
    uc=U[1:-1,1:-1,1:-1]; vc=V[1:-1,1:-1,1:-1]
    if P is not None:
        p_x=(P[1:-1,2:,1:-1]-P[1:-1,:-2,1:-1])/(2*dx)
        p_y=(P[1:-1,1:-1,2:]-P[1:-1,1:-1,:-2])/(2*dy)
    else:
        p_x=p_y=0.0
    res_u=u_t+uc*u_x+vc*u_y+p_x-nu*(u_xx+u_yy)
    res_v=v_t+uc*v_x+vc*v_y+p_y-nu*(v_xx+v_yy)
    res_div=u_x+v_y
    return (res_u**2+res_v**2+res_div**2).mean()


def ic_bc_loss(U, V, P, U_exact, V_exact, P_exact):
    """初始条件 (IC) 与周期性边界条件 (Periodic BC)"""
    # IC: t=0 时的精确场匹配
    ic_u = ((U[0] - U_exact[0])**2).mean()
    ic_v = ((V[0] - V_exact[0])**2).mean()
    ic_p = ((P[0] - P_exact[0])**2).mean()
    
    # BC: X 和 Y 方向的首尾相接 (周期性)
    bc_u = ((U[:, 0, :] - U[:, -1, :])**2 + (U[:, :, 0] - U[:, :, -1])**2).mean()
    bc_v = ((V[:, 0, :] - V[:, -1, :])**2 + (V[:, :, 0] - V[:, :, -1])**2).mean()
    bc_p = ((P[:, 0, :] - P[:, -1, :])**2 + (P[:, :, 0] - P[:, :, -1])**2).mean()
    
    return ic_u + ic_v + ic_p + bc_u + bc_v + bc_p


def unified_loss_fn(model, xyt, X, Y, T, U_exact, V_exact, P_exact, dx, dy, dt):
    U, V, P = infer(model, xyt, X, Y, T)
    return pde_residual_pytorch(U, V, P, dx, dy, dt) + ic_bc_loss(U, V, P, U_exact, V_exact, P_exact)


# ==========================================
# 4. 训练主干与可视化
# ==========================================
def train_and_save(backend_name, model_fn, loss_fn=None,
                   max_epochs=50000, lr=1e-3, loss_threshold=1e-3, runs=1):
    """
    model_fn: callable -> model (e.g. MLP, PhyCNN, lambda: torch.compile(MLP()))
    loss_fn: (model, xyt, X, Y, T, U_exact, V_exact, P_exact, dx, dy, dt) -> scalar
             defaults to unified_loss_fn
    """
    if loss_fn is None:
        loss_fn = unified_loss_fn

    X, Y, T, dx, dy, dt = make_grid()
    U_exact, V_exact, P_exact = exact_uvp(X, Y, T)
    xyt = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)

    all_elapsed = []; all_t2s = []

    for run_i in range(runs):
        model = model_fn().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)

        history = []; t2s = None; t2s_ep = None; t0 = time.time()
        step_times = []
        print(f"[{backend_name}] run {run_i+1}/{runs} | Grid: {Nx}x{Ny}x{Nt}")

        for ep in range(1, max_epochs + 1):
            t_step = time.time()
            opt.zero_grad()
            loss = loss_fn(model, xyt, X, Y, T, U_exact, V_exact, P_exact, dx, dy, dt)
            loss.backward(); opt.step(); sch.step()
            torch.cuda.synchronize()
            step_times.append(time.time() - t_step)

            wall = time.time()-t0; lv = loss.item()
            history.append((wall, ep, lv))
            if t2s is None and lv < loss_threshold:
                t2s = wall; t2s_ep = ep
                print(f"  [{backend_name}] run {run_i+1}: T2S={wall:.1f}s at ep {ep}")
                break
            if ep % 5000 == 0:
                print(f"[{backend_name}] run {run_i+1} ep {ep:5d} loss={lv:.3e} t={wall:.1f}s")

        elapsed = time.time()-t0
        mem = torch.cuda.max_memory_allocated()/1e9
        avg_step_ms = np.mean(step_times) * 1000
        all_elapsed.append(elapsed); all_t2s.append(t2s)
        print(f"[{backend_name}] run {run_i+1}: {elapsed:.1f}s  mem={mem:.3f}GB  avg_step={avg_step_ms:.2f}ms  epochs={t2s_ep or ep}")

    ckpt = OUT/f"model_{backend_name}.pt"
    torch.save(getattr(model,'_orig_mod',model).state_dict(), ckpt)

    # Memory bandwidth: bytes read/written per step / avg_step_time
    # U,V,P each [Nt,Nx,Ny] float32 = 4 bytes; stencil reads ~14 neighbors per point
    n_interior = (Nt-2)*(Nx-2)*(Ny-2)
    bytes_per_step = 3 * Nt * Nx * Ny * 4 * 14  # rough estimate
    mem_bw_gbs = bytes_per_step / (avg_step_ms * 1e-3) / 1e9

    t2s_v = [t for t in all_t2s if t is not None]
    metrics = dict(
        history=history, elapsed=elapsed, mem_gb=mem,
        t2s=t2s, t2s_ep=t2s_ep,
        avg_step_ms=avg_step_ms,
        mem_bw_gbs=mem_bw_gbs,
        all_elapsed=all_elapsed, all_t2s=all_t2s,
    )
    np.save(OUT/f"meta_{backend_name}.npy", metrics)

    # plot
    with torch.no_grad():
        U_pred, V_pred, _ = infer(model, xyt, X, Y, T)
    Ue=U_exact.cpu().numpy(); Ve=V_exact.cpu().numpy()
    Up=U_pred.reshape(Nt,Nx,Ny).cpu().numpy(); Vp=V_pred.reshape(Nt,Nx,Ny).cpu().numpy()
    l2 = np.linalg.norm(Up-Ue)/(np.linalg.norm(Ue)+1e-12)
    _plot(Up, Vp, Ue, Ve, X, Y, backend_name, l2)

    print(f"\n[{backend_name}] === Summary ===")
    print(f"  T2S          : {t2s:.1f}s" if t2s else "  T2S          : N/A")
    print(f"  Total_Epochs : {t2s_ep or ep}")
    print(f"  Avg_Step_ms  : {avg_step_ms:.2f}")
    print(f"  Peak_Mem_GB  : {mem:.3f}")
    print(f"  Mem_BW_GBs   : {mem_bw_gbs:.1f}")
    print(f"  L2_err       : {l2:.4e}")
    return dict(elapsed=np.median(all_elapsed), mem_gb=mem, t2s=t2s,
                t2s_ep=t2s_ep, avg_step_ms=avg_step_ms,
                mem_bw_gbs=mem_bw_gbs, l2=l2, history=history)


def infer(model, xyt, X, Y, T):
    """Infer U, V, P from model regardless of type (MLP or PhyCNN, compiled or not)."""
    inner = getattr(model, '_orig_mod', model)
    if isinstance(inner, PhyCNN):
        U, V, P = model(X, Y, T)
    else:
        u_f, v_f, p_f = model(xyt)
        U = u_f.reshape(Nt, Nx, Ny)
        V = v_f.reshape(Nt, Nx, Ny)
        P = p_f.reshape(Nt, Nx, Ny)
    return U, V, P


def base_argparser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--runs', type=int, default=1)
    p.add_argument('--max-epochs', type=int, default=50000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--threshold', type=float, default=1e-3)
    p.add_argument('--gpu', type=int, default=0)
    return p


def _plot(U, V, Ue, Ve, X, Y, name, l2):
    x_np = X[0, :, 0].cpu().numpy()
    y_np = Y[0, 0, :].cpu().numpy()
    ti = Nt // 2  # 取中间时间步画图
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, (f, fe, label) in enumerate([(U, Ue, 'u'), (V, Ve, 'v')]):
        # Pred
        im = axes[row, 0].contourf(x_np, y_np, f[ti].T, levels=20, cmap='RdBu_r')
        plt.colorbar(im, ax=axes[row, 0]); axes[row, 0].set_title(f'PINN {label}')
        # Exact
        im2 = axes[row, 1].contourf(x_np, y_np, fe[ti].T, levels=20, cmap='RdBu_r')
        plt.colorbar(im2, ax=axes[row, 1]); axes[row, 1].set_title(f'Exact {label}')
        # Error
        err = np.abs(f[ti] - fe[ti])
        im3 = axes[row, 2].contourf(x_np, y_np, err.T, levels=20, cmap='hot_r')
        plt.colorbar(im3, ax=axes[row, 2]); axes[row, 2].set_title(f'|Error|')
        
    plt.suptitle(f'TGV 2D ({name}) | t={ti}/{Nt} | L2={l2:.2e}', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUT / f"solution_{name}.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    # 快速测试入口：先跑 5000 步看看显存和耗时
    print("="*50)
    train_and_save("CAN-PINN_Baseline", MLP, max_epochs=5000)
    print("="*50)
    train_and_save("PhyCNN_Baseline", PhyCNN, max_epochs=5000)