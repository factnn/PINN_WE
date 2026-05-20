#!/usr/bin/env python3
"""
Taylor-Green Vortex 3D — 有粘 NS, Re=100 验证案例

目标: 验证方程正确性。Re=100 纯层流衰减，有近似解析解:
    Ek(t) ≈ Ek(0) * exp(-2*nu*k0^2*t)
    其中 nu = mu/rho0, k0 = 1/L

参数: γ=1.4, Ma=0.1, Re=100, Pr=1, L=1, V0=1
域:   [-π, π]³, t ∈ [0, 5]
"""
import sys
import os
import time
import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

current_file_dir = os.path.dirname(os.path.abspath(__file__))
pinn_src_path = os.path.abspath(os.path.join(current_file_dir, '..', '..', '..', 'PINNsrc'))
if pinn_src_path not in sys.path:
    sys.path.insert(0, pinn_src_path)

from PINNs import PINNs_WE_NS_3D, gradients, setup_seed, to_numpy

setup_seed(42)
dtype = torch.float64

# ========== 物理参数 ==========
gamma = 1.4
L = 1.0
V0 = 1.0
rho0 = 1.0
Ma = 0.1
c0 = V0 / Ma
p0 = rho0 * c0**2 / gamma  # ≈ 71.43
Re = 100.0
mu = rho0 * V0 * L / Re     # = 0.01
Pr = 1.0
nu = mu / rho0               # 运动粘度

pi = math.pi
Xs, Xe = -pi * L, pi * L
Ys, Ye = -pi * L, pi * L
Zs, Ze = -pi * L, pi * L
Ts, Te = 0.0, 1.0

print(f'Taylor-Green Vortex 3D — Re={Re} 验证案例')
print(f'γ={gamma}, L={L}, V0={V0}, Ma={Ma}, Pr={Pr}')
print(f'ρ0={rho0}, p0={p0:.2f}, c0={c0:.1f}')
print(f'Re={Re}, μ={mu:.6f}, ν={nu:.6f}')
print(f'域: [{Xs:.2f}, {Xe:.2f}]³, t=[{Ts}, {Te}]')
print(f'理论: Ek(t)/Ek(0) ≈ exp(-2νt) = exp({-2*nu*Te:.4f}) = {np.exp(-2*nu*Te):.4f} at t={Te}')

# ========== GPU ==========
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {cuda}')

# ========== 初始条件 ==========
def IC_TGV(x):
    N = x.shape[0]
    xc, yc, zc = x[:, 1]/L, x[:, 2]/L, x[:, 3]/L
    u_init = V0 * np.sin(xc) * np.cos(yc) * np.cos(zc)
    v_init = -V0 * np.cos(xc) * np.sin(yc) * np.cos(zc)
    w_init = np.zeros(N)
    p_init = p0 + (rho0 * V0**2 / 16.0) * (
        np.cos(2*xc) + np.cos(2*yc)) * (np.cos(2*zc) + 2)
    rho_init = np.full(N, rho0)
    return rho_init, u_init, v_init, w_init, p_init


# ========== 采样 ==========
from smt.sampling_methods import LHS

N_ic = 8000
N_int = 30000
N_bc = 1500
resample_every = 500  # 每500 epoch重新采样内部配点和BC点

xlimits_ic = np.array([[0., 0.], [Xs, Xe], [Ys, Ye], [Zs, Ze]])
xlimits_int = np.array([[Ts, Te], [Xs, Xe], [Ys, Ye], [Zs, Ze]])

def tt(arr, rg=False):
    return torch.tensor(arr, requires_grad=rg, dtype=dtype).to(cuda)

# IC 只采一次（有确定的label）
x_ic_np = LHS(xlimits=xlimits_ic)(N_ic)
rho_ic, u_ic, v_ic, w_ic, p_ic = IC_TGV(x_ic_np)
x_ic_t = tt(x_ic_np, True)
rho_ic_t = tt(rho_ic); u_ic_t = tt(u_ic)
v_ic_t = tt(v_ic); w_ic_t = tt(w_ic); p_ic_t = tt(p_ic)

def sample_bc_pair(fixed_dim, fixed_lo, fixed_hi, Npts):
    t_bc = np.random.uniform(Ts, Te, Npts)
    coords = [np.random.uniform(Xs, Xe, Npts),
              np.random.uniform(Ys, Ye, Npts),
              np.random.uniform(Zs, Ze, Npts)]
    xL = np.stack([t_bc] + coords, axis=1)
    xR = xL.copy()
    xL[:, fixed_dim+1] = fixed_lo
    xR[:, fixed_dim+1] = fixed_hi
    return xL, xR

def resample_collocation():
    """重采样内部配点和BC点，返回所有tensor"""
    x_int_np = LHS(xlimits=xlimits_int)(N_int)
    x_int_t = tt(x_int_np, True)

    xL_x, xR_x = sample_bc_pair(0, Xs, Xe, N_bc)
    xL_y, xR_y = sample_bc_pair(1, Ys, Ye, N_bc)
    xL_z, xR_z = sample_bc_pair(2, Zs, Ze, N_bc)

    return (x_int_t,
            tt(xL_x, True), tt(xR_x, True),
            tt(xL_y, True), tt(xR_y, True),
            tt(xL_z, True), tt(xR_z, True))

# 初始采样
(x_int_t,
 x_bc_xL_t, x_bc_xR_t,
 x_bc_yL_t, x_bc_yR_t,
 x_bc_zL_t, x_bc_zR_t) = resample_collocation()

print(f'采样: IC={N_ic}, 内部={N_int}, BC={N_bc*6}, 重采样间隔={resample_every}')

# ========== 网络 ==========
Nl, Nn = 6, 128
model = PINNs_WE_NS_3D(Nl=Nl, Nn=Nn, rho0=rho0, p0=p0, V0=V0).to(dtype).to(cuda)
n_params = sum(p.numel() for p in model.parameters())
print(f'网络: {Nl} layers × {Nn} neurons, {n_params} params')

# ========== 训练 ==========
def compute_loss():
    loss_pde = model.loss_pde(x_int_t, k=0, mu=mu, gamma=gamma,
                               rho_ref=rho0, p_ref=p0, V_ref=V0, Pr=Pr)
    loss_ic = model.loss_ic(x_ic_t, rho_ic_t, u_ic_t, v_ic_t, w_ic_t, p_ic_t,
                             rho_ref=rho0, p_ref=p0, V_ref=V0)

    y_xL = model(x_bc_xL_t); y_xR = model(x_bc_xR_t)
    y_yL = model(x_bc_yL_t); y_yR = model(x_bc_yR_t)
    y_zL = model(x_bc_zL_t); y_zR = model(x_bc_zR_t)
    loss_bc = (torch.mean((y_xL - y_xR)**2) +
               torch.mean((y_yL - y_yR)**2) +
               torch.mean((y_zL - y_zR)**2))

    return loss_pde, loss_ic, loss_bc

# Adam
print('\n========== Adam 阶段 ==========')
epochs_adam = 20000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

loss_history = []
tic = time.time()

for epoch in range(1, epochs_adam + 1):
    # 每 resample_every 步重采样内部配点和BC点
    if epoch > 1 and epoch % resample_every == 0:
        (x_int_t,
         x_bc_xL_t, x_bc_xR_t,
         x_bc_yL_t, x_bc_yR_t,
         x_bc_zL_t, x_bc_zR_t) = resample_collocation()

    optimizer.zero_grad()
    loss_pde, loss_ic, loss_bc = compute_loss()
    loss = 10*loss_pde + 10*loss_ic + 10*loss_bc
    loss.backward()
    optimizer.step()
    scheduler.step()

    loss_history.append(to_numpy(loss))

    if epoch % 2000 == 0 or epoch == 1:
        lr_now = optimizer.param_groups[0]['lr']
        print(f'  epoch {epoch:5d}  loss={loss.item():.4e}  '
              f'pde={loss_pde.item():.4e}  ic={loss_ic.item():.4e}  '
              f'bc={loss_bc.item():.4e}  lr={lr_now:.1e}')

adam_time = time.time() - tic
print(f'Adam 完成: {adam_time:.1f}s')

# LBFGS
print('\n========== LBFGS 阶段 ==========')
optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20,
                                     history_size=50, line_search_fn='strong_wolfe')
epochs_lbfgs = 500
tic_lbfgs = time.time()

for epoch in range(1, epochs_lbfgs + 1):
    def closure():
        optimizer_lbfgs.zero_grad()
        loss_pde, loss_ic, loss_bc = compute_loss()
        loss = 10*loss_pde + 10*loss_ic + 10*loss_bc
        loss.backward()
        return loss

    loss = optimizer_lbfgs.step(closure)
    loss_history.append(to_numpy(loss))

    if epoch % 100 == 0 or epoch == 1:
        print(f'  LBFGS {epoch:4d}  loss={loss.item():.4e}')

lbfgs_time = time.time() - tic_lbfgs
total_time = time.time() - tic
print(f'LBFGS 完成: {lbfgs_time:.1f}s')
print(f'总训练时间: {total_time:.1f}s')

# ========== 评估 ==========
model.eval()
print('\n========== 评估 ==========')

# 计算动能演化
Nt_eval = 50
t_eval = np.linspace(Ts, Te, Nt_eval)
Ek_arr = np.zeros(Nt_eval)

nk = 24
xk = np.linspace(Xs, Xe, nk)
yk = np.linspace(Ys, Ye, nk)
zk = np.linspace(Zs, Ze, nk)
Xk, Yk, Zk = np.meshgrid(xk, yk, zk, indexing='ij')
Xk_f, Yk_f, Zk_f = Xk.flatten(), Yk.flatten(), Zk.flatten()

for i, t_val in enumerate(t_eval):
    Tk_f = np.full_like(Xk_f, t_val)
    x_k = np.stack([Tk_f, Xk_f, Yk_f, Zk_f], axis=1)
    x_k_t = torch.tensor(x_k, dtype=dtype).to(cuda)

    with torch.no_grad():
        pred = model(x_k_t)
    rho_k = to_numpy(pred[:, 0])
    u_k = to_numpy(pred[:, 2])
    v_k = to_numpy(pred[:, 3])
    w_k = to_numpy(pred[:, 4])

    ke = rho_k * (u_k**2 + v_k**2 + w_k**2)
    Ek_arr[i] = np.mean(ke) / (2.0 * rho0)

# 理论值 (不可压缩近似, 最低波数 k0=1/L=1)
# Ek(t) ≈ Ek(0) * exp(-2*nu*k0^2*t)
Ek_theory = Ek_arr[0] * np.exp(-2.0 * nu * t_eval)

# 耗散率
dt_eval = t_eval[1] - t_eval[0]
eps_pinn = np.zeros(Nt_eval)
eps_pinn[1:-1] = -(Ek_arr[2:] - Ek_arr[:-2]) / (2*dt_eval)
eps_pinn[0] = -(Ek_arr[1] - Ek_arr[0]) / dt_eval
eps_pinn[-1] = -(Ek_arr[-1] - Ek_arr[-2]) / dt_eval

eps_theory = 2.0 * nu * Ek_theory  # ε = 2ν Ek (低Re近似)

# ========== 绘图 ==========
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_name = f'TGV3D_Re{Re:.0f}_Ma{Ma}_Viscous_{timestamp}'
output_dir = os.path.join(current_file_dir, 'output', run_name)
os.makedirs(output_dir, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Ek(t)
ax = axes[0, 0]
ax.plot(t_eval, Ek_arr, 'b-', lw=2, label='PINN')
ax.plot(t_eval, Ek_theory, 'r--', lw=2, label=r'Theory: $E_k(0) e^{-2\nu t}$')
ax.set_xlabel('t')
ax.set_ylabel('Ek')
ax.set_title('Kinetic Energy')
ax.legend()
ax.grid(True, alpha=0.3)

# Ek/Ek(0)
ax = axes[0, 1]
ax.plot(t_eval, Ek_arr / Ek_arr[0], 'b-', lw=2, label='PINN')
ax.plot(t_eval, np.exp(-2*nu*t_eval), 'r--', lw=2, label=r'$e^{-2\nu t}$')
ax.set_xlabel('t')
ax.set_ylabel('Ek / Ek(0)')
ax.set_title('Normalized Kinetic Energy')
ax.legend()
ax.grid(True, alpha=0.3)

# 耗散率
ax = axes[1, 0]
ax.plot(t_eval, eps_pinn, 'b-', lw=2, label='PINN')
ax.plot(t_eval, eps_theory, 'r--', lw=2, label=r'Theory: $2\nu E_k$')
ax.set_xlabel('t')
ax.set_ylabel('ε = -dEk/dt')
ax.set_title('Dissipation Rate')
ax.legend()
ax.grid(True, alpha=0.3)

# Loss
ax = axes[1, 1]
ax.semilogy(loss_history, 'b-', lw=0.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss')
ax.grid(True, alpha=0.3)

fig.suptitle(f'TGV Re={Re:.0f}, Ma={Ma}, Pr={Pr} — 方程验证',
             fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'Re100_validation.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)

# 流场快照 (z=0)
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))
t_snapshots = [0.0, 0.5, 1.0]

for idx, t_val in enumerate(t_snapshots):
    nx_eval = 40
    x_lin = np.linspace(Xs, Xe, nx_eval)
    y_lin = np.linspace(Ys, Ye, nx_eval)
    Xg, Yg = np.meshgrid(x_lin, y_lin, indexing='ij')
    Tg = np.full_like(Xg, t_val)
    Zg = np.zeros_like(Xg)

    x_test = np.stack([Tg, Xg, Yg, Zg], axis=-1).reshape(-1, 4)
    x_test_t = torch.tensor(x_test, dtype=dtype).to(cuda)

    with torch.no_grad():
        pred = model(x_test_t)
    u_p = to_numpy(pred[:, 2]).reshape(nx_eval, nx_eval)
    v_p = to_numpy(pred[:, 3]).reshape(nx_eval, nx_eval)
    vel = np.sqrt(u_p**2 + v_p**2)

    ax = axes2[idx]
    c = ax.contourf(Xg, Yg, vel, levels=20, cmap='jet')
    skip = 4
    ax.quiver(Xg[::skip, ::skip], Yg[::skip, ::skip],
              u_p[::skip, ::skip], v_p[::skip, ::skip],
              color='k', alpha=0.5, scale=15)
    ax.set_title(f't = {t_val:.1f}')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(c, ax=ax, label='|V|')

fig2.suptitle('Velocity field (z=0 slice)', fontsize=14)
fig2.tight_layout()
fig2.savefig(os.path.join(output_dir, 'Re100_flow_snapshots.png'),
             dpi=150, bbox_inches='tight')
plt.close(fig2)

# ========== 物理量分布图 (z=0 切面, 多时刻多变量) ==========
print('绘制物理量分布图...')
nx_eval = 60
x_lin = np.linspace(Xs, Xe, nx_eval)
y_lin = np.linspace(Ys, Ye, nx_eval)
Xg, Yg = np.meshgrid(x_lin, y_lin, indexing='ij')
Zg = np.zeros_like(Xg)

t_snapshots_full = [0.0, 0.25, 0.5, 1.0]
n_snap = len(t_snapshots_full)

# 收集所有时刻的预测值
all_preds = {}
for t_val in t_snapshots_full:
    Tg = np.full_like(Xg, t_val)
    x_test = np.stack([Tg, Xg, Yg, Zg], axis=-1).reshape(-1, 4)
    x_test_t = torch.tensor(x_test, dtype=dtype).to(cuda)
    with torch.no_grad():
        pred = model(x_test_t)
    all_preds[t_val] = {
        'rho': to_numpy(pred[:, 0]).reshape(nx_eval, nx_eval),
        'p':   to_numpy(pred[:, 1]).reshape(nx_eval, nx_eval),
        'u':   to_numpy(pred[:, 2]).reshape(nx_eval, nx_eval),
        'v':   to_numpy(pred[:, 3]).reshape(nx_eval, nx_eval),
        'w':   to_numpy(pred[:, 4]).reshape(nx_eval, nx_eval),
    }

# --- 图3: 密度 ρ 分布 ---
fig3, axes3 = plt.subplots(1, n_snap, figsize=(5*n_snap, 4.5))
for idx, t_val in enumerate(t_snapshots_full):
    rho_p = all_preds[t_val]['rho']
    ax = axes3[idx]
    c = ax.contourf(Xg, Yg, rho_p, levels=20, cmap='coolwarm')
    ax.set_title(f't = {t_val:.1f}')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(c, ax=ax, format='%.4f')
fig3.suptitle(r'Density $\rho$ (z=0 slice)', fontsize=14)
fig3.tight_layout()
fig3.savefig(os.path.join(output_dir, 'Re100_density.png'), dpi=150, bbox_inches='tight')
plt.close(fig3)

# --- 图4: 压力 p 分布 ---
fig4, axes4 = plt.subplots(1, n_snap, figsize=(5*n_snap, 4.5))
for idx, t_val in enumerate(t_snapshots_full):
    p_p = all_preds[t_val]['p']
    ax = axes4[idx]
    c = ax.contourf(Xg, Yg, p_p, levels=20, cmap='coolwarm')
    ax.set_title(f't = {t_val:.1f}')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(c, ax=ax, format='%.2f')
fig4.suptitle(r'Pressure $p$ (z=0 slice)', fontsize=14)
fig4.tight_layout()
fig4.savefig(os.path.join(output_dir, 'Re100_pressure.png'), dpi=150, bbox_inches='tight')
plt.close(fig4)

# --- 图5: 速度分量 u, v, w ---
for comp_name, comp_idx in [('u', 'u'), ('v', 'v'), ('w', 'w')]:
    fig5, axes5 = plt.subplots(1, n_snap, figsize=(5*n_snap, 4.5))
    for idx, t_val in enumerate(t_snapshots_full):
        val = all_preds[t_val][comp_idx]
        ax = axes5[idx]
        vmax = max(abs(val.min()), abs(val.max()), 1e-6)
        c = ax.contourf(Xg, Yg, val, levels=20, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f't = {t_val:.1f}')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(c, ax=ax)
    fig5.suptitle(f'Velocity {comp_name} (z=0 slice)', fontsize=14)
    fig5.tight_layout()
    fig5.savefig(os.path.join(output_dir, f'Re100_velocity_{comp_name}.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig5)

# --- 图6: 涡量 ω_z = ∂v/∂x - ∂u/∂y (用有限差分近似) ---
fig6, axes6 = plt.subplots(1, n_snap, figsize=(5*n_snap, 4.5))
dx = x_lin[1] - x_lin[0]
dy = y_lin[1] - y_lin[0]
for idx, t_val in enumerate(t_snapshots_full):
    u_p = all_preds[t_val]['u']
    v_p = all_preds[t_val]['v']
    # 中心差分
    dvdx = np.gradient(v_p, dx, axis=0)
    dudy = np.gradient(u_p, dy, axis=1)
    omega_z = dvdx - dudy
    ax = axes6[idx]
    vmax = max(abs(omega_z.min()), abs(omega_z.max()), 1e-6)
    c = ax.contourf(Xg, Yg, omega_z, levels=20, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f't = {t_val:.1f}')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(c, ax=ax)
fig6.suptitle(r'Vorticity $\omega_z = \partial v/\partial x - \partial u/\partial y$ (z=0)', fontsize=14)
fig6.tight_layout()
fig6.savefig(os.path.join(output_dir, 'Re100_vorticity_z.png'), dpi=150, bbox_inches='tight')
plt.close(fig6)

# --- 图7: 速度大小 |V| + 密度对比 (t=0 理论 vs PINN) ---
fig7, axes7 = plt.subplots(2, 2, figsize=(12, 10))
# t=0 理论速度
xc0 = Xg / L; yc0 = Yg / L; zc0 = Zg / L
u_exact = V0 * np.sin(xc0) * np.cos(yc0) * np.cos(zc0)
v_exact = -V0 * np.cos(xc0) * np.sin(yc0) * np.cos(zc0)
vel_exact = np.sqrt(u_exact**2 + v_exact**2)
vel_pinn0 = np.sqrt(all_preds[0.0]['u']**2 + all_preds[0.0]['v']**2)

ax = axes7[0, 0]
c = ax.contourf(Xg, Yg, vel_exact, levels=20, cmap='jet')
ax.set_title('|V| Exact t=0'); ax.set_aspect('equal')
plt.colorbar(c, ax=ax)

ax = axes7[0, 1]
c = ax.contourf(Xg, Yg, vel_pinn0, levels=20, cmap='jet')
ax.set_title('|V| PINN t=0'); ax.set_aspect('equal')
plt.colorbar(c, ax=ax)

# t=0 理论压力
p_exact = p0 + (rho0 * V0**2 / 16.0) * (np.cos(2*xc0) + np.cos(2*yc0)) * (np.cos(2*zc0) + 2)
p_pinn0 = all_preds[0.0]['p']

ax = axes7[1, 0]
c = ax.contourf(Xg, Yg, p_exact, levels=20, cmap='coolwarm')
ax.set_title('p Exact t=0'); ax.set_aspect('equal')
plt.colorbar(c, ax=ax, format='%.2f')

ax = axes7[1, 1]
c = ax.contourf(Xg, Yg, p_pinn0, levels=20, cmap='coolwarm')
ax.set_title('p PINN t=0'); ax.set_aspect('equal')
plt.colorbar(c, ax=ax, format='%.2f')

fig7.suptitle('IC comparison: Exact vs PINN at t=0 (z=0 slice)', fontsize=14)
fig7.tight_layout()
fig7.savefig(os.path.join(output_dir, 'Re100_IC_comparison.png'), dpi=150, bbox_inches='tight')
plt.close(fig7)

# --- 图8: y=0 切面 (x-z 平面) 速度分布 ---
nz_eval = 60
z_lin = np.linspace(Zs, Ze, nz_eval)
Xxz, Zxz = np.meshgrid(x_lin, z_lin, indexing='ij')
Yxz = np.zeros_like(Xxz)  # y=0

fig8, axes8 = plt.subplots(1, n_snap, figsize=(5*n_snap, 4.5))
for idx, t_val in enumerate(t_snapshots_full):
    Txz = np.full_like(Xxz, t_val)
    x_test = np.stack([Txz, Xxz, Yxz, Zxz], axis=-1).reshape(-1, 4)
    x_test_t = torch.tensor(x_test, dtype=dtype).to(cuda)
    with torch.no_grad():
        pred = model(x_test_t)
    u_p = to_numpy(pred[:, 2]).reshape(nx_eval, nz_eval)
    w_p = to_numpy(pred[:, 4]).reshape(nx_eval, nz_eval)
    vel_xz = np.sqrt(u_p**2 + w_p**2)
    ax = axes8[idx]
    c = ax.contourf(Xxz, Zxz, vel_xz, levels=20, cmap='jet')
    ax.set_title(f't = {t_val:.1f}')
    ax.set_xlabel('x'); ax.set_ylabel('z')
    ax.set_aspect('equal')
    plt.colorbar(c, ax=ax, label='|V|')
fig8.suptitle('Velocity magnitude (y=0 slice, x-z plane)', fontsize=14)
fig8.tight_layout()
fig8.savefig(os.path.join(output_dir, 'Re100_xz_velocity.png'), dpi=150, bbox_inches='tight')
plt.close(fig8)

print('所有物理量分布图已保存')

# ========== 保存 + 打印结果 ==========
np.savez(os.path.join(output_dir, 'Re100_results.npz'),
         t=t_eval, Ek=Ek_arr, Ek_theory=Ek_theory,
         eps_pinn=eps_pinn, eps_theory=eps_theory,
         loss_history=np.array(loss_history))

# 保存模型
torch.save(model.state_dict(), os.path.join(output_dir, 'Re100_model.pt'))

print(f'\n========== 结果 ==========')
print(f'输出目录: {output_dir}')
print(f'总训练时间: {total_time:.1f}s')
print(f'最终 Loss: {loss_history[-1]:.4e}')
print(f'')
print(f'动能验证:')
print(f'  Ek(0) PINN   = {Ek_arr[0]:.6f}')
print(f'  Ek({Te}) PINN  = {Ek_arr[-1]:.6f}')
print(f'  Ek({Te}) theory= {Ek_theory[-1]:.6f}')
print(f'  Ek ratio PINN   = {Ek_arr[-1]/Ek_arr[0]:.6f}')
print(f'  Ek ratio theory = {np.exp(-2*nu*Te):.6f}')
rel_err = abs(Ek_arr[-1]/Ek_arr[0] - np.exp(-2*nu*Te)) / np.exp(-2*nu*Te) * 100
print(f'  相对误差 = {rel_err:.2f}%')
