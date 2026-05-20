# ========== 小规模测试：验证4卡流程 ==========
import sys
import os

current_file_dir = os.path.dirname(os.path.abspath(__file__))
pinn_src_path = os.path.join(current_file_dir, '..', '..', '..', 'PINNsrc')
pinn_src_path = os.path.abspath(pinn_src_path)
if pinn_src_path not in sys.path:
    sys.path.insert(0, pinn_src_path)

from PINNs import *
from utility import select_gpu, create_output_dir
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import math

dtype = torch.float64
setup_seed(42)

# ========== 物理参数 ==========
gamma = 1.4
L = 1.0
V0 = 1.0
rho0 = 1.0
Ma = 0.1
c0 = V0 / Ma
p0 = rho0 * c0**2 / gamma  # ≈ 71.4

pi = math.pi
Xs, Xe = -pi * L, pi * L
Ys, Ye = -pi * L, pi * L
Zs, Ze = -pi * L, pi * L
Ts, Te = 0.0, 20.0

Re = 1600.0
mu = rho0 * V0 * L / Re

print(f'=== 小规模测试 ===')
print(f'ρ0={rho0}, p0={p0:.2f}, Re={Re}, μ={mu:.6e}')

# ========== 多卡设置 ==========
num_gpus = min(4, torch.cuda.device_count())
gpu_ids = list(range(num_gpus))
torch.cuda.set_device(0)
print(f'使用 {num_gpus} 张GPU')

# ========== 初始条件 ==========
def IC_TGV(x):
    N = x.shape[0]
    xc, yc, zc = x[:, 1] / L, x[:, 2] / L, x[:, 3] / L
    u_init = V0 * np.sin(xc) * np.cos(yc) * np.cos(zc)
    v_init = -V0 * np.cos(xc) * np.sin(yc) * np.cos(zc)
    w_init = np.zeros(N)
    p_init = p0 + (rho0 * V0**2 / 16.0) * (np.cos(2*xc) + np.cos(2*yc)) * (np.cos(2*zc) + 2)
    rho_init = np.full(N, rho0)
    return rho_init, u_init, v_init, w_init, p_init

# ========== 小规模采样 ==========
from smt.sampling_methods import LHS

N_ic = 2000     # 小规模
N_int = 5000    # 小规模
Nbc = 500       # 小规模

xlimits_ic = np.array([[0., 0.], [Xs, Xe], [Ys, Ye], [Zs, Ze]])
x_ic = LHS(xlimits=xlimits_ic)(N_ic)
rho_ic, u_ic, v_ic, w_ic, p_ic = IC_TGV(x_ic)

xlimits_int = np.array([[Ts, Te], [Xs, Xe], [Ys, Ye], [Zs, Ze]])
x_int_np = LHS(xlimits=xlimits_int)(N_int)

t_bc = np.random.uniform(Ts, Te, Nbc)
y_bc = np.random.uniform(Ys, Ye, Nbc)
z_bc = np.random.uniform(Zs, Ze, Nbc)
x_bc_xL_np = np.stack([t_bc, np.full(Nbc, Xs), y_bc, z_bc], axis=1)
x_bc_xR_np = np.stack([t_bc, np.full(Nbc, Xe), y_bc, z_bc], axis=1)

t_bc2 = np.random.uniform(Ts, Te, Nbc)
x_bc2 = np.random.uniform(Xs, Xe, Nbc)
z_bc2 = np.random.uniform(Zs, Ze, Nbc)
x_bc_yL_np = np.stack([t_bc2, x_bc2, np.full(Nbc, Ys), z_bc2], axis=1)
x_bc_yR_np = np.stack([t_bc2, x_bc2, np.full(Nbc, Ye), z_bc2], axis=1)

t_bc3 = np.random.uniform(Ts, Te, Nbc)
x_bc3 = np.random.uniform(Xs, Xe, Nbc)
y_bc3 = np.random.uniform(Ys, Ye, Nbc)
x_bc_zL_np = np.stack([t_bc3, x_bc3, y_bc3, np.full(Nbc, Zs)], axis=1)
x_bc_zR_np = np.stack([t_bc3, x_bc3, y_bc3, np.full(Nbc, Ze)], axis=1)

print(f'采样：IC={N_ic}, 内部={N_int}, BC={Nbc*6}')

# ========== 数据分配到多GPU ==========
def split_to_gpus(arr_np, req_grad=False):
    if num_gpus == 1:
        return [torch.tensor(arr_np, requires_grad=req_grad, dtype=dtype).to('cuda:0')]
    n = len(arr_np)
    chunk_size = n // num_gpus
    chunks = []
    for i in range(num_gpus):
        start = i * chunk_size
        end = n if i == num_gpus - 1 else (i + 1) * chunk_size
        chunk = torch.tensor(arr_np[start:end], requires_grad=req_grad,
                           dtype=dtype).to(f'cuda:{i}')
        chunks.append(chunk)
    return chunks

x_ic_chunks = split_to_gpus(x_ic, True)
rho_ic_chunks = split_to_gpus(rho_ic, False)
u_ic_chunks = split_to_gpus(u_ic, False)
v_ic_chunks = split_to_gpus(v_ic, False)
w_ic_chunks = split_to_gpus(w_ic, False)
p_ic_chunks = split_to_gpus(p_ic, False)

x_int_chunks = split_to_gpus(x_int_np, True)

x_bc_xL_chunks = split_to_gpus(x_bc_xL_np, True)
x_bc_xR_chunks = split_to_gpus(x_bc_xR_np, True)
x_bc_yL_chunks = split_to_gpus(x_bc_yL_np, True)
x_bc_yR_chunks = split_to_gpus(x_bc_yR_np, True)
x_bc_zL_chunks = split_to_gpus(x_bc_zL_np, True)
x_bc_zR_chunks = split_to_gpus(x_bc_zR_np, True)

# ========== 模型 ==========
Nl = 4    # 小网络
Nn = 64   # 小网络
models = []
for i in range(num_gpus):
    model = PINNs_WE_NS_3D(Nl, Nn).to(f'cuda:{i}').to(dtype)
    models.append(model)

# 同步模型参数
if num_gpus > 1:
    with torch.no_grad():
        for i in range(1, num_gpus):
            for param0, parami in zip(models[0].parameters(), models[i].parameters()):
                parami.data.copy_(param0.data.to(f'cuda:{i}'))

print(f'模型：Nl={Nl}, Nn={Nn}, {num_gpus}副本')

# ========== 训练 ==========
optimizer = torch.optim.Adam(models[0].parameters(), lr=1e-3)

N_adam = 500  # 小规模
N_lbfgs = 50  # 小规模

def train_step(epoch):
    k = 0
    def closure():
        for m in models:
            for param in m.parameters():
                if param.grad is not None:
                    param.grad.zero_()

        loss_pde_total = 0.0
        loss_ic_total = 0.0
        loss_bc_total = 0.0

        for i in range(num_gpus):
            loss_pde_i = models[i].loss_pde(x_int_chunks[i], k=k, mu=mu, gamma=gamma,
                                            rho_ref=rho0, p_ref=p0, V_ref=V0)
            loss_ic_i = models[i].loss_ic(
                x_ic_chunks[i], rho_ic_chunks[i], u_ic_chunks[i],
                v_ic_chunks[i], w_ic_chunks[i], p_ic_chunks[i],
                rho_ref=rho0, p_ref=p0, V_ref=V0)

            y_xL = models[i](x_bc_xL_chunks[i])
            y_xR = models[i](x_bc_xR_chunks[i])
            y_yL = models[i](x_bc_yL_chunks[i])
            y_yR = models[i](x_bc_yR_chunks[i])
            y_zL = models[i](x_bc_zL_chunks[i])
            y_zR = models[i](x_bc_zR_chunks[i])

            loss_bc_i = (torch.mean((y_xL - y_xR)**2) +
                        torch.mean((y_yL - y_yR)**2) +
                        torch.mean((y_zL - y_zR)**2))

            loss_pde_total += loss_pde_i.to('cuda:0')
            loss_ic_total += loss_ic_i.to('cuda:0')
            loss_bc_total += loss_bc_i.to('cuda:0')

        loss_pde_avg = loss_pde_total / num_gpus
        loss_ic_avg = loss_ic_total / num_gpus
        loss_bc_avg = loss_bc_total / num_gpus

        loss = 1*loss_pde_avg + 10*loss_ic_avg + 10*loss_bc_avg

        if epoch % 100 == 0 or epoch == 1:
            print(f'  epoch {epoch} loss:{loss:.6e} pde:{loss_pde_avg:.6e} ic:{loss_ic_avg:.6e} bc:{loss_bc_avg:.6e}')

        loss.backward()

        if num_gpus > 1:
            with torch.no_grad():
                for params in zip(*[m.parameters() for m in models]):
                    grads = [param.grad.to('cuda:0') if param.grad is not None else None for param in params]
                    if all(g is not None for g in grads):
                        avg_grad = sum(grads) / num_gpus
                        for idx, param in enumerate(params):
                            param.grad = avg_grad.to(f'cuda:{idx}')

        return loss

    return optimizer.step(closure)

print(f'\n--- Adam {N_adam} steps ---')
t0 = time.time()
for epoch in range(1, N_adam + 1):
    train_step(epoch)
t1 = time.time()
print(f'Adam完成，用时 {t1-t0:.1f}s')

# ========== LBFGS阶段 ==========
print(f'\n--- LBFGS {N_lbfgs} steps ---')

# 同步参数到GPU 0
if num_gpus > 1:
    with torch.no_grad():
        for i in range(1, num_gpus):
            for param0, parami in zip(models[0].parameters(), models[i].parameters()):
                param0.data.add_(parami.data.to('cuda:0'))
        for param in models[0].parameters():
            param.data.div_(num_gpus)

# 合并数据到GPU 0
x_int_gpu0 = torch.cat([c.to('cuda:0') for c in x_int_chunks], dim=0)
x_ic_gpu0 = torch.cat([c.to('cuda:0') for c in x_ic_chunks], dim=0)
rho_ic_gpu0 = torch.cat([c.to('cuda:0') for c in rho_ic_chunks], dim=0)
u_ic_gpu0 = torch.cat([c.to('cuda:0') for c in u_ic_chunks], dim=0)
v_ic_gpu0 = torch.cat([c.to('cuda:0') for c in v_ic_chunks], dim=0)
w_ic_gpu0 = torch.cat([c.to('cuda:0') for c in w_ic_chunks], dim=0)
p_ic_gpu0 = torch.cat([c.to('cuda:0') for c in p_ic_chunks], dim=0)
x_bc_xL_gpu0 = torch.cat([c.to('cuda:0') for c in x_bc_xL_chunks], dim=0)
x_bc_xR_gpu0 = torch.cat([c.to('cuda:0') for c in x_bc_xR_chunks], dim=0)
x_bc_yL_gpu0 = torch.cat([c.to('cuda:0') for c in x_bc_yL_chunks], dim=0)
x_bc_yR_gpu0 = torch.cat([c.to('cuda:0') for c in x_bc_yR_chunks], dim=0)
x_bc_zL_gpu0 = torch.cat([c.to('cuda:0') for c in x_bc_zL_chunks], dim=0)
x_bc_zR_gpu0 = torch.cat([c.to('cuda:0') for c in x_bc_zR_chunks], dim=0)

optimizer_lbfgs = torch.optim.LBFGS(models[0].parameters(), max_iter=20)

def closure_lbfgs():
    optimizer_lbfgs.zero_grad()
    loss_pde = models[0].loss_pde(x_int_gpu0, k=0, mu=mu, gamma=gamma,
                                   rho_ref=rho0, p_ref=p0, V_ref=V0)
    loss_ic = models[0].loss_ic(x_ic_gpu0, rho_ic_gpu0, u_ic_gpu0,
                                 v_ic_gpu0, w_ic_gpu0, p_ic_gpu0,
                                 rho_ref=rho0, p_ref=p0, V_ref=V0)
    y_xL = models[0](x_bc_xL_gpu0)
    y_xR = models[0](x_bc_xR_gpu0)
    y_yL = models[0](x_bc_yL_gpu0)
    y_yR = models[0](x_bc_yR_gpu0)
    y_zL = models[0](x_bc_zL_gpu0)
    y_zR = models[0](x_bc_zR_gpu0)
    loss_bc = (torch.mean((y_xL - y_xR)**2) +
               torch.mean((y_yL - y_yR)**2) +
               torch.mean((y_zL - y_zR)**2))
    loss = 1*loss_pde + 10*loss_ic + 10*loss_bc
    loss.backward()
    return loss

t2 = time.time()
for epoch in range(1, N_lbfgs + 1):
    loss = optimizer_lbfgs.step(closure_lbfgs)
    if epoch % 10 == 0:
        print(f'  LBFGS epoch {epoch}, loss={loss:.6e}')
t3 = time.time()
print(f'LBFGS完成，用时 {t3-t2:.1f}s')

# ========== 快速验证：计算几个时刻的动能 ==========
print(f'\n--- 动能验证 ---')
models[0].eval()
t_check = [0, 5, 10, 20]
for t_val in t_check:
    nx = 16
    x_grid = np.linspace(Xs, Xe, nx)
    y_grid = np.linspace(Ys, Ye, nx)
    z_grid = np.linspace(Zs, Ze, nx)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    T = np.full_like(X, t_val)
    x_test = np.stack([T.ravel(), X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    x_test_t = torch.tensor(x_test, dtype=dtype).to('cuda:0')
    with torch.no_grad():
        y_pred = models[0](x_test_t).cpu().numpy()
    rho_pred = y_pred[:, 0]
    u_pred, v_pred, w_pred = y_pred[:, 2], y_pred[:, 3], y_pred[:, 4]
    U2 = u_pred**2 + v_pred**2 + w_pred**2
    Ek = np.mean(0.5 * rho_pred * U2) / rho0
    print(f'  t={t_val:5.1f}: Ek={Ek:.6f}, rho_mean={rho_pred.mean():.4f}, |u|_max={np.sqrt(U2).max():.4f}')

print(f'\n=== 流程验证完成！总用时 {time.time()-t0:.1f}s ===')
