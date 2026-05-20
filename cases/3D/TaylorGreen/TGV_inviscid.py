# ========== 路径设置（必须放在最前面）==========
import sys
import os

current_file_dir = os.path.dirname(os.path.abspath(__file__))
pinn_src_path = os.path.join(current_file_dir, '..', '..', '..', 'PINNsrc')
pinn_src_path = os.path.abspath(pinn_src_path)
if pinn_src_path not in sys.path:
    sys.path.insert(0, pinn_src_path)

# ========== 导入模块 ==========
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
# Taylor-Green Vortex: 周期性立方体域 [-πL, πL]³
# 参数设置参考论文，但算无粘Euler方程
gamma = 1.4
L = 1.0
V0 = 1.0
rho0 = 1.0  # 和论文一致
Ma = 0.1
c0 = V0 / Ma  # 声速 = 10
p0 = rho0 * c0**2 / gamma  # 从状态方程得到 p0 ≈ 71.4

# 计算域
pi = math.pi
Xs, Xe = -pi * L, pi * L
Ys, Ye = -pi * L, pi * L
Zs, Ze = -pi * L, pi * L
Ts, Te = 0.0, 20.0  # 和论文一致

# Reynolds number (控制粘性耗散)
Re = 1600.0
mu = rho0 * V0 * L / Re  # 动力粘度

print(f'Taylor-Green Vortex 3D')
print(f'γ={gamma}, L={L}, V0={V0}, Ma={Ma}')
print(f'ρ0={rho0:.6f}, p0={p0}, c0={c0:.2f}')
print(f'Re={Re}, μ={mu:.6e}')
print(f'域: [{Xs:.4f}, {Xe:.4f}]³')
print(f'时间: [{Ts}, {Te}]')

# ========== 多卡设置 ==========
num_gpus = 1  # 单卡验证
available_gpus = torch.cuda.device_count()
if num_gpus > available_gpus:
    print(f'警告：请求{num_gpus}张卡，但只有{available_gpus}张可用，使用{available_gpus}张')
    num_gpus = available_gpus

if num_gpus > 0:
    gpu_ids = list(range(num_gpus))
    torch.cuda.set_device(gpu_ids[0])
    cuda = torch.device('cuda')
    print(f'使用 {num_gpus} 张GPU: {gpu_ids}')
    for i in gpu_ids:
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    cuda = torch.device('cpu')
    print('使用 CPU')

# ========== 初始条件：Taylor-Green Vortex ==========
def IC_TGV(x):
    """
    Taylor-Green Vortex 初始条件
    x: (N, 4) -> [t, x, y, z]
    U =  V0 * sin(x/L) * cos(y/L) * cos(z/L)
    V = -V0 * cos(x/L) * sin(y/L) * cos(z/L)
    W = 0
    p = p0 + (ρ0*V0²/16) * (cos(2x/L) + cos(2y/L)) * (cos(2z/L) + 2)
    ρ = ρ0 (均匀温度，低马赫数)
    """
    N = x.shape[0]
    xc = x[:, 1] / L
    yc = x[:, 2] / L
    zc = x[:, 3] / L

    u_init = V0 * np.sin(xc) * np.cos(yc) * np.cos(zc)
    v_init = -V0 * np.cos(xc) * np.sin(yc) * np.cos(zc)
    w_init = np.zeros(N)
    p_init = p0 + (rho0 * V0**2 / 16.0) * (
        np.cos(2*xc) + np.cos(2*yc)) * (np.cos(2*zc) + 2)
    rho_init = np.full(N, rho0)

    return rho_init, u_init, v_init, w_init, p_init


# ========== 训练函数（多卡版本）==========
def train(epoch, stage='Adam', total_epochs=None):
    k = 0  # 标准PINN，无权重函数

    def closure():
        # 清零所有模型的梯度
        for m in models:
            for param in m.parameters():
                if param.grad is not None:
                    param.grad.zero_()

        # 在每个GPU上并行计算loss
        loss_pde_total = 0.0
        loss_ic_total = 0.0
        loss_bc_total = 0.0

        for i in range(num_gpus):
            # 每个GPU计算自己的loss
            loss_pde_i = models[i].loss_pde(x_int_chunks[i], k=k, mu=mu, gamma=gamma,
                                            rho_ref=rho0, p_ref=p0, V_ref=V0)
            loss_ic_i = models[i].loss_ic(
                x_ic_chunks[i], rho_ic_chunks[i], u_ic_chunks[i],
                v_ic_chunks[i], w_ic_chunks[i], p_ic_chunks[i],
                rho_ref=rho0, p_ref=p0, V_ref=V0)

            # BC loss
            y_xL = models[i](x_bc_xL_chunks[i])
            y_xR = models[i](x_bc_xR_chunks[i])
            y_yL = models[i](x_bc_yL_chunks[i])
            y_yR = models[i](x_bc_yR_chunks[i])
            y_zL = models[i](x_bc_zL_chunks[i])
            y_zR = models[i](x_bc_zR_chunks[i])

            loss_bc_i = (torch.mean((y_xL - y_xR)**2) +
                        torch.mean((y_yL - y_yR)**2) +
                        torch.mean((y_zL - y_zR)**2))

            # 累加loss（转到GPU 0）
            loss_pde_total += loss_pde_i.to('cuda:0')
            loss_ic_total += loss_ic_i.to('cuda:0')
            loss_bc_total += loss_bc_i.to('cuda:0')

        # 平均loss
        loss_pde_avg = loss_pde_total / num_gpus
        loss_ic_avg = loss_ic_total / num_gpus
        loss_bc_avg = loss_bc_total / num_gpus

        loss = 10*loss_pde_avg + 10*loss_ic_avg + 10*loss_bc_avg

        if epoch % 5000 == 0 or epoch == 1:
            print(f'epoch {epoch} loss:{loss:.6e} '
                  f'pde:{loss_pde_avg:.6e} '
                  f'ic:{loss_ic_avg:.6e} '
                  f'bc:{loss_bc_avg:.6e}')

        # 反向传播
        loss.backward()

        # 同步梯度：将所有GPU的梯度平均后同步
        if num_gpus > 1:
            with torch.no_grad():
                for params in zip(*[m.parameters() for m in models]):
                    # 收集所有GPU的梯度
                    grads = [p.grad.to('cuda:0') if p.grad is not None else None for p in params]
                    if all(g is not None for g in grads):
                        avg_grad = sum(grads) / num_gpus
                        # 分发平均梯度到所有GPU
                        for i, p in enumerate(params):
                            p.grad = avg_grad.to(f'cuda:{i}')

        return loss

    loss = optimizer.step(closure)
    return loss


# ========== 采样点设置 ==========
from smt.sampling_methods import LHS

# 初始条件采样 (t=0)
xlimits_ic = np.array([
    [0., 0.], [Xs, Xe], [Ys, Ye], [Zs, Ze]])
sampling_ic = LHS(xlimits=xlimits_ic)
x_ic = sampling_ic(32000)  # 增大到32000

rho_ic, u_ic, v_ic, w_ic, p_ic = IC_TGV(x_ic)

# 内部PDE残差点采样
xlimits_int = np.array([
    [Ts, Te], [Xs, Xe], [Ys, Ye], [Zs, Ze]])
sampling_int = LHS(xlimits=xlimits_int)
x_int_np = sampling_int(80000)  # 增大到80000

# 周期边界条件采样 (3对面)
Nbc = 8000  # 增大到8000
t_bc = np.random.uniform(Ts, Te, Nbc)
y_bc = np.random.uniform(Ys, Ye, Nbc)
z_bc = np.random.uniform(Zs, Ze, Nbc)

# x方向: x=Xs vs x=Xe
x_bc_xL_np = np.stack([t_bc, np.full(Nbc, Xs), y_bc, z_bc], axis=1)
x_bc_xR_np = np.stack([t_bc, np.full(Nbc, Xe), y_bc, z_bc], axis=1)

# y方向: y=Ys vs y=Ye
t_bc2 = np.random.uniform(Ts, Te, Nbc)
x_bc2 = np.random.uniform(Xs, Xe, Nbc)
z_bc2 = np.random.uniform(Zs, Ze, Nbc)
x_bc_yL_np = np.stack([t_bc2, x_bc2, np.full(Nbc, Ys), z_bc2], axis=1)
x_bc_yR_np = np.stack([t_bc2, x_bc2, np.full(Nbc, Ye), z_bc2], axis=1)

# z方向: z=Zs vs z=Ze
t_bc3 = np.random.uniform(Ts, Te, Nbc)
x_bc3 = np.random.uniform(Xs, Xe, Nbc)
y_bc3 = np.random.uniform(Ys, Ye, Nbc)
x_bc_zL_np = np.stack([t_bc3, x_bc3, y_bc3, np.full(Nbc, Zs)], axis=1)
x_bc_zR_np = np.stack([t_bc3, x_bc3, y_bc3, np.full(Nbc, Ze)], axis=1)

print(f'采样完成：IC={len(x_ic)}, 内部={len(x_int_np)}, BC={Nbc*6}')

# ========== 转换为Tensor（多卡版本：数据分batch）==========
def to_tensor(arr, req_grad=False):
    return torch.tensor(arr, requires_grad=req_grad,
                        dtype=dtype).to(cuda)

# 将数据分成num_gpus份，分配到不同GPU
def split_to_gpus(arr_np, req_grad=False):
    """将numpy数组分成num_gpus份，每份放到对应GPU上"""
    if num_gpus == 1:
        return [to_tensor(arr_np, req_grad)]

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

# IC数据分配到多GPU
x_ic_chunks = split_to_gpus(x_ic, True)
rho_ic_chunks = [torch.tensor(rho_ic[i*len(rho_ic)//num_gpus:(i+1)*len(rho_ic)//num_gpus if i<num_gpus-1 else len(rho_ic)],
                              dtype=dtype).to(f'cuda:{i}') for i in range(num_gpus)]
u_ic_chunks = [torch.tensor(u_ic[i*len(u_ic)//num_gpus:(i+1)*len(u_ic)//num_gpus if i<num_gpus-1 else len(u_ic)],
                            dtype=dtype).to(f'cuda:{i}') for i in range(num_gpus)]
v_ic_chunks = [torch.tensor(v_ic[i*len(v_ic)//num_gpus:(i+1)*len(v_ic)//num_gpus if i<num_gpus-1 else len(v_ic)],
                            dtype=dtype).to(f'cuda:{i}') for i in range(num_gpus)]
w_ic_chunks = [torch.tensor(w_ic[i*len(w_ic)//num_gpus:(i+1)*len(w_ic)//num_gpus if i<num_gpus-1 else len(w_ic)],
                            dtype=dtype).to(f'cuda:{i}') for i in range(num_gpus)]
p_ic_chunks = [torch.tensor(p_ic[i*len(p_ic)//num_gpus:(i+1)*len(p_ic)//num_gpus if i<num_gpus-1 else len(p_ic)],
                            dtype=dtype).to(f'cuda:{i}') for i in range(num_gpus)]

# 内部点分配到多GPU
x_int_chunks = split_to_gpus(x_int_np, True)

# BC数据分配到多GPU
x_bc_xL_chunks = split_to_gpus(x_bc_xL_np, True)
x_bc_xR_chunks = split_to_gpus(x_bc_xR_np, True)
x_bc_yL_chunks = split_to_gpus(x_bc_yL_np, True)
x_bc_yR_chunks = split_to_gpus(x_bc_yR_np, True)
x_bc_zL_chunks = split_to_gpus(x_bc_zL_np, True)
x_bc_zR_chunks = split_to_gpus(x_bc_zR_np, True)

print(f'数据已分配到 {num_gpus} 张GPU')

# ========== 网络结构（多卡版本）==========
# 为每个GPU创建独立的模型副本
models = []
for i in range(num_gpus):
    m = PINNs_WE_NS_3D(Nl=6, Nn=256).to(f'cuda:{i}').double()  # 增大到Nn=256
    models.append(m)

# 同步所有模型的参数（初始化相同）
if num_gpus > 1:
    with torch.no_grad():
        for i in range(1, num_gpus):
            for p_main, p_i in zip(models[0].parameters(), models[i].parameters()):
                p_i.data.copy_(p_main.data)
    print(f'已创建 {num_gpus} 个模型副本，参数已同步')

print(f'网络参数量: {sum(p.numel() for p in models[0].parameters())}')
print('Start training...')

epoch = 0
epochi = epoch
loss_history = []
tic_total = time.time()

# ========== 创建输出目录 ==========
base_output_dir = create_output_dir(
    'TGV3D', gpu_count=num_gpus, tag='Re1600_Ma01_Euler')
print(f'[输出目录] {base_output_dir}')

# ========== Adam阶段 ==========
print('========== Adam阶段 ==========')
lr = 0.001
# 收集所有GPU模型的参数
all_params = []
for m in models:
    all_params.extend(m.parameters())
optimizer = torch.optim.Adam(all_params, lr=lr)
epochs = 50000
tic = time.time()

for epoch in range(1+epochi, epochs+epochi):
    loss = train(epoch, stage='Adam')
    loss_history.append(to_numpy(loss))

    if loss < 0.001:
        print(f'Loss < 0.001，提前停止')
        break

toc = time.time()
print(f'Adam阶段完成，用时: {toc - tic:.2f}秒')
epochi = epoch

# ========== LBFGS阶段 ==========
print('========== LBFGS阶段 ==========')
# LBFGS只使用GPU 0的模型（LBFGS要求所有参数在同一设备）
# 先将其他GPU的参数同步到GPU 0
if num_gpus > 1:
    print('同步所有GPU参数到GPU 0...')
    with torch.no_grad():
        for p0, *other_ps in zip(*[m.parameters() for m in models]):
            # 平均所有GPU的参数
            avg_param = sum([p.data.to('cuda:0') for p in [p0] + other_ps]) / num_gpus
            p0.data.copy_(avg_param)

optimizer = torch.optim.LBFGS(
    models[0].parameters(), lr=1.0, max_iter=30)
epochs_lbfgs = 1500
tic_lbfgs = time.time()

# LBFGS阶段使用单GPU训练
def train_lbfgs(epoch):
    k = 0
    def closure():
        optimizer.zero_grad()
        # 只在GPU 0上计算
        loss_pde = models[0].loss_pde(x_int_chunks[0], k=k, mu=mu, gamma=gamma,
                                      rho_ref=rho0, p_ref=p0, V_ref=V0)
        loss_ic = models[0].loss_ic(
            x_ic_chunks[0], rho_ic_chunks[0], u_ic_chunks[0],
            v_ic_chunks[0], w_ic_chunks[0], p_ic_chunks[0],
            rho_ref=rho0, p_ref=p0, V_ref=V0)

        y_xL = models[0](x_bc_xL_chunks[0])
        y_xR = models[0](x_bc_xR_chunks[0])
        y_yL = models[0](x_bc_yL_chunks[0])
        y_yR = models[0](x_bc_yR_chunks[0])
        y_zL = models[0](x_bc_zL_chunks[0])
        y_zR = models[0](x_bc_zR_chunks[0])

        loss_bc = (torch.mean((y_xL - y_xR)**2) +
                  torch.mean((y_yL - y_yR)**2) +
                  torch.mean((y_zL - y_zR)**2))

        loss = 10*loss_pde + 10*loss_ic + 10*loss_bc

        if epoch % 100 == 0:
            print(f'LBFGS Epoch {epoch}, Loss: {loss:.8f}')
        loss.backward()
        return loss

    return optimizer.step(closure)

for epoch in range(1+epochi, epochs_lbfgs+epochi):
    loss = train_lbfgs(epoch)
    loss_history.append(to_numpy(loss))

toc_lbfgs = time.time()
print(f'LBFGS阶段完成，用时: {toc_lbfgs - tic_lbfgs:.2f}秒')
training_time = time.time() - tic_total
print(f'总训练时间: {training_time:.2f}秒')

# ========== 预测：不同时刻的流场 ==========
print('开始预测...')
# 使用GPU 0的模型进行预测
actual_model = models[0]
actual_model.eval()

# z=0 切面的评估网格
nx_eval = 50
x_lin = np.linspace(Xs, Xe, nx_eval)
y_lin = np.linspace(Ys, Ye, nx_eval)
Xg, Yg = np.meshgrid(x_lin, y_lin, indexing='ij')
Zg = np.zeros_like(Xg)  # z=0 切面

# 图1：不同时刻的速度场 (z=0切面)
t_snapshots = [0.0, 2.0, 5.0, 10.0]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, t_val in enumerate(t_snapshots):
    Tg = np.full_like(Xg, t_val)
    x_test_np = np.stack(
        [Tg, Xg, Yg, Zg], axis=-1).reshape(-1, 4)
    x_test = torch.tensor(
        x_test_np, dtype=dtype).to(cuda)

    with torch.no_grad():
        pred = actual_model(x_test)
    u_p = to_numpy(pred[:, 2]).reshape(nx_eval, nx_eval)
    v_p = to_numpy(pred[:, 3]).reshape(nx_eval, nx_eval)
    w_p = to_numpy(pred[:, 4]).reshape(nx_eval, nx_eval)
    vel_mag = np.sqrt(u_p**2 + v_p**2 + w_p**2)

    ax = axes[idx]
    c = ax.contourf(Xg, Yg, vel_mag, levels=20,
                    cmap='jet')
    # 稀疏quiver
    skip = 5
    ax.quiver(Xg[::skip, ::skip], Yg[::skip, ::skip],
              u_p[::skip, ::skip], v_p[::skip, ::skip],
              color='k', alpha=0.5, scale=15)
    ax.set_title(f't = {t_val:.1f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(c, ax=ax, label='|V|')

plt.suptitle('Taylor-Green Vortex: Velocity Field (z=0 slice)',
             fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(base_output_dir, 'flow_snapshots.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print('流场快照图已保存')

# ========== 图2：动能随时间变化 ==========
print('计算动能演化...')
Nt_eval = 100
t_eval_arr = np.linspace(Ts, Te, Nt_eval)
Ek_arr = np.zeros(Nt_eval)

# 3D采样网格用于体积积分
nk = 20
xk = np.linspace(Xs, Xe, nk)
yk = np.linspace(Ys, Ye, nk)
zk = np.linspace(Zs, Ze, nk)
Xk, Yk, Zk = np.meshgrid(xk, yk, zk, indexing='ij')
Xk_flat = Xk.flatten()
Yk_flat = Yk.flatten()
Zk_flat = Zk.flatten()
Omega = (Xe - Xs) * (Ye - Ys) * (Ze - Zs)  # 域体积

for i, t_val in enumerate(t_eval_arr):
    Tk_flat = np.full_like(Xk_flat, t_val)
    x_k = np.stack([Tk_flat, Xk_flat,
                     Yk_flat, Zk_flat], axis=1)
    x_k_t = torch.tensor(
        x_k, dtype=dtype).to(cuda)

    with torch.no_grad():
        pred_k = actual_model(x_k_t)
    rho_k = to_numpy(pred_k[:, 0])
    u_k = to_numpy(pred_k[:, 2])
    v_k = to_numpy(pred_k[:, 3])
    w_k = to_numpy(pred_k[:, 4])

    # Ek = (1/(2*ρ0*Ω)) * ∫ ρ*(u²+v²+w²) dΩ
    # 用均匀网格的均值近似体积积分
    ke = rho_k * (u_k**2 + v_k**2 + w_k**2)
    Ek_arr[i] = np.mean(ke) / (2.0 * rho0)

# 耗散率 ε = -dEk/dt（中心差分）
dt_eval = t_eval_arr[1] - t_eval_arr[0]
eps_arr = np.zeros(Nt_eval)
eps_arr[1:-1] = -(Ek_arr[2:] - Ek_arr[:-2]) / (2*dt_eval)
eps_arr[0] = -(Ek_arr[1] - Ek_arr[0]) / dt_eval
eps_arr[-1] = -(Ek_arr[-1] - Ek_arr[-2]) / dt_eval

# 绘制动能和耗散率
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 理论参考：无粘Euler方程动能应该守恒
Ek_theory = np.full_like(t_eval_arr, Ek_arr[0])

# 左上：Ek 线性坐标
axes[0, 0].plot(t_eval_arr, Ek_arr, 'b-', lw=2, label='PINN')
axes[0, 0].plot(t_eval_arr, Ek_theory, 'r--', lw=2, label='Theory (conserved)')
axes[0, 0].set_xlabel('t')
axes[0, 0].set_ylabel('Ek')
axes[0, 0].set_title('Kinetic Energy (linear)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 右上：Ek 对数坐标
axes[0, 1].semilogy(t_eval_arr, Ek_arr, 'b-', lw=2, label='PINN')
axes[0, 1].semilogy(t_eval_arr, Ek_theory, 'r--', lw=2, label='Theory (conserved)')
axes[0, 1].set_xlabel('t')
axes[0, 1].set_ylabel('Ek')
axes[0, 1].set_title('Kinetic Energy (log scale)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 左下：耗散率
axes[1, 0].plot(t_eval_arr, eps_arr, 'r-', lw=2)
axes[1, 0].set_xlabel('t')
axes[1, 0].set_ylabel('ε(Ek)')
axes[1, 0].set_title('Dissipation Rate ε = -dEk/dt')
axes[1, 0].grid(True, alpha=0.3)

# 右下：耗散率对数坐标
eps_pos = np.clip(eps_arr, 1e-15, None)
axes[1, 1].semilogy(t_eval_arr, eps_pos, 'r-', lw=2)
axes[1, 1].set_xlabel('t')
axes[1, 1].set_ylabel('ε(Ek)')
axes[1, 1].set_title('Dissipation Rate (log scale)')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Taylor-Green Vortex: Energy Evolution',
             fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(base_output_dir,
            'energy_evolution.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print('动能演化图已保存')

# ========== 图3：Loss History ==========
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(loss_history, 'b-', lw=0.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss History')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(base_output_dir,
            'loss_history.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print('Loss曲线图已保存')

# ========== 保存数据 ==========
np.savez(os.path.join(base_output_dir, 'results.npz'),
         t=t_eval_arr, Ek=Ek_arr, eps=eps_arr,
         loss_history=np.array(loss_history))

print(f'\n========== 完成 ==========')
print(f'输出目录: {base_output_dir}')
print(f'总训练时间: {training_time:.2f}秒')
print(f'最终Loss: {loss_history[-1]:.6e}')
print(f'初始动能: {Ek_arr[0]:.6f}')
print(f'终止动能: {Ek_arr[-1]:.6f}')
