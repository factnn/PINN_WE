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
num_gpus = 4  # 单卡验证
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

        loss = 1*loss_pde_avg + 10*loss_ic_avg + 10*loss_bc_avg

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
                    grads = [param.grad.to('cuda:0') if param.grad is not None else None for param in params]
                    if all(g is not None for g in grads):
                        avg_grad = sum(grads) / num_gpus
                        # 分发平均梯度到所有GPU
                        for idx, param in enumerate(params):
                            param.grad = avg_grad.to(f'cuda:{idx}')

        # 梯度裁剪防止爆炸
        torch.nn.utils.clip_grad_norm_(models[0].parameters(), max_norm=1.0)

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
rho_ic_chunks = split_to_gpus(rho_ic, False)
u_ic_chunks = split_to_gpus(u_ic, False)
v_ic_chunks = split_to_gpus(v_ic, False)
w_ic_chunks = split_to_gpus(w_ic, False)
p_ic_chunks = split_to_gpus(p_ic, False)

# 内部点分配到多GPU
x_int_chunks = split_to_gpus(x_int_np, True)

# BC点分配到多GPU
x_bc_xL_chunks = split_to_gpus(x_bc_xL_np, True)
x_bc_xR_chunks = split_to_gpus(x_bc_xR_np, True)
x_bc_yL_chunks = split_to_gpus(x_bc_yL_np, True)
x_bc_yR_chunks = split_to_gpus(x_bc_yR_np, True)
x_bc_zL_chunks = split_to_gpus(x_bc_zL_np, True)
x_bc_zR_chunks = split_to_gpus(x_bc_zR_np, True)

# ========== 创建多个模型副本 ==========
Nl = 6
Nn = 256
models = []
for i in range(num_gpus):
    model = PINNs_WE_NS_3D(Nl, Nn).to(f'cuda:{i}').to(dtype)
    models.append(model)

# 同步所有模型参数（从GPU 0复制到其他GPU）
if num_gpus > 1:
    with torch.no_grad():
        for i in range(1, num_gpus):
            for param0, parami in zip(models[0].parameters(), models[i].parameters()):
                parami.data.copy_(param0.data.to(f'cuda:{i}'))

print(f'模型创建完成：{num_gpus}个副本，每个{Nn}层×64神经元')

# ========== 优化器（只对GPU 0的模型）==========
optimizer = torch.optim.Adam(models[0].parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)

# ========== 训练 ==========
print('开始Adam训练...')
t0 = time.time()
best_loss = float('inf')
best_state = None
for epoch in range(1, 50001):
    loss = train(epoch, 'Adam', 50000)
    scheduler.step()
    # 保存最优模型
    loss_val = loss.item() if hasattr(loss, 'item') else loss
    if not math.isnan(loss_val) and loss_val < best_loss:
        best_loss = loss_val
        best_state = {k: v.clone() for k, v in models[0].state_dict().items()}
t1 = time.time()
print(f'Adam完成，用时 {t1-t0:.2f}s，最佳loss={best_loss:.6e}')

# 恢复最优参数
if best_state is not None:
    models[0].load_state_dict(best_state)
    print('已恢复Adam阶段最优参数')

# LBFGS阶段：多卡计算梯度，GPU 0更新参数
print('同步参数准备LBFGS...')
# 先把Adam阶段多卡的参数平均到GPU 0
if num_gpus > 1:
    with torch.no_grad():
        for i in range(1, num_gpus):
            for param0, parami in zip(models[0].parameters(), models[i].parameters()):
                param0.data.add_(parami.data.to('cuda:0'))
        for param in models[0].parameters():
            param.data.div_(num_gpus)

optimizer_lbfgs = torch.optim.LBFGS(models[0].parameters(), max_iter=20)

def closure_lbfgs():
    optimizer_lbfgs.zero_grad()

    # 先同步GPU 0的最新参数到其他GPU
    if num_gpus > 1:
        with torch.no_grad():
            for i in range(1, num_gpus):
                for param0, parami in zip(models[0].parameters(), models[i].parameters()):
                    parami.data.copy_(param0.data.to(f'cuda:{i}'))

    # 多卡并行计算loss
    loss_pde_total = 0.0
    loss_ic_total = 0.0
    loss_bc_total = 0.0

    for i in range(num_gpus):
        loss_pde_i = models[i].loss_pde(x_int_chunks[i], k=0, mu=mu, gamma=gamma,
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
    loss.backward()

    # 汇总各卡梯度到GPU 0
    if num_gpus > 1:
        with torch.no_grad():
            for params in zip(*[m.parameters() for m in models]):
                grads = [param.grad.to('cuda:0') if param.grad is not None else None for param in params]
                if all(g is not None for g in grads):
                    avg_grad = sum(grads) / num_gpus
                    params[0].grad = avg_grad  # 只需要GPU 0的梯度给LBFGS

    return loss

print('开始LBFGS训练...')
t2 = time.time()
for epoch in range(1, 1001):
    loss = optimizer_lbfgs.step(closure_lbfgs)
    if epoch % 100 == 0:
        print(f'LBFGS epoch {epoch}, loss={loss:.6e}')
t3 = time.time()
print(f'LBFGS完成，用时 {t3-t2:.2f}s')

# ========== 保存模型 ==========
output_dir = create_output_dir('TGV_3D')
torch.save(models[0].state_dict(), f'{output_dir}/model.pth')
print(f'模型已保存到 {output_dir}/model.pth')

# ========== 可视化 ==========
print('生成可视化...')
models[0].eval()

# 计算动能演化
t_eval = np.linspace(Ts, Te, 100)
Ek_list = []
for t_val in t_eval:
    nx, ny, nz = 32, 32, 32
    x_grid = np.linspace(Xs, Xe, nx)
    y_grid = np.linspace(Ys, Ye, ny)
    z_grid = np.linspace(Zs, Ze, nz)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    T = np.full_like(X, t_val)
    x_test = np.stack([T.ravel(), X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    x_test_t = torch.tensor(x_test, dtype=dtype).to('cuda:0')

    with torch.no_grad():
        y_pred = models[0](x_test_t).cpu().numpy()

    rho_pred = y_pred[:, 0]
    u_pred = y_pred[:, 2]
    v_pred = y_pred[:, 3]
    w_pred = y_pred[:, 4]

    U2 = u_pred**2 + v_pred**2 + w_pred**2
    Ek = np.mean(0.5 * rho_pred * U2) / rho0
    Ek_list.append(Ek)

Ek_list = np.array(Ek_list)

# 理论值（无粘性应保持常数）
Ek_theory = np.full_like(t_eval, Ek_list[0])

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 线性尺度
axes[0, 0].plot(t_eval, Ek_list, 'b-', linewidth=2, label='PINN')
axes[0, 0].plot(t_eval, Ek_theory, 'r--', linewidth=2, label='Theory (inviscid)')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Kinetic Energy')
axes[0, 0].set_title('Energy Evolution (Linear Scale)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 对数尺度
axes[0, 1].semilogy(t_eval, Ek_list, 'b-', linewidth=2, label='PINN')
axes[0, 1].semilogy(t_eval, Ek_theory, 'r--', linewidth=2, label='Theory')
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Kinetic Energy (log)')
axes[0, 1].set_title('Energy Evolution (Log Scale)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 归一化能量
Ek_norm = Ek_list / Ek_list[0]
axes[1, 0].plot(t_eval, Ek_norm, 'b-', linewidth=2)
axes[1, 0].axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Theory')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('E(t) / E(0)')
axes[1, 0].set_title('Normalized Energy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 能量保持率
retention = Ek_norm * 100
axes[1, 1].plot(t_eval, retention, 'b-', linewidth=2)
axes[1, 1].axhline(y=100, color='r', linestyle='--', linewidth=2, label='100%')
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Energy Retention (%)')
axes[1, 1].set_title('Energy Conservation')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/energy_evolution.png', dpi=150, bbox_inches='tight')
print(f'能量演化图已保存到 {output_dir}/energy_evolution.png')

print(f'\n最终能量保持率：')
print(f't=0: {retention[0]:.2f}%')
print(f't=5: {retention[len(t_eval)//4]:.2f}%')
print(f't=10: {retention[len(t_eval)//2]:.2f}%')
print(f't=20: {retention[-1]:.2f}%')
print('训练完成！')
