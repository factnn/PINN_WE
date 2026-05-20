"""
实际验证: FastLaplacian vs Baseline Autograd 训练对比

问题: 3D NS 方程简化版 (Burgers 3D with viscosity)
    u_t + u·∇u = μ∇²u

用相同的网络、数据、超参, 只替换二阶导计算方式:
1. Baseline: 标准 nested autograd
2. FastLaplacian: FD batched (h=0.1)
3. FastLaplacian adaptive: h=0.3→0.05

比较: 收敛曲线 (epoch), 收敛曲线 (wall time), 精度
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_laplacian import FastLaplacian, AdaptiveFDLaplacian

torch.manual_seed(42)
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
print(f'Device: {cuda}\n')

# ========== 通用工具 ==========
def gradients(out, inp):
    return torch.autograd.grad(out, inp, grad_outputs=torch.ones_like(out), create_graph=True)

def build_net(Nl=6, Nn=128):
    layers = [nn.Linear(4, Nn), nn.Tanh()]
    for _ in range(Nl - 2):
        layers += [nn.Linear(Nn, Nn), nn.Tanh()]
    layers += [nn.Linear(Nn, 3)]  # (u, v, w)
    return nn.Sequential(*layers)


# ========== 问题定义: 3D Burgers with viscosity ==========
# u_t + u*u_x + v*u_y + w*u_z = mu * (u_xx + u_yy + u_zz)
# v_t + u*v_x + v*v_y + w*v_z = mu * (v_xx + v_yy + v_zz)
# w_t + u*w_x + v*w_y + w*w_z = mu * (w_xx + w_yy + w_zz)
#
# IC: u(0,x,y,z) = sin(x)*cos(y)*cos(z), v = -cos(x)*sin(y)*cos(z), w = 0
# Domain: [0,1] × [-π,π]³, periodic BC

mu = 0.01
pi = np.pi


def exact_ic(x):
    """初始条件"""
    u = torch.sin(x[:, 1]) * torch.cos(x[:, 2]) * torch.cos(x[:, 3])
    v = -torch.cos(x[:, 1]) * torch.sin(x[:, 2]) * torch.cos(x[:, 3])
    w = torch.zeros_like(u)
    return torch.stack([u, v, w], dim=1)


# ========== 训练方法 ==========

def loss_baseline(net, x_int, x_ic, ic_true):
    """Baseline: nested autograd"""
    # PDE loss
    y = net(x_int)
    u, v, w = y[:, 0:1], y[:, 1:2], y[:, 2:3]

    # 一阶导
    du = gradients(u, x_int)[0]
    u_t, u_x, u_y, u_z = du[:, 0:1], du[:, 1:2], du[:, 2:3], du[:, 3:4]
    dv = gradients(v, x_int)[0]
    v_t, v_x, v_y, v_z = dv[:, 0:1], dv[:, 1:2], dv[:, 2:3], dv[:, 3:4]
    dw = gradients(w, x_int)[0]
    w_t, w_x, w_y, w_z = dw[:, 0:1], dw[:, 1:2], dw[:, 2:3], dw[:, 3:4]

    # 二阶导 (nested autograd!)
    u_xx = gradients(u_x, x_int)[0][:, 1:2]
    u_yy = gradients(u_y, x_int)[0][:, 2:3]
    u_zz = gradients(u_z, x_int)[0][:, 3:4]
    v_xx = gradients(v_x, x_int)[0][:, 1:2]
    v_yy = gradients(v_y, x_int)[0][:, 2:3]
    v_zz = gradients(v_z, x_int)[0][:, 3:4]
    w_xx = gradients(w_x, x_int)[0][:, 1:2]
    w_yy = gradients(w_y, x_int)[0][:, 2:3]
    w_zz = gradients(w_z, x_int)[0][:, 3:4]

    res_u = u_t + u*u_x + v*u_y + w*u_z - mu*(u_xx + u_yy + u_zz)
    res_v = v_t + u*v_x + v*v_y + w*v_z - mu*(v_xx + v_yy + v_zz)
    res_w = w_t + u*w_x + v*w_y + w*w_z - mu*(w_xx + w_yy + w_zz)

    loss_pde = (res_u**2).mean() + (res_v**2).mean() + (res_w**2).mean()

    # IC loss
    y_ic = net(x_ic)
    loss_ic = ((y_ic - ic_true)**2).mean()

    return loss_pde + 10 * loss_ic, loss_pde.item(), loss_ic.item()


def loss_fast(net, x_int, x_ic, ic_true, fast_lap):
    """FastLaplacian: FD for second derivatives"""
    y = net(x_int)
    u, v, w = y[:, 0:1], y[:, 1:2], y[:, 2:3]

    # 一阶导 (还是用autograd, 但不需要create_graph因为不再嵌套)
    # 注意: 一阶导仍然需要 create_graph=True 因为 u*u_x 等乘积项需要反向传播
    du = gradients(u, x_int)[0]
    u_t, u_x, u_y, u_z = du[:, 0:1], du[:, 1:2], du[:, 2:3], du[:, 3:4]
    dv = gradients(v, x_int)[0]
    v_t, v_x, v_y, v_z = dv[:, 0:1], dv[:, 1:2], dv[:, 2:3], dv[:, 3:4]
    dw = gradients(w, x_int)[0]
    w_t, w_x, w_y, w_z = dw[:, 0:1], dw[:, 1:2], dw[:, 2:3], dw[:, 3:4]

    # 二阶导 (FastLaplacian!)
    lap = fast_lap(net, x_int, y0=y, output_indices=[0, 1, 2])  # (N, 3)
    lap_u, lap_v, lap_w = lap[:, 0:1], lap[:, 1:2], lap[:, 2:3]

    res_u = u_t + u*u_x + v*u_y + w*u_z - mu*lap_u
    res_v = v_t + u*v_x + v*v_y + w*v_z - mu*lap_v
    res_w = w_t + u*w_x + v*w_y + w*w_z - mu*lap_w

    loss_pde = (res_u**2).mean() + (res_v**2).mean() + (res_w**2).mean()

    y_ic = net(x_ic)
    loss_ic = ((y_ic - ic_true)**2).mean()

    return loss_pde + 10 * loss_ic, loss_pde.item(), loss_ic.item()


def loss_fd_only(net, x_int, x_ic, ic_true, fast_lap):
    """纯FD: 一阶导也用FD (最大加速)"""
    N = x_int.shape[0]
    h = fast_lap.h if hasattr(fast_lap, 'h') else fast_lap._fast_lap.h
    x_data = x_int.data

    y = net(x_int)
    u, v, w = y[:, 0:1], y[:, 1:2], y[:, 2:3]

    # 一阶导用 FD (中心差分)
    u_derivs = torch.zeros(N, 4, device=x_int.device)  # t, x, y, z
    v_derivs = torch.zeros(N, 4, device=x_int.device)
    w_derivs = torch.zeros(N, 4, device=x_int.device)
    lap_u = torch.zeros(N, 1, device=x_int.device)
    lap_v = torch.zeros(N, 1, device=x_int.device)
    lap_w = torch.zeros(N, 1, device=x_int.device)

    # 对所有4个维度做扰动, 一次性
    perturbations = [x_int]
    for dim in range(4):
        e = torch.zeros_like(x_data); e[:, dim] = h
        perturbations.extend([x_data + e, x_data - e])
    x_batch = torch.cat(perturbations, dim=0)
    y_batch = net(x_batch)

    y0 = y_batch[:N]

    for i in range(4):
        yp = y_batch[(2*i+1)*N:(2*i+2)*N]
        ym = y_batch[(2*i+2)*N:(2*i+3)*N]
        # 一阶导
        u_derivs[:, i] = (yp[:, 0] - ym[:, 0]) / (2*h)
        v_derivs[:, i] = (yp[:, 1] - ym[:, 1]) / (2*h)
        w_derivs[:, i] = (yp[:, 2] - ym[:, 2]) / (2*h)
        # 二阶导 (只对空间维度)
        if i >= 1:
            lap_u += (yp[:, 0:1] - 2*y0[:, 0:1] + ym[:, 0:1]) / (h*h)
            lap_v += (yp[:, 1:2] - 2*y0[:, 1:2] + ym[:, 1:2]) / (h*h)
            lap_w += (yp[:, 2:3] - 2*y0[:, 2:3] + ym[:, 2:3]) / (h*h)

    u_t = u_derivs[:, 0:1]; u_x = u_derivs[:, 1:2]
    u_y = u_derivs[:, 2:3]; u_z = u_derivs[:, 3:4]
    v_t = v_derivs[:, 0:1]; v_x = v_derivs[:, 1:2]
    v_y = v_derivs[:, 2:3]; v_z = v_derivs[:, 3:4]
    w_t = w_derivs[:, 0:1]; w_x = w_derivs[:, 1:2]
    w_y = w_derivs[:, 2:3]; w_z = w_derivs[:, 3:4]

    res_u = u_t + u*u_x + v*u_y + w*u_z - mu*lap_u
    res_v = v_t + u*v_x + v*v_y + w*v_z - mu*lap_v
    res_w = w_t + u*w_x + v*w_y + w*w_z - mu*lap_w

    loss_pde = (res_u**2).mean() + (res_v**2).mean() + (res_w**2).mean()

    y_ic = net(x_ic)
    loss_ic = ((y_ic - ic_true)**2).mean()

    return loss_pde + 10 * loss_ic, loss_pde.item(), loss_ic.item()


# ========== 采样 ==========
N_int = 10000
N_ic = 3000

x_int = torch.zeros(N_int, 4, dtype=dtype, device=cuda)
x_int[:, 0] = torch.rand(N_int, device=cuda) * 0.5  # t in [0, 0.5]
x_int[:, 1] = torch.rand(N_int, device=cuda) * 2*pi - pi
x_int[:, 2] = torch.rand(N_int, device=cuda) * 2*pi - pi
x_int[:, 3] = torch.rand(N_int, device=cuda) * 2*pi - pi
x_int.requires_grad_(True)

x_ic = torch.zeros(N_ic, 4, dtype=dtype, device=cuda)
x_ic[:, 1] = torch.rand(N_ic, device=cuda) * 2*pi - pi
x_ic[:, 2] = torch.rand(N_ic, device=cuda) * 2*pi - pi
x_ic[:, 3] = torch.rand(N_ic, device=cuda) * 2*pi - pi
x_ic.requires_grad_(True)
ic_true = exact_ic(x_ic).detach()

print(f'Collocation: N_int={N_int}, N_ic={N_ic}')

# ========== 训练 ==========
Nl, Nn = 5, 64
N_epochs = 2000
lr = 1e-3

methods = {
    'baseline': {
        'loss_fn': lambda net, x, xic, ic: loss_baseline(net, x, xic, ic),
    },
    'fast_h0.1': {
        'loss_fn': lambda net, x, xic, ic: loss_fast(net, x, xic, ic,
                                                       FastLaplacian([1,2,3], h=0.1)),
    },
    'fast_h0.3': {
        'loss_fn': lambda net, x, xic, ic: loss_fast(net, x, xic, ic,
                                                       FastLaplacian([1,2,3], h=0.3)),
    },
    'fd_only_h0.1': {
        'loss_fn': lambda net, x, xic, ic: loss_fd_only(net, x, xic, ic,
                                                          FastLaplacian([1,2,3], h=0.1)),
    },
}

all_histories = {}

for name, cfg in methods.items():
    print(f'\n{"="*60}')
    print(f'Training: {name}')
    print(f'{"="*60}')

    torch.manual_seed(42)
    net = build_net(Nl, Nn).to(dtype).to(cuda)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    losses = []
    pde_losses = []
    ic_losses = []
    times_per_ep = []
    t_start = time.time()

    for ep in range(1, N_epochs + 1):
        optimizer.zero_grad()
        torch.cuda.synchronize(); t0 = time.perf_counter()

        loss, l_pde, l_ic = cfg['loss_fn'](net, x_int, x_ic, ic_true)

        torch.cuda.synchronize(); t1 = time.perf_counter()

        loss.backward()
        optimizer.step()
        scheduler.step()

        torch.cuda.synchronize(); t2 = time.perf_counter()

        losses.append(loss.item())
        pde_losses.append(l_pde)
        ic_losses.append(l_ic)
        times_per_ep.append(t2 - t0)

        if ep % 500 == 0 or ep == 1:
            avg_ms = np.mean(times_per_ep[-100:]) * 1000
            print(f'  ep={ep:5d}  loss={loss.item():.4e}  pde={l_pde:.4e}  '
                  f'ic={l_ic:.4e}  {avg_ms:.1f}ms/ep')

    wall = time.time() - t_start
    avg_ms = np.mean(times_per_ep) * 1000
    all_histories[name] = {
        'losses': losses, 'pde': pde_losses, 'ic': ic_losses,
        'times': times_per_ep, 'wall': wall, 'avg_ms': avg_ms,
    }
    print(f'  Total: {wall:.1f}s, avg: {avg_ms:.1f}ms/ep')
    del net, optimizer; torch.cuda.empty_cache()


# ========== 绘图 ==========
output_dir = os.path.dirname(os.path.abspath(__file__))
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss vs epoch
ax = axes[0, 0]
for name, h in all_histories.items():
    ax.semilogy(h['losses'], label=f'{name} ({h["avg_ms"]:.1f}ms/ep)', alpha=0.8)
ax.set_xlabel('Epoch'); ax.set_ylabel('Total Loss')
ax.set_title('Loss vs Epoch'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Loss vs wall time
ax = axes[0, 1]
for name, h in all_histories.items():
    cum_time = np.cumsum(h['times'])
    ax.semilogy(cum_time, h['losses'], label=name, alpha=0.8)
ax.set_xlabel('Wall time (s)'); ax.set_ylabel('Total Loss')
ax.set_title('Loss vs Wall Time (speed matters!)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# PDE loss vs epoch
ax = axes[1, 0]
for name, h in all_histories.items():
    ax.semilogy(h['pde'], label=name, alpha=0.8)
ax.set_xlabel('Epoch'); ax.set_ylabel('PDE Loss')
ax.set_title('PDE Residual'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# IC loss vs epoch
ax = axes[1, 1]
for name, h in all_histories.items():
    ax.semilogy(h['ic'], label=name, alpha=0.8)
ax.set_xlabel('Epoch'); ax.set_ylabel('IC Loss')
ax.set_title('IC Loss'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig.suptitle(f'FastLaplacian vs Baseline: 3D Burgers + Viscosity (μ={mu})\n'
             f'Network: {Nl}×{Nn}, N_int={N_int}, N_ic={N_ic}',
             fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'training_comparison_real.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ========== 汇总 ==========
print(f'\n{"="*70}')
print(f'SUMMARY')
print(f'{"="*70}')
base_wall = all_histories['baseline']['wall']
print(f'{"Method":<20} {"Final Loss":<14} {"Avg ms/ep":<12} {"Wall(s)":<10} {"Speedup":<10}')
print(f'{"-"*66}')
for name, h in all_histories.items():
    sp = base_wall / h['wall']
    print(f'{name:<20} {h["losses"][-1]:<14.4e} {h["avg_ms"]:<12.1f} {h["wall"]:<10.1f} {sp:<10.2f}x')

print(f'\nPlot saved: {output_dir}/training_comparison_real.png')
print('Done!')
