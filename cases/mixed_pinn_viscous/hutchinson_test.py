"""
Hutchinson/Stein Stochastic Laplacian Estimator for PINNs

核心idea: 用随机方向的有限差分估计Laplacian, 替代 per-axis FD 或 nested autograd.

数学基础:
  Hutchinson trace estimator:
    Tr(H) = E_ξ[ξᵀ H ξ]   where ξ is Rademacher or Gaussian random vector

  对于 Laplacian (= Tr(Hessian)):
    ∇²f(x) = Tr(∂²f/∂x²) = E_ξ[ ξᵀ · ∇²f · ξ ]
            = E_ξ[ (f(x+εξ) + f(x-εξ) - 2f(x)) / ε² ]
    where ξ ~ N(0,I) on spatial dims only

  关键优势:
    - 标准FD: 需要 2d 次扰动 (d=空间维度, 每个轴正负各1次)
    - Hutchinson: 只需 2K 次扰动 (K=随机样本数, 通常K=1~3)
    - 3D时: 6次 vs 2次, 省3x forward pass!

  精度: 期望是无偏的, 方差可以通过多次采样降低.
        对于训练来说, 方差被mini-batch averaging部分抵消.

验证:
  1. 精度: Hutchinson估计 vs 精确autograd Laplacian
  2. 速度: Hutchinson vs FD batched vs nested autograd
  3. 训练收敛性: 噪声估计是否影响收敛
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
print(f'Device: {cuda}\n')

pi = np.pi


def build_net(Nl, Nn, n_in=4, n_out=3):
    layers = [nn.Linear(n_in, Nn), nn.Tanh()]
    for _ in range(Nl - 2):
        layers += [nn.Linear(Nn, Nn), nn.Tanh()]
    layers += [nn.Linear(Nn, n_out)]
    return nn.Sequential(*layers)


def gradients(out, inp):
    return torch.autograd.grad(out, inp, grad_outputs=torch.ones_like(out),
                               create_graph=True)[0]


# ========== 三种Laplacian计算方法 ==========

def laplacian_autograd(net, x, spatial_dims=[1,2,3], output_indices=[0,1,2]):
    """Baseline: nested autograd (精确但慢)"""
    y = net(x)
    N = x.shape[0]
    n_out = len(output_indices)
    lap = torch.zeros(N, n_out, device=x.device, dtype=x.dtype)

    for j, oi in enumerate(output_indices):
        g = gradients(y[:, oi:oi+1], x)
        for d in spatial_dims:
            g2 = gradients(g[:, d:d+1], x)[:, d:d+1]
            lap[:, j:j+1] += g2
    return lap


def laplacian_fd(net, x, spatial_dims=[1,2,3], output_indices=[0,1,2], h=0.1):
    """FD batched: 中心差分 (快但有截断误差)"""
    N = x.shape[0]
    x_data = x.data

    perturbations = [x]
    for dim in spatial_dims:
        e = torch.zeros_like(x_data); e[:, dim] = h
        perturbations.append(x_data + e)
        perturbations.append(x_data - e)

    x_batch = torch.cat(perturbations, dim=0)
    y_batch = net(x_batch)
    y0 = y_batch[:N]

    lap = torch.zeros(N, len(output_indices), device=x.device, dtype=x.dtype)
    for i, dim in enumerate(spatial_dims):
        yp = y_batch[(2*i+1)*N:(2*i+2)*N]
        ym = y_batch[(2*i+2)*N:(2*i+3)*N]
        for j, oi in enumerate(output_indices):
            lap[:, j] += (yp[:, oi] - 2*y0[:, oi] + ym[:, oi]) / (h*h)
    return lap


def laplacian_hutchinson(net, x, spatial_dims=[1,2,3], output_indices=[0,1,2],
                          h=0.1, K=1, rng='rademacher'):
    """
    Hutchinson stochastic Laplacian estimator.

    ∇²f(x) ≈ (1/K) Σ_k d * (f(x+h*ξ_k) + f(x-h*ξ_k) - 2f(x)) / h²

    where ξ_k are random unit vectors on spatial dims,
    and d = number of spatial dimensions.

    Args:
        K: number of random directions (more = less variance, more forward passes)
        rng: 'rademacher' (±1 on each dim) or 'gaussian' (normal random)
    """
    N = x.shape[0]
    d = len(spatial_dims)
    x_data = x.data

    perturbations = [x]  # center point (keeps grad connection)

    for k in range(K):
        # Sample random direction on spatial dims
        if rng == 'rademacher':
            # Rademacher: each component ±1 with equal probability
            xi = torch.zeros_like(x_data)
            signs = torch.randint(0, 2, (N, d), device=x.device, dtype=x.dtype) * 2 - 1
            for i, dim in enumerate(spatial_dims):
                xi[:, dim] = signs[:, i]
            # Normalize to unit vector: |xi| = sqrt(d), so divide by sqrt(d)
            xi = xi / np.sqrt(d)
        elif rng == 'gaussian':
            xi = torch.zeros_like(x_data)
            gauss = torch.randn(N, d, device=x.device, dtype=x.dtype)
            norm = gauss.norm(dim=1, keepdim=True).clamp(min=1e-8)
            gauss = gauss / norm  # unit vector
            for i, dim in enumerate(spatial_dims):
                xi[:, dim] = gauss[:, i]
        else:
            raise ValueError(f'Unknown rng: {rng}')

        perturbations.append(x_data + h * xi)
        perturbations.append(x_data - h * xi)

    x_batch = torch.cat(perturbations, dim=0)
    y_batch = net(x_batch)
    y0 = y_batch[:N]

    lap = torch.zeros(N, len(output_indices), device=x.device, dtype=x.dtype)

    for k in range(K):
        yp = y_batch[(2*k+1)*N:(2*k+2)*N]
        ym = y_batch[(2*k+2)*N:(2*k+3)*N]
        for j, oi in enumerate(output_indices):
            # d * (f(x+hξ) + f(x-hξ) - 2f(x)) / h²
            # Factor d comes from E[ξᵢ²] = 1/d for unit vectors
            lap[:, j] += d * (yp[:, oi] + ym[:, oi] - 2*y0[:, oi]) / (h*h)

    lap = lap / K  # average over K samples

    return lap


# ========== Test 1: 精度验证 ==========
print('='*70)
print('Test 1: Accuracy — Hutchinson vs FD vs Autograd')
print('='*70)

Nl, Nn = 6, 128
net = build_net(Nl, Nn).to(dtype).to(cuda)
N = 5000
x = torch.rand(N, 4, dtype=dtype, device=cuda) * 2 - 1
x.requires_grad_(True)

# Reference: autograd
lap_ref = laplacian_autograd(net, x).detach()

# FD
for h in [0.1, 0.3]:
    lap_fd = laplacian_fd(net, x, h=h).detach()
    err = (lap_fd - lap_ref).abs()
    rel = err / (lap_ref.abs() + 1e-8)
    print(f'  FD h={h}: mean_err={err.mean():.4e}, rel_err={rel.mean():.4e}')

# Hutchinson with different K
for K in [1, 2, 3, 5, 10]:
    errors = []
    for trial in range(10):
        lap_h = laplacian_hutchinson(net, x, h=0.1, K=K, rng='rademacher').detach()
        err = (lap_h - lap_ref).abs()
        errors.append(err.mean().item())
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    print(f'  Hutchinson K={K:2d} (Rademacher): mean_err={mean_err:.4e} ± {std_err:.4e}')

# Gaussian
for K in [1, 3, 10]:
    errors = []
    for trial in range(10):
        lap_h = laplacian_hutchinson(net, x, h=0.1, K=K, rng='gaussian').detach()
        err = (lap_h - lap_ref).abs()
        errors.append(err.mean().item())
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    print(f'  Hutchinson K={K:2d} (Gaussian):   mean_err={mean_err:.4e} ± {std_err:.4e}')


# ========== Test 2: 速度对比 ==========
print(f'\n{"="*70}')
print('Test 2: Speed — per-iteration timing')
print('='*70)

N = 10000
x = torch.rand(N, 4, dtype=dtype, device=cuda) * 2 - 1
x.requires_grad_(True)

# Autograd baseline
torch.cuda.synchronize()
times = []
for _ in range(30):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    lap = laplacian_autograd(net, x)
    loss = (lap**2).mean()
    loss.backward()
    torch.cuda.synchronize(); t1 = time.perf_counter()
    times.append(t1 - t0)
    net.zero_grad()
autograd_ms = np.mean(times[5:]) * 1000
print(f'  Autograd (nested):    {autograd_ms:7.2f} ms')

# FD batched
torch.cuda.synchronize()
times = []
for _ in range(30):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    lap = laplacian_fd(net, x, h=0.1)
    loss = (lap**2).mean()
    loss.backward()
    torch.cuda.synchronize(); t1 = time.perf_counter()
    times.append(t1 - t0)
    net.zero_grad()
fd_ms = np.mean(times[5:]) * 1000
print(f'  FD batched (h=0.1):   {fd_ms:7.2f} ms  ({autograd_ms/fd_ms:.2f}x)')

# Hutchinson K=1
for K in [1, 2, 3]:
    torch.cuda.synchronize()
    times = []
    for _ in range(30):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        lap = laplacian_hutchinson(net, x, h=0.1, K=K)
        loss = (lap**2).mean()
        loss.backward()
        torch.cuda.synchronize(); t1 = time.perf_counter()
        times.append(t1 - t0)
        net.zero_grad()
    hutch_ms = np.mean(times[5:]) * 1000
    print(f'  Hutchinson K={K}:       {hutch_ms:7.2f} ms  ({autograd_ms/hutch_ms:.2f}x)')


# ========== Test 3: 训练对比 ==========
print(f'\n{"="*70}')
print('Test 3: Training convergence — 3D viscous Burgers')
print('='*70)

mu = 0.01

def exact_ic(x):
    u = torch.sin(x[:, 1]) * torch.cos(x[:, 2]) * torch.cos(x[:, 3])
    v = -torch.cos(x[:, 1]) * torch.sin(x[:, 2]) * torch.cos(x[:, 3])
    w = torch.zeros_like(u)
    return torch.stack([u, v, w], dim=1)


def train_method(method_name, lap_fn, N_epochs=3000):
    """Train with given Laplacian method and return history"""
    torch.manual_seed(42)
    net = build_net(6, 128).to(dtype).to(cuda)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_epochs, eta_min=1e-5)

    N_int, N_ic = 10000, 3000
    x_int = torch.zeros(N_int, 4, dtype=dtype, device=cuda)
    x_int[:, 0] = torch.rand(N_int, device=cuda) * 0.5
    x_int[:, 1:] = torch.rand(N_int, 3, device=cuda) * 2*pi - pi
    x_int.requires_grad_(True)

    x_ic = torch.zeros(N_ic, 4, dtype=dtype, device=cuda)
    x_ic[:, 1:] = torch.rand(N_ic, 3, device=cuda) * 2*pi - pi
    x_ic.requires_grad_(True)
    ic_true = exact_ic(x_ic).detach()

    losses = []
    pde_losses = []
    times_per_ep = []

    for ep in range(1, N_epochs + 1):
        optimizer.zero_grad()
        torch.cuda.synchronize(); t0 = time.perf_counter()

        y = net(x_int)
        u, v, w = y[:, 0:1], y[:, 1:2], y[:, 2:3]

        # 一阶导 (autograd, 所有方法都需要)
        du = gradients(u, x_int)
        u_t, u_x, u_y, u_z = du[:, 0:1], du[:, 1:2], du[:, 2:3], du[:, 3:4]
        dv = gradients(v, x_int)
        v_t, v_x, v_y, v_z = dv[:, 0:1], dv[:, 1:2], dv[:, 2:3], dv[:, 3:4]
        dw = gradients(w, x_int)
        w_t, w_x, w_y, w_z = dw[:, 0:1], dw[:, 1:2], dw[:, 2:3], dw[:, 3:4]

        # 二阶导 (Laplacian — 这里用不同方法!)
        lap = lap_fn(net, x_int)  # (N, 3)
        lap_u, lap_v, lap_w = lap[:, 0:1], lap[:, 1:2], lap[:, 2:3]

        res_u = u_t + u*u_x + v*u_y + w*u_z - mu*lap_u
        res_v = v_t + u*v_x + v*v_y + w*v_z - mu*lap_v
        res_w = w_t + u*w_x + v*w_y + w*w_z - mu*lap_w

        l_pde = (res_u**2).mean() + (res_v**2).mean() + (res_w**2).mean()
        l_ic = ((net(x_ic) - ic_true)**2).mean()
        loss = l_pde + 10 * l_ic

        loss.backward()
        optimizer.step()
        scheduler.step()

        torch.cuda.synchronize(); t1 = time.perf_counter()
        losses.append(loss.item())
        pde_losses.append(l_pde.item())
        times_per_ep.append((t1 - t0) * 1000)

        if ep % 500 == 0 or ep == 1:
            avg_ms = np.mean(times_per_ep[-100:])
            print(f'  [{method_name}] ep={ep:5d}  loss={loss.item():.4e}  '
                  f'pde={l_pde.item():.4e}  {avg_ms:.1f}ms/ep')

    avg_ms = np.mean(times_per_ep)
    print(f'  [{method_name}] Done: avg={avg_ms:.1f}ms/ep, final_loss={losses[-1]:.4e}')
    del net, optimizer
    torch.cuda.empty_cache()
    return losses, pde_losses, times_per_ep


# 定义不同的Laplacian方法
methods = {
    'autograd': lambda net, x: laplacian_autograd(net, x),
    'FD_h0.1': lambda net, x: laplacian_fd(net, x, h=0.1),
    'Hutch_K1': lambda net, x: laplacian_hutchinson(net, x, h=0.1, K=1),
    'Hutch_K2': lambda net, x: laplacian_hutchinson(net, x, h=0.1, K=2),
    'Hutch_K3': lambda net, x: laplacian_hutchinson(net, x, h=0.1, K=3),
}

all_results = {}
for name, fn in methods.items():
    print(f'\n--- {name} ---')
    losses, pde_losses, times = train_method(name, fn, N_epochs=3000)
    all_results[name] = {
        'losses': losses, 'pde': pde_losses, 'times': times,
        'avg_ms': np.mean(times)
    }


# ========== 绘图 ==========
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(output_dir, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss vs epoch
ax = axes[0]
for name, r in all_results.items():
    ax.semilogy(r['losses'], label=f'{name} ({r["avg_ms"]:.1f}ms)', alpha=0.7)
ax.set_xlabel('Epoch'); ax.set_ylabel('Total Loss')
ax.set_title('Loss vs Epoch'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Loss vs wall time
ax = axes[1]
for name, r in all_results.items():
    cum_t = np.cumsum(r['times']) / 1000  # seconds
    ax.semilogy(cum_t, r['losses'], label=name, alpha=0.7)
ax.set_xlabel('Wall Time (s)'); ax.set_ylabel('Total Loss')
ax.set_title('Loss vs Wall Time'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ms/epoch bar chart
ax = axes[2]
names = list(all_results.keys())
ms_vals = [all_results[n]['avg_ms'] for n in names]
colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6', '#f39c12']
bars = ax.bar(names, ms_vals, color=colors[:len(names)])
for bar, ms in zip(bars, ms_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{ms:.1f}', ha='center', fontsize=9)
ax.set_ylabel('ms/epoch')
ax.set_title('Speed Comparison')
ax.tick_params(axis='x', rotation=30)

fig.suptitle('Hutchinson Stochastic Laplacian vs FD vs Autograd\n'
             '3D Viscous Burgers, 6×128 network, 10k points',
             fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'hutchinson_comparison.png'), dpi=150)
plt.close()

# ========== 汇总 ==========
print(f'\n{"="*70}')
print('SUMMARY')
print('='*70)
base_ms = all_results['autograd']['avg_ms']
print(f'{"Method":<15} {"ms/ep":<10} {"Speedup":<10} {"Final Loss":<14} {"Forward passes":<15}')
print('-'*64)
for name, r in all_results.items():
    sp = base_ms / r['avg_ms']
    n_fwd = {'autograd': '1+9nested', 'FD_h0.1': '1+6', 'Hutch_K1': '1+2',
             'Hutch_K2': '1+4', 'Hutch_K3': '1+6'}
    print(f'{name:<15} {r["avg_ms"]:<10.1f} {sp:<10.2f}x {r["losses"][-1]:<14.4e} {n_fwd.get(name, "?"):<15}')

print(f'\nPlot saved to {output_dir}/hutchinson_comparison.png')
print('Done!')
