"""
Benchmark v4: 找到最优FD步长 + 实际训练对比

关键发现: FD精度由 截断误差(O(h²)) + 舍入误差(O(ε/h²)) 决定
最优 h* ≈ (ε)^(1/4)
- float32: ε≈1e-7, h*≈ε^0.25≈0.003
- float64: ε≈1e-16, h*≈1e-4

本版测试:
1. 精细扫描h找到最优步长
2. 用最优步长实际训练PINN, 对比baseline的收敛性
3. 混合精度: forward pass用float32 (GPU快), FD差分用float64 (精度好)
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42)
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {cuda}\n')

def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,
                               grad_outputs=torch.ones_like(outputs),
                               create_graph=True)
def to_numpy(t):
    return t.detach().cpu().numpy()

def build_net(Nl, Nn, n_in=4, n_out=5):
    layers = [nn.Linear(n_in, Nn), nn.Tanh()]
    for _ in range(Nl - 2):
        layers += [nn.Linear(Nn, Nn), nn.Tanh()]
    layers += [nn.Linear(Nn, n_out)]
    return nn.Sequential(*layers)


def viscous_baseline(net, x, mu=0.01):
    y = net(x)
    u, v, w = y[:, 2:3], y[:, 3:4], y[:, 4:5]
    du = gradients(u, x)[0]; dv = gradients(v, x)[0]; dw = gradients(w, x)[0]
    lap_u = gradients(du[:, 1:2], x)[0][:, 1:2] + \
            gradients(du[:, 2:3], x)[0][:, 2:3] + \
            gradients(du[:, 3:4], x)[0][:, 3:4]
    lap_v = gradients(dv[:, 1:2], x)[0][:, 1:2] + \
            gradients(dv[:, 2:3], x)[0][:, 2:3] + \
            gradients(dv[:, 3:4], x)[0][:, 3:4]
    lap_w = gradients(dw[:, 1:2], x)[0][:, 1:2] + \
            gradients(dw[:, 2:3], x)[0][:, 2:3] + \
            gradients(dw[:, 3:4], x)[0][:, 3:4]
    loss = mu**2 * ((lap_u**2).mean() + (lap_v**2).mean() + (lap_w**2).mean())
    return loss, torch.cat([lap_u, lap_v, lap_w], dim=1).detach()


def viscous_fd_batched(net, x, mu=0.01, h=3e-3):
    N = x.shape[0]
    x_data = x.data
    perturbations = [x]
    for dim in [1, 2, 3]:
        e = torch.zeros_like(x_data); e[:, dim] = h
        perturbations.append(x_data + e)
        perturbations.append(x_data - e)
    x_batch = torch.cat(perturbations, dim=0)
    y_batch = net(x_batch)
    y0 = y_batch[:N]
    lap = torch.zeros_like(y0[:, 2:5])
    for i in range(3):
        yp = y_batch[(2*i+1)*N:(2*i+2)*N]
        ym = y_batch[(2*i+2)*N:(2*i+3)*N]
        lap += (yp[:, 2:5] - 2*y0[:, 2:5] + ym[:, 2:5]) / (h*h)
    loss = mu**2 * (lap**2).mean()
    return loss, lap.detach()


def viscous_fd_mixedprec(net, x, mu=0.01, h=1e-4):
    """混合精度FD: 网络forward用float32, 差分用float64"""
    N = x.shape[0]
    x_data = x.data

    # 所有点一起forward (float32)
    perturbations = [x]
    for dim in [1, 2, 3]:
        e = torch.zeros_like(x_data); e[:, dim] = h
        perturbations.append(x_data + e)
        perturbations.append(x_data - e)
    x_batch = torch.cat(perturbations, dim=0)

    y_batch_f32 = net(x_batch)
    # 转float64做差分
    y_batch = y_batch_f32.double()

    y0 = y_batch[:N]
    lap = torch.zeros(N, 3, dtype=torch.float64, device=cuda)
    for i in range(3):
        yp = y_batch[(2*i+1)*N:(2*i+2)*N]
        ym = y_batch[(2*i+2)*N:(2*i+3)*N]
        lap += (yp[:, 2:5] - 2*y0[:, 2:5] + ym[:, 2:5]) / (h*h)

    # 转回float32做loss
    lap = lap.float()
    loss = mu**2 * (lap**2).mean()
    return loss, lap.detach()


# ========== Part 1: 最优步长搜索 ==========
print('Part 1: Optimal h search')
print('='*70)

Nl, Nn = 5, 64
net = build_net(Nl, Nn, 4, 5).to(torch.float32).to(cuda)
# 训练几步
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
x_test = torch.rand(500, 4, dtype=torch.float32, device=cuda) * 2 - 1
x_test.requires_grad_(True)
for _ in range(30):
    opt.zero_grad(); loss, _ = viscous_baseline(net, x_test); loss.backward(); opt.step()

# Reference: float64 autograd
net64 = build_net(Nl, Nn, 4, 5).to(torch.float64).to(cuda)
net64.load_state_dict({k: v.double() for k, v in net.state_dict().items()})
x64 = x_test.detach().double().requires_grad_(True)
_, lap_ref = viscous_baseline(net64, x64)
lap_ref = lap_ref.float()

print(f'|lap| ref: mean={lap_ref.abs().mean():.6f}, max={lap_ref.abs().max():.6f}\n')

h_values = [3e-1, 1e-1, 5e-2, 3e-2, 1e-2, 5e-3, 3e-3, 1e-3, 5e-4, 3e-4, 1e-4]
print(f'{"h":<10} {"mean_err":<14} {"rel_err":<14} {"max_err":<14}')
print(f'{"-"*52}')

best_h, best_err = None, float('inf')
for h in h_values:
    x32 = x_test.detach().float().requires_grad_(True)
    _, lap_fd = viscous_fd_batched(net, x32, h=h)
    err = (lap_fd - lap_ref).abs()
    rel = err / (lap_ref.abs() + 1e-10)
    me = err.mean().item()
    if me < best_err:
        best_err = me
        best_h = h
    print(f'{h:<10.0e} {me:<14.4e} {rel.mean().item():<14.4e} {err.max().item():<14.4e}')

print(f'\nBest h (float32): {best_h:.0e}, mean_err={best_err:.4e}')

# 混合精度
print(f'\nMixed precision FD (float64 差分):')
print(f'{"h":<10} {"mean_err":<14} {"rel_err":<14} {"max_err":<14}')
print(f'{"-"*52}')

best_h_mp, best_err_mp = None, float('inf')
for h in h_values:
    x32 = x_test.detach().float().requires_grad_(True)
    _, lap_mp = viscous_fd_mixedprec(net, x32, h=h)
    err = (lap_mp - lap_ref).abs()
    rel = err / (lap_ref.abs() + 1e-10)
    me = err.mean().item()
    if me < best_err_mp:
        best_err_mp = me
        best_h_mp = h
    print(f'{h:<10.0e} {me:<14.4e} {rel.mean().item():<14.4e} {err.max().item():<14.4e}')

print(f'\nBest h (mixed prec): {best_h_mp:.0e}, mean_err={best_err_mp:.4e}')


# ========== Part 2: 训练收敛对比 ==========
print(f'\n{"="*70}')
print(f'Part 2: Training convergence comparison (500 epochs)')
print(f'{"="*70}')

Nl, Nn, N_pts = 5, 64, 5000
N_train = 500

train_methods = [
    ('baseline_autograd', lambda net, x: viscous_baseline(net, x)),
    (f'fd_h={best_h:.0e}', lambda net, x: viscous_fd_batched(net, x, h=best_h)),
    (f'fd_mp_h={best_h_mp:.0e}', lambda net, x: viscous_fd_mixedprec(net, x, h=best_h_mp)),
]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for name, method_fn in train_methods:
    torch.manual_seed(42)
    net = build_net(Nl, Nn, 4, 5).to(torch.float32).to(cuda)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.rand(N_pts, 4, dtype=torch.float32, device=cuda) * 2 - 1
    x.requires_grad_(True)

    losses = []
    times_total = []
    t_start = time.time()

    for ep in range(N_train):
        optimizer.zero_grad()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        loss, _ = method_fn(net, x)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(); t2 = time.perf_counter()

        losses.append(loss.item())
        times_total.append(t2 - t0)

    wall = time.time() - t_start
    avg_ms = np.mean(times_total) * 1000

    print(f'  {name:<25} final_loss={losses[-1]:.3e}  avg={avg_ms:.2f}ms/ep  wall={wall:.1f}s')

    ax1.semilogy(losses, label=f'{name} ({avg_ms:.1f}ms/ep)')
    cum_time = np.cumsum(times_total)
    ax2.semilogy(cum_time, losses, label=name)

    del net, optimizer, x; torch.cuda.empty_cache()

ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Loss vs Epoch'); ax1.legend(); ax1.grid(True, alpha=0.3)
ax2.set_xlabel('Wall time (s)'); ax2.set_ylabel('Loss')
ax2.set_title('Loss vs Wall time (speed matters!)'); ax2.legend(); ax2.grid(True, alpha=0.3)

fig.suptitle(f'Training: Autograd vs FD (Laplacian), {Nl}×{Nn}, {N_pts} pts', fontsize=14)
fig.tight_layout()
output_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(output_dir, 'training_comparison.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nPlot saved: {output_dir}/training_comparison.png')

# ========== Part 3: 大规模速度对比 ==========
print(f'\n{"="*70}')
print(f'Part 3: Scaling — larger networks and more points')
print(f'{"="*70}')

scale_configs = [
    (5, 64, 5000),
    (5, 64, 10000),
    (5, 64, 30000),
    (6, 128, 5000),
    (6, 128, 10000),
    (6, 128, 30000),
    (8, 256, 10000),
]

N_ep = 30
print(f'{"Config":<16} {"Baseline(ms)":<14} {"FD(ms)":<14} {"Speedup":<10}')
print(f'{"-"*54}')

for Nl, Nn, N_pts in scale_configs:
    results = {}
    for mname, mfn in [('base', lambda n, x: viscous_baseline(n, x)),
                        ('fd', lambda n, x: viscous_fd_batched(n, x, h=best_h))]:
        net = build_net(Nl, Nn, 4, 5).to(torch.float32).to(cuda)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        x = torch.rand(N_pts, 4, dtype=torch.float32, device=cuda) * 2 - 1
        x.requires_grad_(True)

        # warmup
        for _ in range(3):
            opt.zero_grad()
            loss, _ = mfn(net, x)
            loss.backward(); opt.step()
        torch.cuda.synchronize()

        ts = []
        for _ in range(N_ep):
            opt.zero_grad()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            loss, _ = mfn(net, x)
            loss.backward(); opt.step()
            torch.cuda.synchronize(); t1 = time.perf_counter()
            ts.append(t1 - t0)

        results[mname] = np.mean(ts) * 1000
        del net, opt, x; torch.cuda.empty_cache()

    sp = results['base'] / results['fd'] if results['fd'] > 0 else 0
    print(f'{Nl}x{Nn}_{N_pts//1000}k{"":<6} {results["base"]:<14.2f} {results["fd"]:<14.2f} {sp:<10.2f}x')

print('\nDone!')
