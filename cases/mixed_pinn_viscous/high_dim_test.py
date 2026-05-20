"""
高维 Laplacian 对比: Hutchinson vs FD vs Autograd

问题: d维 heat equation
  u_t = μ * Σᵢ ∂²u/∂xᵢ²    (i=1..d)

精确解:
  IC: u(0, x₁...x_d) = Πᵢ sin(xᵢ)
  Solution: u(t, x) = exp(-d*μ*t) * Πᵢ sin(xᵢ)

  Laplacian 精确值: ∇²u = -d * Πᵢ sin(xᵢ)   (每个维度贡献 -sin(xᵢ)*Π_{j≠i}sin(xⱼ))

关键对比:
  FD: 需要 2d 次扰动 forward pass → 随d线性增长
  Hutchinson K=1: 只需 2 次 → 与d无关!

测试维度: d = 3, 5, 10, 20, 50, 100
"""
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

torch.manual_seed(42)
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
print(f'Device: {cuda}\n')

pi = np.pi
mu = 0.01


def build_net(n_in, Nl=5, Nn=128, n_out=1):
    layers = [nn.Linear(n_in, Nn), nn.Tanh()]
    for _ in range(Nl - 2):
        layers += [nn.Linear(Nn, Nn), nn.Tanh()]
    layers += [nn.Linear(Nn, n_out)]
    return nn.Sequential(*layers)


def gradients(out, inp):
    return torch.autograd.grad(out, inp, grad_outputs=torch.ones_like(out),
                               create_graph=True)[0]


# ========== 三种 Laplacian 方法 ==========

def laplacian_autograd(net, x, spatial_dims):
    """Nested autograd — 精确但 O(d) nested backward"""
    y = net(x)
    g = gradients(y, x)
    lap = torch.zeros_like(y)
    for d in spatial_dims:
        g2 = gradients(g[:, d:d+1], x)[:, d:d+1]
        lap += g2
    return lap


def laplacian_fd(net, x, spatial_dims, h=0.1):
    """FD batched — 需要 2d 次 forward pass"""
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
    lap = torch.zeros_like(y0)
    for i, dim in enumerate(spatial_dims):
        yp = y_batch[(2*i+1)*N:(2*i+2)*N]
        ym = y_batch[(2*i+2)*N:(2*i+3)*N]
        lap += (yp - 2*y0 + ym) / (h*h)
    return lap


def laplacian_hutchinson(net, x, spatial_dims, h=0.1, K=1):
    """Hutchinson — 只需 2K 次 forward pass, 与d无关!"""
    N = x.shape[0]
    d = len(spatial_dims)
    x_data = x.data
    perturbations = [x]

    for k in range(K):
        xi = torch.zeros_like(x_data)
        signs = torch.randint(0, 2, (N, d), device=x.device, dtype=x.dtype) * 2 - 1
        for i, dim in enumerate(spatial_dims):
            xi[:, dim] = signs[:, i]
        xi = xi / np.sqrt(d)  # normalize

        perturbations.append(x_data + h * xi)
        perturbations.append(x_data - h * xi)

    x_batch = torch.cat(perturbations, dim=0)
    y_batch = net(x_batch)
    y0 = y_batch[:N]

    lap = torch.zeros_like(y0)
    for k in range(K):
        yp = y_batch[(2*k+1)*N:(2*k+2)*N]
        ym = y_batch[(2*k+2)*N:(2*k+3)*N]
        lap += d * (yp + ym - 2*y0) / (h*h)
    return lap / K


# ========== 精确解 ==========
def exact_solution(x, d, mu):
    """u(t,x) = exp(-d*mu*t) * Π sin(xᵢ)"""
    t = x[:, 0:1]
    prod = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    for i in range(1, d+1):
        prod = prod * torch.sin(x[:, i:i+1])
    return torch.exp(-d * mu * t) * prod


def exact_laplacian(x, d):
    """∇²u = -d * Π sin(xᵢ)  (at t=0)"""
    prod = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    for i in range(1, d+1):
        prod = prod * torch.sin(x[:, i:i+1])
    return -d * prod


# ========== 实验 ==========
dims_to_test = [3, 5, 10, 20, 50, 100]
N = 5000  # 配点数
N_timing = 3000  # 计时用的配点
n_warmup = 5
n_repeats = 30

results_speed = {}    # {d: {method: ms}}
results_accuracy = {} # {d: {method: rel_err}}
results_train = {}    # {d: {method: final_loss}}

for d in dims_to_test:
    print(f'\n{"="*70}')
    print(f'Dimension d = {d}  (input: 1+{d} = {d+1}D, spatial dims: {d})')
    print(f'{"="*70}')

    n_in = d + 1  # t + d spatial dims
    spatial_dims = list(range(1, d+1))

    net = build_net(n_in).to(dtype).to(cuda)
    n_params = sum(p.numel() for p in net.parameters())
    print(f'  Network: 5x128, params={n_params}')

    # 采样: t=0, x_i ∈ [-π, π]
    x = torch.zeros(N, n_in, dtype=dtype, device=cuda)
    x[:, 0] = 0.0  # t=0 for accuracy test
    x[:, 1:] = torch.rand(N, d, device=cuda) * 2*pi - pi
    x.requires_grad_(True)

    # ---- 精度测试 ----
    print(f'\n  --- Accuracy (vs exact Laplacian at t=0) ---')
    lap_exact = exact_laplacian(x, d).detach()

    results_accuracy[d] = {}

    # Autograd (skip for d > 20, too slow)
    if d <= 20:
        try:
            lap_ag = laplacian_autograd(net, x, spatial_dims).detach()
            err = (lap_ag - lap_exact).abs()
            rel = (err / (lap_exact.abs() + 1e-8)).mean().item()
            print(f'  Autograd:       rel_err = {rel:.4e}')
            results_accuracy[d]['autograd'] = rel
        except RuntimeError as e:
            print(f'  Autograd:       FAILED (OOM or too slow)')
            results_accuracy[d]['autograd'] = float('nan')
    else:
        print(f'  Autograd:       SKIPPED (d={d} too high)')
        results_accuracy[d]['autograd'] = float('nan')

    # FD
    if d <= 50:  # FD with d=100 needs 200 forward passes, might OOM
        try:
            lap_fd = laplacian_fd(net, x, spatial_dims, h=0.3).detach()
            err = (lap_fd - lap_exact).abs()
            rel = (err / (lap_exact.abs() + 1e-8)).mean().item()
            print(f'  FD (h=0.3):     rel_err = {rel:.4e}')
            results_accuracy[d]['FD'] = rel
        except RuntimeError:
            print(f'  FD:             OOM!')
            results_accuracy[d]['FD'] = float('nan')
    else:
        print(f'  FD:             SKIPPED (d={d}, would need {2*d} forward passes)')
        results_accuracy[d]['FD'] = float('nan')

    # Hutchinson K=1
    errs_h1 = []
    for _ in range(20):
        lap_h1 = laplacian_hutchinson(net, x, spatial_dims, h=0.3, K=1).detach()
        err = (lap_h1 - lap_exact).abs()
        rel = (err / (lap_exact.abs() + 1e-8)).mean().item()
        errs_h1.append(rel)
    print(f'  Hutch K=1:      rel_err = {np.mean(errs_h1):.4e} ± {np.std(errs_h1):.4e}')
    results_accuracy[d]['Hutch_K1'] = np.mean(errs_h1)

    # Hutchinson K=3
    errs_h3 = []
    for _ in range(20):
        lap_h3 = laplacian_hutchinson(net, x, spatial_dims, h=0.3, K=3).detach()
        err = (lap_h3 - lap_exact).abs()
        rel = (err / (lap_exact.abs() + 1e-8)).mean().item()
        errs_h3.append(rel)
    print(f'  Hutch K=3:      rel_err = {np.mean(errs_h3):.4e} ± {np.std(errs_h3):.4e}')
    results_accuracy[d]['Hutch_K3'] = np.mean(errs_h3)

    del x; torch.cuda.empty_cache()

    # ---- 速度测试 ----
    print(f'\n  --- Speed (Laplacian only, N={N_timing}) ---')
    x_t = torch.zeros(N_timing, n_in, dtype=dtype, device=cuda)
    x_t[:, 0] = torch.rand(N_timing, device=cuda) * 0.5
    x_t[:, 1:] = torch.rand(N_timing, d, device=cuda) * 2*pi - pi
    x_t.requires_grad_(True)

    results_speed[d] = {}

    # Autograd
    if d <= 20:
        torch.cuda.synchronize()
        times = []
        for rep in range(n_warmup + n_repeats):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            lap = laplacian_autograd(net, x_t, spatial_dims)
            loss = (lap**2).mean()
            loss.backward()
            torch.cuda.synchronize(); t1 = time.perf_counter()
            if rep >= n_warmup:
                times.append(t1 - t0)
            net.zero_grad()
        ms = np.mean(times) * 1000
        print(f'  Autograd:       {ms:8.2f} ms  (2*{d} nested backward)')
        results_speed[d]['autograd'] = ms
    else:
        print(f'  Autograd:       SKIPPED')
        results_speed[d]['autograd'] = float('nan')

    # FD
    if d <= 100:
        try:
            torch.cuda.synchronize()
            times = []
            for rep in range(n_warmup + n_repeats):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                lap = laplacian_fd(net, x_t, spatial_dims, h=0.3)
                loss = (lap**2).mean()
                loss.backward()
                torch.cuda.synchronize(); t1 = time.perf_counter()
                if rep >= n_warmup:
                    times.append(t1 - t0)
                net.zero_grad()
            ms = np.mean(times) * 1000
            print(f'  FD (h=0.3):     {ms:8.2f} ms  ({2*d} forward passes)')
            results_speed[d]['FD'] = ms
        except RuntimeError:
            print(f'  FD:             OOM!')
            results_speed[d]['FD'] = float('nan')
    else:
        results_speed[d]['FD'] = float('nan')

    # Hutchinson K=1
    torch.cuda.synchronize()
    times = []
    for rep in range(n_warmup + n_repeats):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        lap = laplacian_hutchinson(net, x_t, spatial_dims, h=0.3, K=1)
        loss = (lap**2).mean()
        loss.backward()
        torch.cuda.synchronize(); t1 = time.perf_counter()
        if rep >= n_warmup:
            times.append(t1 - t0)
        net.zero_grad()
    ms = np.mean(times) * 1000
    print(f'  Hutch K=1:      {ms:8.2f} ms  (2 forward passes, always!)')
    results_speed[d]['Hutch_K1'] = ms

    # Hutchinson K=3
    torch.cuda.synchronize()
    times = []
    for rep in range(n_warmup + n_repeats):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        lap = laplacian_hutchinson(net, x_t, spatial_dims, h=0.3, K=3)
        loss = (lap**2).mean()
        loss.backward()
        torch.cuda.synchronize(); t1 = time.perf_counter()
        if rep >= n_warmup:
            times.append(t1 - t0)
        net.zero_grad()
    ms = np.mean(times) * 1000
    print(f'  Hutch K=3:      {ms:8.2f} ms  (6 forward passes)')
    results_speed[d]['Hutch_K3'] = ms

    del x_t, net; torch.cuda.empty_cache()

    # ---- 训练测试 (short, 500 epochs) ----
    print(f'\n  --- Training (500 epochs, heat equation) ---')
    results_train[d] = {}

    for method_name, lap_fn in [
        ('FD', lambda net, x, sd=spatial_dims: laplacian_fd(net, x, sd, h=0.3)),
        ('Hutch_K1', lambda net, x, sd=spatial_dims: laplacian_hutchinson(net, x, sd, h=0.3, K=1)),
    ]:
        # skip FD for very high d
        if method_name == 'FD' and d > 50:
            print(f'  [{method_name}] SKIPPED (d={d})')
            results_train[d][method_name] = {'loss': float('nan'), 'ms': float('nan')}
            continue

        torch.manual_seed(42)
        net = build_net(n_in).to(dtype).to(cuda)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)

        N_tr = min(5000, max(1000, 50000 // d))  # 减少高维的点数
        x_tr = torch.zeros(N_tr, n_in, dtype=dtype, device=cuda)
        x_tr[:, 0] = torch.rand(N_tr, device=cuda) * 0.5
        x_tr[:, 1:] = torch.rand(N_tr, d, device=cuda) * 2*pi - pi
        x_tr.requires_grad_(True)

        # IC points
        N_ic = min(2000, N_tr)
        x_ic = torch.zeros(N_ic, n_in, dtype=dtype, device=cuda)
        x_ic[:, 1:] = torch.rand(N_ic, d, device=cuda) * 2*pi - pi
        ic_true = exact_solution(x_ic, d, mu).detach()

        times_ep = []
        losses_ep = []
        for ep in range(1, 501):
            opt.zero_grad()
            torch.cuda.synchronize(); t0 = time.perf_counter()

            y = net(x_tr)
            # u_t
            u_t = gradients(y, x_tr)[:, 0:1]
            # Laplacian
            lap = lap_fn(net, x_tr)
            # PDE: u_t = μ * ∇²u
            res = u_t - mu * lap
            l_pde = (res**2).mean()
            l_ic = ((net(x_ic) - ic_true)**2).mean()
            loss = l_pde + 10 * l_ic

            loss.backward()
            opt.step()

            torch.cuda.synchronize(); t1 = time.perf_counter()
            times_ep.append((t1 - t0) * 1000)
            losses_ep.append(loss.item())

        avg_ms = np.mean(times_ep[10:])  # skip warmup
        final_loss = losses_ep[-1]
        print(f'  [{method_name}] avg={avg_ms:.1f}ms/ep, final_loss={final_loss:.4e}')
        results_train[d][method_name] = {'loss': final_loss, 'ms': avg_ms}

        del net, opt, x_tr, x_ic
        torch.cuda.empty_cache()


# ========== 绘图 ==========
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(output_dir, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. Speed vs dimension
ax = axes[0]
for method in ['autograd', 'FD', 'Hutch_K1', 'Hutch_K3']:
    ds = [d for d in dims_to_test if d in results_speed and method in results_speed[d]]
    ms = [results_speed[d][method] for d in ds]
    valid = [(d, m) for d, m in zip(ds, ms) if not np.isnan(m)]
    if valid:
        ax.plot([v[0] for v in valid], [v[1] for v in valid],
                'o-', label=method, linewidth=2, markersize=8)

ax.set_xlabel('Spatial Dimension d', fontsize=12)
ax.set_ylabel('Laplacian Time (ms)', fontsize=12)
ax.set_title('Speed: Laplacian Computation Time vs Dimension', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# 2. Speedup vs dimension
ax = axes[1]
for method in ['FD', 'Hutch_K1', 'Hutch_K3']:
    speedups = []
    ds_valid = []
    for d in dims_to_test:
        if d in results_speed:
            # use FD as reference when autograd not available
            ref = results_speed[d].get('autograd', float('nan'))
            if np.isnan(ref):
                ref = results_speed[d].get('FD', float('nan'))
            val = results_speed[d].get(method, float('nan'))
            if not np.isnan(ref) and not np.isnan(val) and val > 0:
                speedups.append(ref / val)
                ds_valid.append(d)
    if speedups:
        ax.plot(ds_valid, speedups, 'o-', label=f'{method} vs autograd/FD',
                linewidth=2, markersize=8)

# theoretical speedup line: Hutch K=1 should scale as d (saving 2d → 2 fwd passes)
d_theory = np.array([3, 5, 10, 20, 50, 100])
ax.plot(d_theory, d_theory, 'k--', alpha=0.3, label='Theoretical: d×')
ax.set_xlabel('Spatial Dimension d', fontsize=12)
ax.set_ylabel('Speedup vs Baseline', fontsize=12)
ax.set_title('Hutchinson Speedup Scales with Dimension', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 3. Training: ms/epoch comparison
ax = axes[2]
for method in ['FD', 'Hutch_K1']:
    ds_valid = []
    ms_valid = []
    for d in dims_to_test:
        if d in results_train and method in results_train[d]:
            ms = results_train[d][method]['ms']
            if not np.isnan(ms):
                ds_valid.append(d)
                ms_valid.append(ms)
    if ds_valid:
        ax.plot(ds_valid, ms_valid, 'o-', label=method, linewidth=2, markersize=8)

ax.set_xlabel('Spatial Dimension d', fontsize=12)
ax.set_ylabel('Training ms/epoch', fontsize=12)
ax.set_title('Training Speed: FD vs Hutchinson', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

fig.suptitle('High-Dimensional Laplacian: Hutchinson vs FD vs Autograd\n'
             'Heat equation u_t = μ∇²u, 5×128 network, N=3000',
             fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'high_dim_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()


# ========== 汇总表格 ==========
print(f'\n{"="*80}')
print('SPEED SUMMARY (ms)')
print(f'{"="*80}')
print(f'{"d":<6}', end='')
for method in ['autograd', 'FD', 'Hutch_K1', 'Hutch_K3']:
    print(f'{method:<14}', end='')
print(f'  {"Hutch_K1/FD speedup":<20}')
print('-'*80)
for d in dims_to_test:
    print(f'{d:<6}', end='')
    for method in ['autograd', 'FD', 'Hutch_K1', 'Hutch_K3']:
        val = results_speed.get(d, {}).get(method, float('nan'))
        if np.isnan(val):
            print(f'{"---":<14}', end='')
        else:
            print(f'{val:<14.2f}', end='')
    # speedup
    fd_ms = results_speed.get(d, {}).get('FD', float('nan'))
    h1_ms = results_speed.get(d, {}).get('Hutch_K1', float('nan'))
    if not np.isnan(fd_ms) and not np.isnan(h1_ms):
        print(f'  {fd_ms/h1_ms:.2f}x', end='')
    print()

print(f'\n{"="*80}')
print('TRAINING SUMMARY')
print(f'{"="*80}')
print(f'{"d":<6} {"FD ms/ep":<12} {"Hutch ms/ep":<14} {"Speedup":<10} {"FD loss":<14} {"Hutch loss":<14}')
print('-'*70)
for d in dims_to_test:
    fd = results_train.get(d, {}).get('FD', {})
    h1 = results_train.get(d, {}).get('Hutch_K1', {})
    fd_ms = fd.get('ms', float('nan'))
    h1_ms = h1.get('ms', float('nan'))
    fd_loss = fd.get('loss', float('nan'))
    h1_loss = h1.get('loss', float('nan'))
    sp = fd_ms / h1_ms if (not np.isnan(fd_ms) and not np.isnan(h1_ms) and h1_ms > 0) else float('nan')
    print(f'{d:<6} {fd_ms:<12.1f} {h1_ms:<14.1f} {sp:<10.2f}x {fd_loss:<14.4e} {h1_loss:<14.4e}')

print(f'\nKey takeaway: Hutchinson K=1 forward passes = 2 (constant!)')
print(f'              FD forward passes = 2d (linear in dimension)')
print(f'              → Hutchinson dominates for d >> 3')
print(f'\nPlot saved: {output_dir}/high_dim_comparison.png')
print('Done!')
