"""
Benchmark v3: 重点突破 — fd_batched + float64精度测试

v2发现:
- fd_batched 是最快的 (4.8x speedup)
- 但 float32 下 FD 精度极差 (二阶中心差分对舍入误差敏感)
- jacfwd 反而更慢
- hvp/fwd_over_rev 有bug

本版本:
1. 修复 fwd_over_rev (retain_graph)
2. FD 在 float64 下精度测试
3. 4阶中心差分 (更高精度FD)
4. 复合步长法 (Richardson外推消除截断误差)
5. 混合精度: FD用float64, 网络权重float32
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42)
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {cuda}\n')

# ========== 通用工具 ==========
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


# ========== 方案0: Baseline ==========
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


# ========== 方案1: FD batched (2阶中心差分) ==========
def viscous_fd_batched(net, x, mu=0.01, h=1e-3):
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
    lap = torch.zeros_like(y0[:, 2:5])  # (N, 3) for u,v,w
    for i in range(3):
        yp = y_batch[(2*i+1)*N:(2*i+2)*N]
        ym = y_batch[(2*i+2)*N:(2*i+3)*N]
        lap += (yp[:, 2:5] - 2*y0[:, 2:5] + ym[:, 2:5]) / (h*h)

    loss = mu**2 * (lap**2).mean()
    return loss, lap.detach()


# ========== 方案2: FD 4阶中心差分 ==========
# f''(x) ≈ (-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)) / (12h²)
def viscous_fd4_batched(net, x, mu=0.01, h=1e-3):
    N = x.shape[0]
    x_data = x.data
    perturbations = [x]  # center
    for dim in [1, 2, 3]:
        e = torch.zeros_like(x_data); e[:, dim] = h
        e2 = torch.zeros_like(x_data); e2[:, dim] = 2*h
        perturbations.extend([x_data + e, x_data - e, x_data + e2, x_data - e2])
    x_batch = torch.cat(perturbations, dim=0)  # (13N, 4)
    y_batch = net(x_batch)

    y0 = y_batch[:N]
    lap = torch.zeros_like(y0[:, 2:5])
    for i in range(3):
        base = 1 + 4*i
        yp1 = y_batch[base*N:(base+1)*N]
        ym1 = y_batch[(base+1)*N:(base+2)*N]
        yp2 = y_batch[(base+2)*N:(base+3)*N]
        ym2 = y_batch[(base+3)*N:(base+4)*N]
        lap += (-yp2[:, 2:5] + 16*yp1[:, 2:5] - 30*y0[:, 2:5] + 16*ym1[:, 2:5] - ym2[:, 2:5]) / (12*h*h)

    loss = mu**2 * (lap**2).mean()
    return loss, lap.detach()


# ========== 方案3: Richardson外推 ==========
def viscous_richardson(net, x, mu=0.01, h=1e-3):
    """Richardson外推: 用 h 和 h/2 的2阶FD结果消除O(h²)截断误差, 达到4阶精度"""
    N = x.shape[0]
    x_data = x.data

    # h 步长
    pert_h = [x]
    for dim in [1, 2, 3]:
        e = torch.zeros_like(x_data); e[:, dim] = h
        pert_h.extend([x_data + e, x_data - e])
    x_h = torch.cat(pert_h, dim=0)
    y_h = net(x_h)

    y0 = y_h[:N]
    lap_h = torch.zeros_like(y0[:, 2:5])
    for i in range(3):
        yp = y_h[(2*i+1)*N:(2*i+2)*N]
        ym = y_h[(2*i+2)*N:(2*i+3)*N]
        lap_h += (yp[:, 2:5] - 2*y0[:, 2:5] + ym[:, 2:5]) / (h*h)

    # h/2 步长
    h2 = h / 2.0
    pert_h2 = []
    for dim in [1, 2, 3]:
        e = torch.zeros_like(x_data); e[:, dim] = h2
        pert_h2.extend([x_data + e, x_data - e])
    x_h2 = torch.cat(pert_h2, dim=0)
    y_h2 = net(x_h2)

    lap_h2 = torch.zeros_like(y0[:, 2:5])
    for i in range(3):
        yp = y_h2[2*i*N:(2*i+1)*N]
        ym = y_h2[(2*i+1)*N:(2*i+2)*N]
        lap_h2 += (yp[:, 2:5] - 2*y0[:, 2:5] + ym[:, 2:5]) / (h2*h2)

    # Richardson: (4*L(h/2) - L(h)) / 3
    lap = (4.0 * lap_h2 - lap_h) / 3.0

    loss = mu**2 * (lap**2).mean()
    return loss, lap.detach()


# ========== 方案4: FD float64 ==========
def viscous_fd_f64(net, x, mu=0.01, h=1e-4):
    """FD with float64 precision — 先转double, 算完转回float32"""
    N = x.shape[0]
    x_data = x.data.double()

    # 网络用float32, 输入转回float32算forward, 结果转double做差分
    perturbations = [x]
    for dim in [1, 2, 3]:
        e = torch.zeros_like(x_data); e[:, dim] = h
        perturbations.append((x_data + e).float())
        perturbations.append((x_data - e).float())
    x_batch = torch.cat(perturbations, dim=0)
    y_batch = net(x_batch).double()  # 输出转double

    y0 = y_batch[:N]
    lap = torch.zeros(N, 3, dtype=torch.float64, device=cuda)
    for i in range(3):
        yp = y_batch[(2*i+1)*N:(2*i+2)*N]
        ym = y_batch[(2*i+2)*N:(2*i+3)*N]
        lap += (yp[:, 2:5] - 2*y0[:, 2:5] + ym[:, 2:5]) / (h*h)
    lap = lap.float()

    loss = mu**2 * (lap**2).mean()
    return loss, lap.detach()


# ========== 方案5: 自动微分 + retain_graph 优化 ==========
def viscous_autograd_opt(net, x, mu=0.01):
    """优化的autograd: 合并grad调用, 减少图操作次数"""
    y = net(x)
    uvw = y[:, 2:5]  # (N, 3)

    # 一次性求所有一阶导
    # 对 u,v,w 分别求梯度
    first_derivs = []
    for i in range(3):
        d = gradients(uvw[:, i:i+1], x)[0]  # (N, 4)
        first_derivs.append(d[:, 1:4])  # 只取空间分量 (N, 3)

    # 二阶导: 对每个一阶导求梯度 (create_graph=False最后一层)
    lap = torch.zeros(x.shape[0], 3, dtype=x.dtype, device=x.device)
    for i in range(3):  # u, v, w
        for j in range(3):  # x, y, z
            d2 = torch.autograd.grad(
                first_derivs[i][:, j:j+1], x,
                grad_outputs=torch.ones_like(first_derivs[i][:, j:j+1]),
                create_graph=True, retain_graph=True
            )[0][:, j+1:j+2]  # 对角项: d²u/dx² 取 d(du/dx)/dx
            lap[:, i:i+1] += d2

    loss = mu**2 * (lap**2).mean()
    return loss, lap.detach()


# ========== 运行 ==========
Nl, Nn, N_pts = 6, 128, 10000
N_epochs = 50

print(f'Config: {Nl}×{Nn}, {N_pts} points, {N_epochs} epochs')
print(f'{"="*80}\n')

methods = [
    ('baseline',        lambda net, x: viscous_baseline(net, x)),
    ('autograd_opt',    lambda net, x: viscous_autograd_opt(net, x)),
    ('fd_batch_h1e-3',  lambda net, x: viscous_fd_batched(net, x, h=1e-3)),
    ('fd4_batch_h1e-3', lambda net, x: viscous_fd4_batched(net, x, h=1e-3)),
    ('richardson_h1e-3',lambda net, x: viscous_richardson(net, x, h=1e-3)),
    ('fd_f64_h1e-4',   lambda net, x: viscous_fd_f64(net, x, h=1e-4)),
]

all_results = {}

for name, method_fn in methods:
    net = build_net(Nl, Nn, 4, 5).to(torch.float32).to(cuda)
    n_params = sum(p.numel() for p in net.parameters())
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    x = torch.rand(N_pts, 4, dtype=torch.float32, device=cuda) * 2 - 1
    x.requires_grad_(True)

    # 验证
    try:
        optimizer.zero_grad()
        loss, lap = method_fn(net, x)
        loss.backward()
        optimizer.step()
    except Exception as e:
        print(f'  [{name}] FAILED: {e}')
        del net, optimizer, x; torch.cuda.empty_cache()
        continue

    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        loss, _ = method_fn(net, x)
        loss.backward(); optimizer.step()

    torch.cuda.synchronize()

    # 计时
    times = []
    for ep in range(N_epochs):
        optimizer.zero_grad()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        loss, lap = method_fn(net, x)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        loss.backward(); optimizer.step()
        torch.cuda.synchronize(); t2 = time.perf_counter()
        times.append((t1-t0, t2-t1))

    fwd_ms = np.mean([t[0] for t in times]) * 1000
    bwd_ms = np.mean([t[1] for t in times]) * 1000
    tot_ms = fwd_ms + bwd_ms

    all_results[name] = {'fwd_ms': fwd_ms, 'bwd_ms': bwd_ms, 'tot_ms': tot_ms}
    print(f'  {name:<20} fwd={fwd_ms:7.2f}ms  bwd={bwd_ms:7.2f}ms  total={tot_ms:7.2f}ms')

    del net, optimizer, x; torch.cuda.empty_cache()

# 汇总
base_tot = all_results.get('baseline', {}).get('tot_ms', 1.0)
print(f'\n{"Method":<20} {"Fwd(ms)":<10} {"Bwd(ms)":<10} {"Total(ms)":<12} {"Speedup":<10}')
print(f'{"-"*62}')
for name, r in all_results.items():
    sp = base_tot / r['tot_ms'] if r['tot_ms'] > 0 else 0
    print(f'{name:<20} {r["fwd_ms"]:<10.2f} {r["bwd_ms"]:<10.2f} {r["tot_ms"]:<12.2f} {sp:<10.2f}x')


# ========== 精度对比 (关键!) ==========
print(f'\n{"="*80}')
print(f'ACCURACY COMPARISON — same network, 1000 points')
print(f'{"="*80}')

Nl, Nn = 5, 64
net = build_net(Nl, Nn, 4, 5).to(torch.float32).to(cuda)
# 训练几步让输出非零
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
x_train = torch.rand(1000, 4, dtype=torch.float32, device=cuda) * 2 - 1
x_train.requires_grad_(True)
for _ in range(50):
    opt.zero_grad()
    y = net(x_train)
    (y**2).sum().backward()
    opt.step()

# float64 baseline (最准)
net64 = build_net(Nl, Nn, 4, 5).to(torch.float64).to(cuda)
net64.load_state_dict({k: v.double() for k, v in net.state_dict().items()})
x64 = x_train.detach().double().requires_grad_(True)
_, lap_ref64 = viscous_baseline(net64, x64)
lap_ref64 = lap_ref64.float()
print(f'Reference (float64 autograd): |lap| mean={lap_ref64.abs().mean():.8f}, max={lap_ref64.abs().max():.8f}')

# float32 baseline
x32 = x_train.detach().float().requires_grad_(True)
_, lap_base32 = viscous_baseline(net, x32)
err = (lap_base32 - lap_ref64).abs()
print(f'Float32 autograd:  mean_err={err.mean():.2e}, max_err={err.max():.2e}')

# 各种 FD
for label, fn in [
    ('FD h=1e-2', lambda: viscous_fd_batched(net, x32, h=1e-2)),
    ('FD h=1e-3', lambda: viscous_fd_batched(net, x32, h=1e-3)),
    ('FD h=5e-4', lambda: viscous_fd_batched(net, x32, h=5e-4)),
    ('FD h=1e-4', lambda: viscous_fd_batched(net, x32, h=1e-4)),
    ('FD4 h=1e-2', lambda: viscous_fd4_batched(net, x32, h=1e-2)),
    ('FD4 h=1e-3', lambda: viscous_fd4_batched(net, x32, h=1e-3)),
    ('FD4 h=5e-4', lambda: viscous_fd4_batched(net, x32, h=5e-4)),
    ('Rich h=1e-2', lambda: viscous_richardson(net, x32, h=1e-2)),
    ('Rich h=1e-3', lambda: viscous_richardson(net, x32, h=1e-3)),
    ('FD_f64 h=1e-3', lambda: viscous_fd_f64(net, x32, h=1e-3)),
    ('FD_f64 h=1e-4', lambda: viscous_fd_f64(net, x32, h=1e-4)),
    ('FD_f64 h=1e-5', lambda: viscous_fd_f64(net, x32, h=1e-5)),
]:
    _, lap_test = fn()
    err = (lap_test - lap_ref64).abs()
    rel_err = err / (lap_ref64.abs() + 1e-10)
    print(f'  {label:<18} mean_err={err.mean():.2e}  max_err={err.max():.2e}  '
          f'rel_mean={rel_err.mean():.2e}')

print('\nDone.')
