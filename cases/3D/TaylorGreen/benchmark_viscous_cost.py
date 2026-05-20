"""
Benchmark: 粘性项计算开销对比

同样的网络、同样的配点，分别跑100个epoch:
1. Euler (mu=0) — 只有一阶导
2. NS (mu=0.01) — 需要二阶导 (粘性应力+热传导)

计时对比，量化粘性项带来的额外开销。
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
pinn_src = os.path.join(repo_root, 'PINNsrc')
for p in [repo_root, pinn_src]:
    if p not in sys.path:
        sys.path.insert(0, p)
from PINNs import PINNs_WE_NS_3D

# ========== 参数 ==========
gamma = 1.4
rho0 = 1.0; V0 = 1.0; Ma = 0.1
c0 = V0 / Ma; p0 = rho0 * c0**2 / gamma
Re = 100.0; mu = rho0 * V0 / Re; Pr = 1.0

cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
print(f'Device: {cuda}')

# ========== 不同配置 ==========
configs = [
    {'Nl': 5, 'Nn': 64,  'N_int': 10000, 'label': '5x64_10k'},
    {'Nl': 6, 'Nn': 128, 'N_int': 10000, 'label': '6x128_10k'},
    {'Nl': 5, 'Nn': 64,  'N_int': 30000, 'label': '5x64_30k'},
    {'Nl': 6, 'Nn': 128, 'N_int': 30000, 'label': '6x128_30k'},
]

N_epochs = 100
import math
pi = math.pi

print(f'\n{"="*70}')
print(f'Benchmark: Euler vs NS (viscous), {N_epochs} epochs each')
print(f'{"="*70}')

results = []

for cfg in configs:
    Nl, Nn, N_int = cfg['Nl'], cfg['Nn'], cfg['N_int']
    label = cfg['label']
    n_params = None

    for mode, mu_val in [('Euler (mu=0)', 0.0), (f'NS (mu={mu:.4f})', mu)]:
        # 新建模型
        model = PINNs_WE_NS_3D(Nl=Nl, Nn=Nn, rho0=rho0, p0=p0, V0=V0).to(dtype).to(cuda)
        if n_params is None:
            n_params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 采样配点
        x_int = torch.rand(N_int, 4, dtype=dtype, device=cuda) * 2 - 1  # [-1,1]^4
        x_int.requires_grad_(True)

        # IC
        N_ic = 5000
        x_ic = torch.rand(N_ic, 4, dtype=dtype, device=cuda) * 2 - 1
        x_ic[:, 0] = 0.0
        x_ic.requires_grad_(True)
        # 简单IC: rho=rho0, p=p0, u=V0*sin, v=-V0*sin, w=0
        rho_ic = torch.full((N_ic,), rho0, dtype=dtype, device=cuda)
        p_ic = torch.full((N_ic,), p0, dtype=dtype, device=cuda)
        u_ic = V0 * torch.sin(x_ic[:, 1].detach())
        v_ic = -V0 * torch.sin(x_ic[:, 2].detach())
        w_ic = torch.zeros(N_ic, dtype=dtype, device=cuda)

        # Warmup (排除CUDA编译开销)
        for _ in range(3):
            optimizer.zero_grad()
            loss = model.loss_pde(x_int, mu=mu_val, gamma=gamma,
                                   rho_ref=rho0, p_ref=p0, V_ref=V0, Pr=Pr)
            loss += model.loss_ic(x_ic, rho_ic, u_ic, v_ic, w_ic, p_ic,
                                   rho_ref=rho0, p_ref=p0, V_ref=V0)
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()

        # 正式计时
        times_fwd = []
        times_bwd = []
        times_total = []

        for ep in range(N_epochs):
            optimizer.zero_grad()

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            loss_pde = model.loss_pde(x_int, mu=mu_val, gamma=gamma,
                                       rho_ref=rho0, p_ref=p0, V_ref=V0, Pr=Pr)
            loss_ic = model.loss_ic(x_ic, rho_ic, u_ic, v_ic, w_ic, p_ic,
                                     rho_ref=rho0, p_ref=p0, V_ref=V0)
            loss = loss_pde + loss_ic

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            loss.backward()

            torch.cuda.synchronize()
            t2 = time.perf_counter()

            optimizer.step()

            torch.cuda.synchronize()
            t3 = time.perf_counter()

            times_fwd.append(t1 - t0)
            times_bwd.append(t2 - t1)
            times_total.append(t3 - t0)

        fwd_ms = np.mean(times_fwd) * 1000
        bwd_ms = np.mean(times_bwd) * 1000
        tot_ms = np.mean(times_total) * 1000

        results.append({
            'label': label, 'mode': mode, 'n_params': n_params,
            'N_int': N_int, 'fwd_ms': fwd_ms, 'bwd_ms': bwd_ms, 'tot_ms': tot_ms
        })

        print(f'\n  [{label}] {mode}  ({n_params} params, {N_int} pts)')
        print(f'    Forward:  {fwd_ms:7.2f} ms/epoch  (loss_pde + loss_ic)')
        print(f'    Backward: {bwd_ms:7.2f} ms/epoch  (loss.backward)')
        print(f'    Total:    {tot_ms:7.2f} ms/epoch')

        del model, optimizer, x_int, x_ic
        torch.cuda.empty_cache()

# ========== 汇总对比 ==========
print(f'\n{"="*70}')
print(f'Summary: Viscous overhead')
print(f'{"="*70}')
print(f'{"Config":<15} {"Euler(ms)":<12} {"NS(ms)":<12} {"Overhead":<12} {"Ratio":<8}')
print(f'{"-"*59}')

for i in range(0, len(results), 2):
    euler = results[i]
    ns = results[i+1]
    overhead = ns['tot_ms'] - euler['tot_ms']
    ratio = ns['tot_ms'] / euler['tot_ms']
    print(f'{euler["label"]:<15} {euler["tot_ms"]:<12.2f} {ns["tot_ms"]:<12.2f} '
          f'{overhead:<12.2f} {ratio:<8.2f}x')

print(f'\nBreakdown (Forward / Backward):')
print(f'{"Config":<15} {"Euler fwd":<11} {"NS fwd":<11} {"Euler bwd":<11} {"NS bwd":<11}')
print(f'{"-"*59}')
for i in range(0, len(results), 2):
    e = results[i]; n = results[i+1]
    print(f'{e["label"]:<15} {e["fwd_ms"]:<11.2f} {n["fwd_ms"]:<11.2f} '
          f'{e["bwd_ms"]:<11.2f} {n["bwd_ms"]:<11.2f}')
