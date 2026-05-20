"""
Forward-Mode AD 深度测试

搞清楚:
1. forward-mode 求一阶导是否真的比reverse-mode快
2. forward-over-reverse 求二阶导 vs nested reverse
3. 为什么之前benchmark没成功
4. 正确的torch.func用法
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.func import jvp, vjp, vmap, jacfwd, jacrev, functional_call
import functools

torch.manual_seed(42)
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
print(f'Device: {cuda}\n')

def to_numpy(t):
    return t.detach().cpu().numpy()

def build_net(Nl, Nn, n_in=4, n_out=5):
    layers = [nn.Linear(n_in, Nn), nn.Tanh()]
    for _ in range(Nl - 2):
        layers += [nn.Linear(Nn, Nn), nn.Tanh()]
    layers += [nn.Linear(Nn, n_out)]
    return nn.Sequential(*layers)

def gradients(out, inp):
    return torch.autograd.grad(out, inp, grad_outputs=torch.ones_like(out), create_graph=True)


# ========== Test 1: 一阶导 — reverse vs forward ==========
print('='*70)
print('Test 1: 一阶导 (Jacobian) — Reverse-mode vs Forward-mode')
print('='*70)

for Nl, Nn in [(5, 64), (6, 128)]:
    net = build_net(Nl, Nn).to(dtype).to(cuda)
    n_params = sum(p.numel() for p in net.parameters())
    N = 10000
    x = torch.rand(N, 4, dtype=dtype, device=cuda) * 2 - 1
    x.requires_grad_(True)

    params = dict(net.named_parameters())

    # ---- Method A: Standard reverse-mode autograd ----
    # 求 du/dx, du/dy, du/dz, dv/dx, ... (9个一阶空间偏导)
    torch.cuda.synchronize()
    times_rev = []
    for _ in range(50):
        torch.cuda.synchronize(); t0 = time.perf_counter()

        y = net(x)
        # 对每个输出分量 (u,v,w) 求一阶导
        all_grads = []
        for i in [2, 3, 4]:  # u, v, w
            g = torch.autograd.grad(y[:, i].sum(), x, create_graph=True)[0]
            all_grads.append(g[:, 1:4])  # 只取空间分量

        torch.cuda.synchronize(); t1 = time.perf_counter()
        times_rev.append(t1 - t0)

    rev_ms = np.mean(times_rev[5:]) * 1000  # skip warmup

    # ---- Method B: Forward-mode via jvp (per-sample) ----
    # 对每个空间方向, 一次 jvp 得到所有输出的方向导数
    def net_fn(xx):
        return net(xx)

    torch.cuda.synchronize()
    times_fwd = []
    for _ in range(50):
        torch.cuda.synchronize(); t0 = time.perf_counter()

        all_derivs = []
        for dim in [1, 2, 3]:  # x, y, z 方向
            tangent = torch.zeros_like(x)
            tangent[:, dim] = 1.0
            # jvp: 前向传播的同时计算 J @ tangent
            _, jvp_val = jvp(net_fn, (x.detach(),), (tangent,))
            all_derivs.append(jvp_val[:, 2:5])  # u,v,w 对该方向的导数

        torch.cuda.synchronize(); t1 = time.perf_counter()
        times_fwd.append(t1 - t0)

    fwd_ms = np.mean(times_fwd[5:]) * 1000

    # ---- Method C: jacrev (完整 Jacobian via reverse) ----
    def f_single(params_d, xi):
        return functional_call(net, params_d, xi.unsqueeze(0)).squeeze(0)

    torch.cuda.synchronize()
    times_jacrev = []
    N_small = min(N, 2000)
    x_small = x[:N_small].detach()
    for _ in range(20):
        torch.cuda.synchronize(); t0 = time.perf_counter()

        J = vmap(jacrev(functools.partial(f_single, params)))(x_small)  # (N, 5, 4)

        torch.cuda.synchronize(); t1 = time.perf_counter()
        times_jacrev.append(t1 - t0)

    jacrev_ms = np.mean(times_jacrev[5:]) * 1000 * (N / N_small)

    # ---- Method D: jacfwd (完整 Jacobian via forward) ----
    torch.cuda.synchronize()
    times_jacfwd = []
    for _ in range(20):
        torch.cuda.synchronize(); t0 = time.perf_counter()

        J = vmap(jacfwd(functools.partial(f_single, params)))(x_small)  # (N, 5, 4)

        torch.cuda.synchronize(); t1 = time.perf_counter()
        times_jacfwd.append(t1 - t0)

    jacfwd_ms = np.mean(times_jacfwd[5:]) * 1000 * (N / N_small)

    print(f'\n  Net: {Nl}x{Nn} ({n_params} params), N={N}')
    print(f'  Reverse-mode (autograd):  {rev_ms:7.2f} ms  ← 3次 backward (u,v,w各一次)')
    print(f'  Forward-mode (jvp):       {fwd_ms:7.2f} ms  ← 3次 forward (x,y,z各一次)')
    print(f'  vmap+jacrev (N={N_small}→{N}):  {jacrev_ms:7.2f} ms  ← per-sample reverse Jacobian')
    print(f'  vmap+jacfwd (N={N_small}→{N}):  {jacfwd_ms:7.2f} ms  ← per-sample forward Jacobian')
    print(f'  Forward/Reverse ratio: {fwd_ms/rev_ms:.2f}x')

    del net, x; torch.cuda.empty_cache()


# ========== Test 2: 二阶导 — nested reverse vs forward-over-reverse ==========
print(f'\n{"="*70}')
print('Test 2: 二阶导 (Laplacian) — nested reverse vs forward-over-reverse')
print('='*70)

for Nl, Nn in [(5, 64), (6, 128)]:
    net = build_net(Nl, Nn).to(dtype).to(cuda)
    N = 10000
    x = torch.rand(N, 4, dtype=dtype, device=cuda) * 2 - 1
    x.requires_grad_(True)

    # ---- Method A: Nested reverse (baseline) ----
    torch.cuda.synchronize()
    times_nested = []
    for _ in range(30):
        torch.cuda.synchronize(); t0 = time.perf_counter()

        y = net(x)
        lap = torch.zeros(N, 3, device=cuda)
        for i, ci in enumerate([2, 3, 4]):
            g = torch.autograd.grad(y[:, ci].sum(), x, create_graph=True)[0]
            for d in [1, 2, 3]:
                g2 = torch.autograd.grad(g[:, d].sum(), x, create_graph=True)[0][:, d:d+1]
                lap[:, i:i+1] += g2

        torch.cuda.synchronize(); t1 = time.perf_counter()
        times_nested.append(t1 - t0)

    nested_ms = np.mean(times_nested[5:]) * 1000

    # ---- Method C: 纯 torch.func — jacfwd(jacrev(f)) ----
    params = dict(net.named_parameters())

    def f_single(params_d, xi):
        return functional_call(net, params_d, xi.unsqueeze(0)).squeeze(0)

    N_small = min(N, 1000)
    x_small = x[:N_small].detach()

    # Hessian diagonal via jacfwd(jacrev(...))
    torch.cuda.synchronize()
    times_hess = []
    for _ in range(10):
        torch.cuda.synchronize(); t0 = time.perf_counter()

        def hess_diag(xi):
            """计算单点的 Hessian 对角线元素"""
            H = jacfwd(jacrev(functools.partial(f_single, params)))(xi)  # (5, 4, 4)
            # Laplacian = sum of diagonal spatial elements (pure functional)
            lap0 = H[2, 1, 1] + H[2, 2, 2] + H[2, 3, 3]
            lap1 = H[3, 1, 1] + H[3, 2, 2] + H[3, 3, 3]
            lap2 = H[4, 1, 1] + H[4, 2, 2] + H[4, 3, 3]
            return torch.stack([lap0, lap1, lap2])

        lap_func = vmap(hess_diag)(x_small)  # (N_small, 3)

        torch.cuda.synchronize(); t1 = time.perf_counter()
        times_hess.append(t1 - t0)

    hess_ms = np.mean(times_hess[3:]) * 1000 * (N / N_small)

    # ---- Method D: 只算对角 Hessian (forward-over-reverse, 不算交叉项) ----
    torch.cuda.synchronize()
    times_diag = []
    for _ in range(10):
        torch.cuda.synchronize(); t0 = time.perf_counter()

        def lap_single(xi):
            """单点 Laplacian via forward-over-reverse"""
            results = []
            for ci in [2, 3, 4]:
                def g_ci(xx, _ci=ci):
                    return f_single(params, xx)[_ci]

                grad_fn = torch.func.grad(g_ci)
                lap_i = torch.tensor(0.0, device=xi.device, dtype=xi.dtype)
                for d in [1, 2, 3]:
                    tangent = torch.zeros_like(xi)
                    tangent[d] = 1.0
                    _, hvp = jvp(grad_fn, (xi,), (tangent,))
                    lap_i = lap_i + hvp[d]
                results.append(lap_i)
            return torch.stack(results)

        lap_diag = vmap(lap_single)(x_small)

        torch.cuda.synchronize(); t1 = time.perf_counter()
        times_diag.append(t1 - t0)

    diag_ms = np.mean(times_diag[3:]) * 1000 * (N / N_small)

    print(f'\n  Net: {Nl}x{Nn}, N={N}')
    print(f'  Nested reverse (baseline):        {nested_ms:8.2f} ms')
    print(f'  jacfwd(jacrev) full Hessian:       {hess_ms:8.2f} ms  (scaled from N={N_small})')
    print(f'  forward-over-reverse diag only:    {diag_ms:8.2f} ms  (scaled from N={N_small})')
    print(f'  FD batched (for reference):        ~7.5 ms')

    del net, x; torch.cuda.empty_cache()


# ========== Test 3: 精度验证 ==========
print(f'\n{"="*70}')
print('Test 3: 精度验证 — forward-mode vs reverse-mode')
print('='*70)

Nl, Nn = 5, 64
net = build_net(Nl, Nn).to(dtype).to(cuda)
N = 100
x = torch.rand(N, 4, dtype=dtype, device=cuda) * 2 - 1
x.requires_grad_(True)
params = dict(net.named_parameters())

# Reference: nested reverse
y = net(x)
lap_rev = torch.zeros(N, 3, device=cuda)
for i, ci in enumerate([2, 3, 4]):
    g = torch.autograd.grad(y[:, ci].sum(), x, create_graph=True, retain_graph=True)[0]
    for d in [1, 2, 3]:
        g2 = torch.autograd.grad(g[:, d].sum(), x, create_graph=True, retain_graph=True)[0][:, d]
        lap_rev[:, i] += g2
lap_rev = lap_rev.detach()

# Forward-over-reverse
def f_single_fn(params_d, xi):
    return functional_call(net, params_d, xi.unsqueeze(0)).squeeze(0)

def lap_single_fn(xi):
    results = []
    for ci in [2, 3, 4]:
        def g_ci(xx, _ci=ci):
            return f_single_fn(params, xx)[_ci]
        grad_fn = torch.func.grad(g_ci)
        lap_i = torch.tensor(0.0, device=xi.device, dtype=xi.dtype)
        for d in [1, 2, 3]:
            tangent = torch.zeros_like(xi)
            tangent[d] = 1.0
            _, hvp = jvp(grad_fn, (xi,), (tangent,))
            lap_i = lap_i + hvp[d]
        results.append(lap_i)
    return torch.stack(results)

lap_fwd = vmap(lap_single_fn)(x.detach())

err = (lap_fwd - lap_rev).abs()
print(f'  Forward-over-reverse vs nested reverse:')
print(f'  mean_err = {err.mean():.2e}, max_err = {err.max():.2e}')
print(f'  (Should be ~0 — both are exact, just different AD modes)')

# jvp for first derivatives
tangent_x = torch.zeros_like(x.detach()); tangent_x[:, 1] = 1.0
_, jvp_x = jvp(lambda xx: net(xx), (x.detach(),), (tangent_x,))
# Compare with autograd
y2 = net(x)
g_u = torch.autograd.grad(y2[:, 2].sum(), x, create_graph=True)[0]
u_x_rev = g_u[:, 1].detach()
u_x_fwd = jvp_x[:, 2].detach()
err_1st = (u_x_fwd - u_x_rev).abs()
print(f'\n  First derivative (du/dx):')
print(f'  mean_err = {err_1st.mean():.2e}, max_err = {err_1st.max():.2e}')


# ========== Test 4: backward兼容性 ==========
print(f'\n{"="*70}')
print('Test 4: 能否 backward (训练兼容性)')
print('='*70)

net2 = build_net(5, 64).to(dtype).to(cuda)
opt = torch.optim.Adam(net2.parameters(), lr=1e-3)
x2 = torch.rand(100, 4, dtype=dtype, device=cuda) * 2 - 1

# jvp结果能否backward到网络参数？
try:
    opt.zero_grad()
    tangent = torch.zeros_like(x2); tangent[:, 1] = 1.0
    _, deriv = jvp(lambda xx: net2(xx), (x2,), (tangent,))
    loss = (deriv[:, 2]**2).mean()  # (du/dx)^2
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in net2.parameters() if p.grad is not None)
    print(f'  jvp → backward: SUCCESS (grad_norm={grad_norm:.4f})')
except Exception as e:
    print(f'  jvp → backward: FAILED ({e})')

# forward-over-reverse能否backward?
params2 = dict(net2.named_parameters())
try:
    opt.zero_grad()
    def f_s(xi):
        return functional_call(net2, params2, xi.unsqueeze(0)).squeeze(0)
    def lap_s(xi):
        def g(xx): return f_s(xx)[2]
        gf = torch.func.grad(g)
        lap_val = torch.tensor(0.0, device=xi.device)
        for d in [1, 2, 3]:
            t = torch.zeros(4, device=xi.device); t[d] = 1.0
            _, hv = jvp(gf, (xi,), (t,))
            lap_val = lap_val + hv[d]
        return lap_val
    x_small2 = x2[:10].detach()
    lap_out = vmap(lap_s)(x_small2)
    loss2 = (lap_out**2).mean()
    loss2.backward()
    grad_norm2 = sum(p.grad.norm().item() for p in net2.parameters() if p.grad is not None)
    print(f'  forward-over-reverse → backward: SUCCESS (grad_norm={grad_norm2:.4f})')
except Exception as e:
    print(f'  forward-over-reverse → backward: FAILED ({e})')

print('\nDone!')
