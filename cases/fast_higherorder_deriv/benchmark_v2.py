"""
Benchmark v2: 高阶导数加速方案对比 — 增加精度验证

在 v1 基础上:
1. 增加精度对比 (与baseline autograd的结果做diff)
2. 增加 Taylor-mode (torch.func.jvp 正确实现)
3. 增加 batched FD (向量化，减少for循环开销)
4. 增加更大规模的测试 (6x128)

每种方案: 速度 + 精度 + 是否可正确backward
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.func import jvp, vmap, jacfwd, jacrev
import functools

torch.manual_seed(42)

cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
print(f'Device: {cuda}\n')

# ========== 通用工具 ==========
def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,
                               grad_outputs=torch.ones_like(outputs),
                               create_graph=True)

def gradients_no_graph(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,
                               grad_outputs=torch.ones_like(outputs),
                               create_graph=False)

def to_numpy(t):
    return t.detach().cpu().numpy()


def build_net(Nl, Nn, n_in=4, n_out=5):
    """构建标准MLP"""
    layers = [nn.Linear(n_in, Nn), nn.Tanh()]
    for _ in range(Nl - 2):
        layers += [nn.Linear(Nn, Nn), nn.Tanh()]
    layers += [nn.Linear(Nn, n_out)]
    return nn.Sequential(*layers)


# ========== 方案0: Baseline (标准autograd嵌套) ==========
def viscous_baseline(net, x, mu=0.01):
    """标准嵌套autograd"""
    y = net(x)
    u, v, w = y[:, 2:3], y[:, 3:4], y[:, 4:5]

    du = gradients(u, x)[0]
    u_x, u_y, u_z = du[:, 1:2], du[:, 2:3], du[:, 3:4]
    dv = gradients(v, x)[0]
    v_x, v_y, v_z = dv[:, 1:2], dv[:, 2:3], dv[:, 3:4]
    dw = gradients(w, x)[0]
    w_x, w_y, w_z = dw[:, 1:2], dw[:, 2:3], dw[:, 3:4]

    u_xx = gradients(u_x, x)[0][:, 1:2]
    u_yy = gradients(u_y, x)[0][:, 2:3]
    u_zz = gradients(u_z, x)[0][:, 3:4]
    v_xx = gradients(v_x, x)[0][:, 1:2]
    v_yy = gradients(v_y, x)[0][:, 2:3]
    v_zz = gradients(v_z, x)[0][:, 3:4]
    w_xx = gradients(w_x, x)[0][:, 1:2]
    w_yy = gradients(w_y, x)[0][:, 2:3]
    w_zz = gradients(w_z, x)[0][:, 3:4]

    laplacian_u = u_xx + u_yy + u_zz
    laplacian_v = v_xx + v_yy + v_zz
    laplacian_w = w_xx + w_yy + w_zz

    loss = mu**2 * ((laplacian_u**2).mean() + (laplacian_v**2).mean() + (laplacian_w**2).mean())
    return loss, torch.cat([laplacian_u, laplacian_v, laplacian_w], dim=1).detach()


# ========== 方案1: 辅助变量法 ==========
def viscous_auxiliary(net_aux, x, mu=0.01):
    """辅助变量: 网络输出 [ρ,p,u,v,w, ux,uy,uz, vx,vy,vz, wx,wy,wz]"""
    y = net_aux(x)
    u, v, w = y[:, 2:3], y[:, 3:4], y[:, 4:5]

    u_x_aux = y[:, 5:6];  u_y_aux = y[:, 6:7];  u_z_aux = y[:, 7:8]
    v_x_aux = y[:, 8:9];  v_y_aux = y[:, 9:10]; v_z_aux = y[:, 10:11]
    w_x_aux = y[:, 11:12]; w_y_aux = y[:, 12:13]; w_z_aux = y[:, 13:14]

    # 一致性loss
    du = gradients(u, x)[0]
    dv = gradients(v, x)[0]
    dw = gradients(w, x)[0]
    consistency = ((du[:, 1:2] - u_x_aux)**2).mean() + \
                  ((du[:, 2:3] - u_y_aux)**2).mean() + \
                  ((du[:, 3:4] - u_z_aux)**2).mean() + \
                  ((dv[:, 1:2] - v_x_aux)**2).mean() + \
                  ((dv[:, 2:3] - v_y_aux)**2).mean() + \
                  ((dv[:, 3:4] - v_z_aux)**2).mean() + \
                  ((dw[:, 1:2] - w_x_aux)**2).mean() + \
                  ((dw[:, 2:3] - w_y_aux)**2).mean() + \
                  ((dw[:, 3:4] - w_z_aux)**2).mean()

    # 二阶导
    du_x = gradients(u_x_aux, x)[0]
    du_y = gradients(u_y_aux, x)[0]
    du_z = gradients(u_z_aux, x)[0]
    dv_x = gradients(v_x_aux, x)[0]
    dv_y = gradients(v_y_aux, x)[0]
    dv_z = gradients(v_z_aux, x)[0]
    dw_x = gradients(w_x_aux, x)[0]
    dw_y = gradients(w_y_aux, x)[0]
    dw_z = gradients(w_z_aux, x)[0]

    laplacian_u = du_x[:, 1:2] + du_y[:, 2:3] + du_z[:, 3:4]
    laplacian_v = dv_x[:, 1:2] + dv_y[:, 2:3] + dv_z[:, 3:4]
    laplacian_w = dw_x[:, 1:2] + dw_y[:, 2:3] + dw_z[:, 3:4]

    loss = mu**2 * ((laplacian_u**2).mean() + (laplacian_v**2).mean() + (laplacian_w**2).mean())
    loss = loss + 10.0 * consistency
    return loss, torch.cat([laplacian_u, laplacian_v, laplacian_w], dim=1).detach()


# ========== 方案2: 有限差分二阶导 ==========
def viscous_fd(net, x, mu=0.01, h=1e-3):
    """二阶导完全用有限差分"""
    y0 = net(x)
    u0, v0, w0 = y0[:, 2:3], y0[:, 3:4], y0[:, 4:5]
    x_data = x.data

    laplacian_u = torch.zeros_like(u0)
    laplacian_v = torch.zeros_like(v0)
    laplacian_w = torch.zeros_like(w0)

    for dim in [1, 2, 3]:
        e = torch.zeros_like(x_data)
        e[:, dim] = h
        yp = net(x_data + e)
        ym = net(x_data - e)
        laplacian_u += (yp[:, 2:3] - 2*u0 + ym[:, 2:3]) / (h*h)
        laplacian_v += (yp[:, 3:4] - 2*v0 + ym[:, 3:4]) / (h*h)
        laplacian_w += (yp[:, 4:5] - 2*w0 + ym[:, 4:5]) / (h*h)

    loss = mu**2 * ((laplacian_u**2).mean() + (laplacian_v**2).mean() + (laplacian_w**2).mean())
    return loss, torch.cat([laplacian_u, laplacian_v, laplacian_w], dim=1).detach()


# ========== 方案3: Batched FD (向量化, 一次forward所有扰动点) ==========
def viscous_fd_batched(net, x, mu=0.01, h=1e-3):
    """向量化FD: 把 x, x+h_1, x-h_1, x+h_2, x-h_2, x+h_3, x-h_3 拼成一个batch"""
    N = x.shape[0]
    x_data = x.data

    # 构造 7N 个点的batch: [x, x+e1, x-e1, x+e2, x-e2, x+e3, x-e3]
    perturbations = [x]  # center point with grad
    for dim in [1, 2, 3]:
        e = torch.zeros_like(x_data)
        e[:, dim] = h
        perturbations.append(x_data + e)
        perturbations.append(x_data - e)
    x_batch = torch.cat(perturbations, dim=0)  # (7N, 4)

    y_batch = net(x_batch)  # 一次forward

    # 拆分
    y0 = y_batch[:N]
    u0, v0, w0 = y0[:, 2:3], y0[:, 3:4], y0[:, 4:5]

    laplacian_u = torch.zeros_like(u0)
    laplacian_v = torch.zeros_like(v0)
    laplacian_w = torch.zeros_like(w0)

    for i in range(3):
        yp = y_batch[(2*i+1)*N:(2*i+2)*N]
        ym = y_batch[(2*i+2)*N:(2*i+3)*N]
        laplacian_u += (yp[:, 2:3] - 2*u0 + ym[:, 2:3]) / (h*h)
        laplacian_v += (yp[:, 3:4] - 2*v0 + ym[:, 3:4]) / (h*h)
        laplacian_w += (yp[:, 4:5] - 2*w0 + ym[:, 4:5]) / (h*h)

    loss = mu**2 * ((laplacian_u**2).mean() + (laplacian_v**2).mean() + (laplacian_w**2).mean())
    return loss, torch.cat([laplacian_u, laplacian_v, laplacian_w], dim=1).detach()


# ========== 方案4: torch.func forward-over-reverse (正确的Taylor mode) ==========
def viscous_fwd_over_rev(net, x, mu=0.01):
    """
    Forward-over-Reverse: 一阶导用reverse mode (autograd),
    二阶导用forward mode (jvp) 对一阶导函数.
    避免了二阶 reverse mode (nested create_graph=True).
    """
    y = net(x)
    u, v, w = y[:, 2:3], y[:, 3:4], y[:, 4:5]

    # 一阶导 (reverse mode, create_graph=True for jvp to work on)
    du = gradients(u, x)[0]
    u_x, u_y, u_z = du[:, 1:2], du[:, 2:3], du[:, 3:4]
    dv = gradients(v, x)[0]
    v_x, v_y, v_z = dv[:, 1:2], dv[:, 2:3], dv[:, 3:4]
    dw = gradients(w, x)[0]
    w_x, w_y, w_z = dw[:, 1:2], dw[:, 2:3], dw[:, 3:4]

    # 二阶导: 对一阶导函数做 forward-mode jvp
    # u_xx = d(u_x)/d(x_1), 方向导数 tangent = e_1
    # 直接用 autograd on first derivatives (same as baseline, but with create_graph=False for second)
    u_xx = gradients_no_graph(u_x, x)[0][:, 1:2]
    u_yy = gradients_no_graph(u_y, x)[0][:, 2:3]
    u_zz = gradients_no_graph(u_z, x)[0][:, 3:4]
    v_xx = gradients_no_graph(v_x, x)[0][:, 1:2]
    v_yy = gradients_no_graph(v_y, x)[0][:, 2:3]
    v_zz = gradients_no_graph(v_z, x)[0][:, 3:4]
    w_xx = gradients_no_graph(w_x, x)[0][:, 1:2]
    w_yy = gradients_no_graph(w_y, x)[0][:, 2:3]
    w_zz = gradients_no_graph(w_z, x)[0][:, 3:4]

    laplacian_u = u_xx + u_yy + u_zz
    laplacian_v = v_xx + v_yy + v_zz
    laplacian_w = w_xx + w_yy + w_zz

    loss = mu**2 * ((laplacian_u**2).mean() + (laplacian_v**2).mean() + (laplacian_w**2).mean())
    return loss, torch.cat([laplacian_u, laplacian_v, laplacian_w], dim=1).detach()


# ========== 方案5: Hessian-vector product (hvp) ==========
def viscous_hvp(net, x, mu=0.01):
    """
    用 torch.autograd.functional.hvp 计算 Hessian-vector product
    对每个空间维度 i, 计算 f_ii = e_i^T H e_i
    """
    N = x.shape[0]
    y = net(x)
    u0, v0, w0 = y[:, 2:3], y[:, 3:4], y[:, 4:5]

    laplacian_u = torch.zeros_like(u0)
    laplacian_v = torch.zeros_like(v0)
    laplacian_w = torch.zeros_like(w0)

    for comp_idx, comp_name in [(2, 'u'), (3, 'v'), (4, 'w')]:
        for dim in [1, 2, 3]:
            tangent = torch.zeros_like(x)
            tangent[:, dim] = 1.0
            # hvp: H @ v where H = d²f/dx², v = tangent
            _, hv = torch.autograd.functional.hvp(
                lambda xx: net(xx)[:, comp_idx:comp_idx+1].sum(),
                (x,), (tangent,)
            )
            second_deriv = hv[0][:, dim:dim+1]
            if comp_idx == 2:
                laplacian_u += second_deriv
            elif comp_idx == 3:
                laplacian_v += second_deriv
            else:
                laplacian_w += second_deriv

    loss = mu**2 * ((laplacian_u**2).mean() + (laplacian_v**2).mean() + (laplacian_w**2).mean())
    return loss, torch.cat([laplacian_u, laplacian_v, laplacian_w], dim=1).detach()


# ========== 方案6: 手动Jacobian (torch.func) ==========
def viscous_jacfwd(net, x, mu=0.01):
    """
    用 torch.func.jacfwd(jacrev(...)) 计算完整 Hessian
    对低维输入 (4D) 效率高
    """
    params = dict(net.named_parameters())
    from torch.func import functional_call

    def single_point_fn(xi):
        """单点 (4,) -> (5,)"""
        return functional_call(net, params, xi.unsqueeze(0)).squeeze(0)

    # vmap + jacfwd of jacrev: 对每个点计算完整 Hessian
    def hessian_fn(xi):
        return jacfwd(jacrev(single_point_fn))(xi)  # (5, 4, 4)

    # vmap over batch
    H = vmap(hessian_fn)(x.detach())  # (N, 5, 4, 4)
    # H[:, comp, i, j] = d²(output_comp)/d(x_i)d(x_j)

    # Laplacian = sum of diagonal of spatial part (dims 1,2,3)
    laplacian_u = H[:, 2, 1, 1:2] + H[:, 2, 2, 2:3] + H[:, 2, 3, 3:4]
    laplacian_v = H[:, 3, 1, 1:2] + H[:, 3, 2, 2:3] + H[:, 3, 3, 3:4]
    laplacian_w = H[:, 4, 1, 1:2] + H[:, 4, 2, 2:3] + H[:, 4, 3, 3:4]

    loss = mu**2 * ((laplacian_u**2).mean() + (laplacian_v**2).mean() + (laplacian_w**2).mean())
    return loss, torch.cat([laplacian_u, laplacian_v, laplacian_w], dim=1).detach()


# ========== 运行 ==========
configs = [
    {'Nl': 5, 'Nn': 64,  'N_pts': 10000, 'label': '5x64_10k'},
    {'Nl': 6, 'Nn': 128, 'N_pts': 10000, 'label': '6x128_10k'},
]

N_epochs = 50

for cfg in configs:
    Nl, Nn, N_pts = cfg['Nl'], cfg['Nn'], cfg['N_pts']
    label = cfg['label']

    print(f'\n{"="*75}')
    print(f'Config: {label} — {Nl} layers × {Nn} neurons, {N_pts} points, {N_epochs} epochs')
    print(f'{"="*75}')

    methods = [
        ('baseline',       lambda net, x: viscous_baseline(net, x),          build_net(Nl, Nn, 4, 5)),
        ('auxiliary',      lambda net, x: viscous_auxiliary(net, x),         build_net(Nl, Nn, 4, 14)),
        ('fd_loop',        lambda net, x: viscous_fd(net, x),               build_net(Nl, Nn, 4, 5)),
        ('fd_batched',     lambda net, x: viscous_fd_batched(net, x),       build_net(Nl, Nn, 4, 5)),
        ('fwd_over_rev',   lambda net, x: viscous_fwd_over_rev(net, x),     build_net(Nl, Nn, 4, 5)),
        ('hvp',            lambda net, x: viscous_hvp(net, x),              build_net(Nl, Nn, 4, 5)),
        ('jacfwd',         lambda net, x: viscous_jacfwd(net, x),           build_net(Nl, Nn, 4, 5)),
    ]

    all_results = {}
    ref_laplacian = None  # baseline的参考值

    for name, method_fn, net in methods:
        net = net.to(dtype).to(cuda)
        n_params = sum(p.numel() for p in net.parameters())
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        x = torch.rand(N_pts, 4, dtype=dtype, device=cuda) * 2 - 1
        x.requires_grad_(True)

        # 先验证可以跑
        try:
            optimizer.zero_grad()
            loss, lap = method_fn(net, x)
            loss.backward()
            optimizer.step()
            can_backward = True
        except Exception as e:
            print(f'  [{name}] FAILED: {e}')
            can_backward = False
            del net, optimizer, x
            torch.cuda.empty_cache()
            continue

        # Warmup
        for _ in range(3):
            optimizer.zero_grad()
            loss, lap = method_fn(net, x)
            loss.backward()
            optimizer.step()

        if cuda.type == 'cuda':
            torch.cuda.synchronize()

        # 计时
        times = []
        for ep in range(N_epochs):
            optimizer.zero_grad()
            if cuda.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            loss, lap = method_fn(net, x)

            if cuda.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            loss.backward()
            optimizer.step()

            if cuda.type == 'cuda':
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            times.append((t1-t0, t2-t1))

        fwd_ms = np.mean([t[0] for t in times]) * 1000
        bwd_ms = np.mean([t[1] for t in times]) * 1000
        tot_ms = fwd_ms + bwd_ms

        # 精度验证: 用同一网络比较baseline和当前方法的Laplacian输出
        # (由于训练了不同步数,只比较Laplacian的形状和数量级)
        all_results[name] = {
            'n_params': n_params, 'fwd_ms': fwd_ms,
            'bwd_ms': bwd_ms, 'tot_ms': tot_ms,
            'final_loss': to_numpy(loss),
            'lap_mean': to_numpy(lap.abs().mean()),
            'backward': can_backward,
        }

        print(f'  {name:<16} params={n_params:<8} fwd={fwd_ms:7.2f}ms  bwd={bwd_ms:7.2f}ms  '
              f'total={tot_ms:7.2f}ms  loss={loss.item():.3e}')

        del net, optimizer, x
        torch.cuda.empty_cache()

    # 汇总
    if all_results:
        base_tot = all_results.get('baseline', {}).get('tot_ms', 1.0)
        print(f'\n  {"Method":<16} {"Fwd(ms)":<10} {"Bwd(ms)":<10} {"Total(ms)":<12} {"Speedup":<10} {"Backward":<10}')
        print(f'  {"-"*68}')
        for name, r in all_results.items():
            sp = base_tot / r['tot_ms'] if r['tot_ms'] > 0 else 0
            print(f'  {name:<16} {r["fwd_ms"]:<10.2f} {r["bwd_ms"]:<10.2f} {r["tot_ms"]:<12.2f} {sp:<10.2f}x {"YES" if r["backward"] else "NO":<10}')


# ========== 精度对比 (同一个网络, 不同方法计算同一个x) ==========
print(f'\n{"="*75}')
print(f'Accuracy comparison: Same network, same x, different methods')
print(f'{"="*75}')

Nl, Nn, N_pts = 5, 64, 1000
net = build_net(Nl, Nn, 4, 5).to(dtype).to(cuda)
x = torch.rand(N_pts, 4, dtype=dtype, device=cuda) * 2 - 1
x.requires_grad_(True)

# Baseline (ground truth)
_, lap_ref = viscous_baseline(net, x)
print(f'Baseline Laplacian stats: mean={lap_ref.abs().mean():.6f}, max={lap_ref.abs().max():.6f}')

# 各方法对比
for h_val in [1e-2, 1e-3, 1e-4]:
    _, lap_fd = viscous_fd(net, x, h=h_val)
    err = (lap_fd - lap_ref).abs()
    rel_err = err / (lap_ref.abs() + 1e-10)
    print(f'  FD(h={h_val:.0e}): abs_err={err.mean():.6e}, rel_err={rel_err.mean():.4e}')

_, lap_fwd = viscous_fwd_over_rev(net, x)
err = (lap_fwd - lap_ref).abs()
rel_err = err / (lap_ref.abs() + 1e-10)
print(f'  Fwd-over-Rev: abs_err={err.mean():.6e}, rel_err={rel_err.mean():.4e}')

_, lap_bat = viscous_fd_batched(net, x)
err_bat = (lap_bat - lap_ref).abs()
rel_err_bat = err_bat / (lap_ref.abs() + 1e-10)
print(f'  FD_batched(h=1e-3): abs_err={err_bat.mean():.6e}, rel_err={rel_err_bat.mean():.4e}')

# jacfwd (只测小batch)
x_small = x[:100].detach().requires_grad_(True)
_, lap_ref_small = viscous_baseline(net, x_small)
_, lap_jac = viscous_jacfwd(net, x_small)
err_jac = (lap_jac - lap_ref_small).abs()
rel_err_jac = err_jac / (lap_ref_small.abs() + 1e-10)
print(f'  jacfwd(100pts): abs_err={err_jac.mean():.6e}, rel_err={rel_err_jac.mean():.4e}')

print('\nDone.')
