"""
Mixed PINN v2 — 优化版

关键优化:
1. 合并grad调用: 对u求一次grad → 同时得到 u_t(动量方程) 和 u_x,u_y,u_z(本构验证)
2. 动量方程中的时间导和本构验证共享同一次autograd调用
3. 只在辅助变量(σ)的空间导数上新增grad调用

grad调用次数对比:
  Baseline: 3(一阶) + 9(二阶nested) = 12次, 其中9次nested
  Mixed v1: 3(时间导) + 9(σ的空间导) + 3(本构验证) = 15次, 0次nested
  Mixed v2: 3(u/v/w全梯度, 含时间导+本构验证) + 9(σ的空间导) = 12次, 0次nested

  → 同样12次grad调用, 但Mixed v2全是一阶, 不需要create_graph嵌套!

进一步优化:
  - 本构loss不需要create_graph=True (因为σ和∂u/∂x的差可以直接backward)
  - 但动量方程中σ*u等乘积需要create_graph → 必须保留

实际上: 动量方程的grad仍需create_graph=True, 因为loss.backward()需要
  但计算图只有1层深, 不是baseline的2层嵌套!
"""
import sys, os, time, argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
pi = np.pi

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--Nl', type=int, default=6)
parser.add_argument('--Nn', type=int, default=128)
parser.add_argument('--N_int', type=int, default=15000)
parser.add_argument('--N_ic', type=int, default=5000)
parser.add_argument('--N_bc', type=int, default=3000)
parser.add_argument('--adam_epochs', type=int, default=10000)
parser.add_argument('--lbfgs_epochs', type=int, default=2000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--mu', type=float, default=0.01)
parser.add_argument('--T_end', type=float, default=0.5)
parser.add_argument('--w_pde', type=float, default=1.0)
parser.add_argument('--w_const', type=float, default=1.0)
parser.add_argument('--w_ic', type=float, default=10.0)
parser.add_argument('--w_bc', type=float, default=5.0)
parser.add_argument('--tag', type=str, default='mixed_v2')
args = parser.parse_args()

cuda = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
mu = args.mu
T_end = args.T_end
print(f'Device: {cuda}')

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(output_dir, exist_ok=True)


def build_net(Nl, Nn, n_in=4, n_out=12):
    layers = [nn.Linear(n_in, Nn), nn.Tanh()]
    for _ in range(Nl - 2):
        layers += [nn.Linear(Nn, Nn), nn.Tanh()]
    layers += [nn.Linear(Nn, n_out)]
    return nn.Sequential(*layers)


def grad_all(out, inp):
    """一次autograd得到out对inp所有分量的导数, create_graph=True"""
    return torch.autograd.grad(out, inp, grad_outputs=torch.ones_like(out),
                               create_graph=True)[0]


def exact_ic(x):
    u = torch.sin(x[:, 1]) * torch.cos(x[:, 2]) * torch.cos(x[:, 3])
    v = -torch.cos(x[:, 1]) * torch.sin(x[:, 2]) * torch.cos(x[:, 3])
    w = torch.zeros_like(u)
    return torch.stack([u, v, w], dim=1)


def exact_ic_grads(x):
    sx, cx = torch.sin(x[:, 1]), torch.cos(x[:, 1])
    sy, cy = torch.sin(x[:, 2]), torch.cos(x[:, 2])
    sz, cz = torch.sin(x[:, 3]), torch.cos(x[:, 3])
    return torch.stack([
        cx*cy*cz, -sx*sy*cz, -sx*cy*sz,       # u_x, u_y, u_z
        sx*sy*cz, -cx*cy*cz,  cx*sy*sz,        # v_x, v_y, v_z
        torch.zeros_like(sx), torch.zeros_like(sx), torch.zeros_like(sx)  # w_x, w_y, w_z
    ], dim=1)


def approx_solution(x, mu):
    t = x[:, 0:1]
    return exact_ic(x) * torch.exp(-3 * mu * t)


def sample_interior(N, T_end):
    x = torch.zeros(N, 4, dtype=dtype, device=cuda)
    x[:, 0] = torch.rand(N, device=cuda) * T_end
    x[:, 1] = torch.rand(N, device=cuda) * 2*pi - pi
    x[:, 2] = torch.rand(N, device=cuda) * 2*pi - pi
    x[:, 3] = torch.rand(N, device=cuda) * 2*pi - pi
    x.requires_grad_(True)
    return x


def sample_ic(N):
    x = torch.zeros(N, 4, dtype=dtype, device=cuda)
    x[:, 1] = torch.rand(N, device=cuda) * 2*pi - pi
    x[:, 2] = torch.rand(N, device=cuda) * 2*pi - pi
    x[:, 3] = torch.rand(N, device=cuda) * 2*pi - pi
    x.requires_grad_(True)
    return x


def sample_periodic_bc(N, T_end):
    pairs = []
    N_per_dim = N // 3
    for dim in [1, 2, 3]:
        x_left = torch.zeros(N_per_dim, 4, dtype=dtype, device=cuda)
        x_left[:, 0] = torch.rand(N_per_dim, device=cuda) * T_end
        for d in [1, 2, 3]:
            if d == dim:
                x_left[:, d] = -pi
            else:
                x_left[:, d] = torch.rand(N_per_dim, device=cuda) * 2*pi - pi
        x_right = x_left.clone()
        x_right[:, dim] = pi
        pairs.append((x_left, x_right))
    return pairs


def loss_mixed_pde_v2(net, x_int):
    """
    优化版Mixed PINN loss.

    关键: 对u/v/w各做一次grad_all → 同时得到:
      - 时间导 u_t (动量方程)
      - 空间导 u_x, u_y, u_z (本构关系验证)

    然后对σ做9次grad (一阶, 不嵌套) → 得到粘性项 ∂σ/∂x

    总计: 3 + 9 = 12次一阶grad (vs baseline的3 + 9次, 但baseline的9次是nested!)
    """
    y = net(x_int)

    # 网络输出
    u = y[:, 0:1]; v = y[:, 1:2]; w = y[:, 2:3]
    sig_ux = y[:, 3:4]; sig_uy = y[:, 4:5]; sig_uz = y[:, 5:6]
    sig_vx = y[:, 6:7]; sig_vy = y[:, 7:8]; sig_vz = y[:, 8:9]
    sig_wx = y[:, 9:10]; sig_wy = y[:, 10:11]; sig_wz = y[:, 11:12]

    # === 合并调用: 一次grad得到时间导+空间导 ===
    du = grad_all(u, x_int)  # (N, 4): [u_t, u_x, u_y, u_z]
    dv = grad_all(v, x_int)  # (N, 4): [v_t, v_x, v_y, v_z]
    dw = grad_all(w, x_int)  # (N, 4): [w_t, w_x, w_y, w_z]

    u_t = du[:, 0:1]; v_t = dv[:, 0:1]; w_t = dw[:, 0:1]

    # === 本构关系验证 (从同一次grad调用中取!) ===
    l_const = (
        (sig_ux - du[:, 1:2])**2 + (sig_uy - du[:, 2:3])**2 + (sig_uz - du[:, 3:4])**2 +
        (sig_vx - dv[:, 1:2])**2 + (sig_vy - dv[:, 2:3])**2 + (sig_vz - dv[:, 3:4])**2 +
        (sig_wx - dw[:, 1:2])**2 + (sig_wy - dw[:, 2:3])**2 + (sig_wz - dw[:, 3:4])**2
    ).mean()

    # === 粘性项: 对σ做一阶空间导 (9次, 不嵌套!) ===
    d_sig_ux = grad_all(sig_ux, x_int)
    d_sig_uy = grad_all(sig_uy, x_int)
    d_sig_uz = grad_all(sig_uz, x_int)
    lap_u = d_sig_ux[:, 1:2] + d_sig_uy[:, 2:3] + d_sig_uz[:, 3:4]

    d_sig_vx = grad_all(sig_vx, x_int)
    d_sig_vy = grad_all(sig_vy, x_int)
    d_sig_vz = grad_all(sig_vz, x_int)
    lap_v = d_sig_vx[:, 1:2] + d_sig_vy[:, 2:3] + d_sig_vz[:, 3:4]

    d_sig_wx = grad_all(sig_wx, x_int)
    d_sig_wy = grad_all(sig_wy, x_int)
    d_sig_wz = grad_all(sig_wz, x_int)
    lap_w = d_sig_wx[:, 1:2] + d_sig_wy[:, 2:3] + d_sig_wz[:, 3:4]

    # === 动量残差 ===
    res_u = u_t + u*sig_ux + v*sig_uy + w*sig_uz - mu*lap_u
    res_v = v_t + u*sig_vx + v*sig_vy + w*sig_vz - mu*lap_v
    res_w = w_t + u*sig_wx + v*sig_wy + w*sig_wz - mu*lap_w

    l_mom = (res_u**2).mean() + (res_v**2).mean() + (res_w**2).mean()

    return l_mom, l_const


def loss_ic(net, x_ic, ic_true, ic_grads_true):
    y = net(x_ic)
    l_vel = ((y[:, 0:3] - ic_true)**2).mean()
    l_grad = ((y[:, 3:12] - ic_grads_true)**2).mean()
    return l_vel + l_grad


def loss_bc(net, bc_pairs):
    loss = torch.tensor(0.0, device=cuda)
    for x_left, x_right in bc_pairs:
        y_l = net(x_left); y_r = net(x_right)
        loss = loss + ((y_l - y_r)**2).mean()
    return loss


# ========== 训练 ==========
print(f'\n3D Viscous Burgers Mixed PINN v2 (Optimized)')
print(f'Net: {args.Nl}x{args.Nn}, mu={mu}')

net = build_net(args.Nl, args.Nn).to(dtype).to(cuda)
n_params = sum(p.numel() for p in net.parameters())
print(f'Parameters: {n_params}')

x_int = sample_interior(args.N_int, T_end)
x_ic = sample_ic(args.N_ic)
ic_true = exact_ic(x_ic).detach()
ic_grads_true = exact_ic_grads(x_ic).detach()
bc_pairs = sample_periodic_bc(args.N_bc, T_end)

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.adam_epochs, eta_min=1e-5)

history = {'total': [], 'momentum': [], 'const': [], 'ic': [], 'bc': [], 'ms_per_ep': []}
t_start = time.time()

print(f'\n===== Adam: {args.adam_epochs} epochs =====')
for ep in range(1, args.adam_epochs + 1):
    optimizer.zero_grad()
    torch.cuda.synchronize(); t0 = time.perf_counter()

    l_mom, l_const = loss_mixed_pde_v2(net, x_int)
    l_ic = loss_ic(net, x_ic, ic_true, ic_grads_true)
    l_bc = loss_bc(net, bc_pairs)
    loss = args.w_pde * l_mom + args.w_const * l_const + args.w_ic * l_ic + args.w_bc * l_bc

    loss.backward()
    optimizer.step()
    scheduler.step()

    torch.cuda.synchronize(); t1 = time.perf_counter()

    history['total'].append(loss.item())
    history['momentum'].append(l_mom.item())
    history['const'].append(l_const.item())
    history['ic'].append(l_ic.item())
    history['bc'].append(l_bc.item())
    history['ms_per_ep'].append((t1 - t0) * 1000)

    if ep % 1000 == 0 or ep == 1:
        avg_ms = np.mean(history['ms_per_ep'][-100:])
        print(f'  ep={ep:6d}  loss={loss.item():.4e}  mom={l_mom.item():.4e}  '
              f'const={l_const.item():.4e}  ic={l_ic.item():.4e}  {avg_ms:.1f}ms/ep')

    if ep % 3000 == 0 and ep < args.adam_epochs:
        x_int = sample_interior(args.N_int, T_end)
        x_ic = sample_ic(args.N_ic)
        ic_true = exact_ic(x_ic).detach()
        ic_grads_true = exact_ic_grads(x_ic).detach()
        bc_pairs = sample_periodic_bc(args.N_bc, T_end)
        print(f'  [Resampled at epoch {ep}]')

adam_time = time.time() - t_start
print(f'Adam done: {adam_time:.1f}s')

# LBFGS
x_int_lbfgs = sample_interior(args.N_int, T_end)
x_ic_lbfgs = sample_ic(args.N_ic)
ic_true_lbfgs = exact_ic(x_ic_lbfgs).detach()
ic_grads_lbfgs = exact_ic_grads(x_ic_lbfgs).detach()
bc_pairs_lbfgs = sample_periodic_bc(args.N_bc, T_end)

optimizer_lbfgs = torch.optim.LBFGS(net.parameters(), lr=1.0,
                                     max_iter=20, line_search_fn='strong_wolfe')

print(f'\n===== LBFGS: {args.lbfgs_epochs} epochs =====')
t_lbfgs_start = time.time()
for ep in range(1, args.lbfgs_epochs + 1):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    def closure():
        optimizer_lbfgs.zero_grad()
        l_mom, l_const = loss_mixed_pde_v2(net, x_int_lbfgs)
        l_ic = loss_ic(net, x_ic_lbfgs, ic_true_lbfgs, ic_grads_lbfgs)
        l_bc = loss_bc(net, bc_pairs_lbfgs)
        loss = args.w_pde * l_mom + args.w_const * l_const + args.w_ic * l_ic + args.w_bc * l_bc
        loss.backward()
        return loss
    loss = optimizer_lbfgs.step(closure)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    if loss is not None:
        history['total'].append(loss.item())
        history['ms_per_ep'].append((t1 - t0) * 1000)
    if ep % 200 == 0 or ep == 1:
        loss_val = loss.item() if loss is not None else float('nan')
        avg_ms = np.mean(history['ms_per_ep'][-50:])
        print(f'  ep={ep:5d}  loss={loss_val:.4e}  {avg_ms:.1f}ms/ep')

total_time = time.time() - t_start
print(f'Total: {total_time:.1f}s')

# Validation
print(f'\n===== Validation =====')
net.eval()
for t_val in [0.0, 0.1, 0.25, 0.5]:
    if t_val > T_end: continue
    x_val = torch.zeros(5000, 4, dtype=dtype, device=cuda)
    x_val[:, 0] = t_val
    x_val[:, 1:] = torch.rand(5000, 3, device=cuda) * 2*pi - pi
    with torch.no_grad():
        y_pred = net(x_val)[:, 0:3]
        y_approx = approx_solution(x_val, mu)
    err = (y_pred - y_approx).abs()
    print(f'  t={t_val:.2f}: mean={err[:,0].mean():.4e}, max={err[:,0].max():.4e}')

x_ic_val = sample_ic(5000)
with torch.no_grad():
    ic_err = (net(x_ic_val)[:, 0:3] - exact_ic(x_ic_val)).abs()
print(f'  IC error: mean={ic_err.mean():.4e}, max={ic_err.max():.4e}')

# Save
avg_adam_ms = np.mean(history['ms_per_ep'][:args.adam_epochs])
torch.save({
    'net_state': net.state_dict(), 'history': history,
    'args': vars(args), 'total_time': total_time,
}, os.path.join(output_dir, f'burgers3d_{args.tag}.pt'))

print(f'\nAvg ms/epoch (Adam): {avg_adam_ms:.1f}')
print(f'Final loss: {history["total"][-1]:.4e}')
print('Done!')
