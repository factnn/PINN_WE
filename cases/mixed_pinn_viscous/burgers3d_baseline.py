"""
3D Viscous Burgers Baseline PINN — 认真训练到收敛

方程: u_t + u·∇u = μ∇²u  (3D vector Burgers with viscosity)
  u_t + u*u_x + v*u_y + w*u_z = μ*(u_xx + u_yy + u_zz)
  v_t + u*v_x + v*v_y + w*v_z = μ*(v_xx + v_yy + v_zz)
  w_t + u*w_x + v*w_y + w*w_z = μ*(w_xx + w_yy + w_zz)

IC (Taylor-Green style):
  u(0,x,y,z) =  sin(x)cos(y)cos(z)
  v(0,x,y,z) = -cos(x)sin(y)cos(z)
  w(0,x,y,z) = 0

Domain: t∈[0,0.5], (x,y,z)∈[-π,π]³, periodic BC
μ = 0.01

验证: 对于小t, 非线性弱, 解近似 u(t)≈u(0)*exp(-μt).
      (精确对于Taylor-Green incompressible Stokes: u=u0*exp(-3μt))
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

# ========== 参数 ==========
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
parser.add_argument('--tag', type=str, default='baseline')
args = parser.parse_args()

cuda = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
print(f'Device: {cuda}, dtype: {dtype}')

mu = args.mu
T_end = args.T_end

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(output_dir, exist_ok=True)


# ========== 网络 ==========
def build_net(Nl, Nn, n_in=4, n_out=3):
    layers = [nn.Linear(n_in, Nn), nn.Tanh()]
    for _ in range(Nl - 2):
        layers += [nn.Linear(Nn, Nn), nn.Tanh()]
    layers += [nn.Linear(Nn, n_out)]
    return nn.Sequential(*layers)


def gradients(out, inp):
    return torch.autograd.grad(out, inp, grad_outputs=torch.ones_like(out),
                               create_graph=True)


# ========== IC ==========
def exact_ic(x):
    """Taylor-Green style IC"""
    u = torch.sin(x[:, 1]) * torch.cos(x[:, 2]) * torch.cos(x[:, 3])
    v = -torch.cos(x[:, 1]) * torch.sin(x[:, 2]) * torch.cos(x[:, 3])
    w = torch.zeros_like(u)
    return torch.stack([u, v, w], dim=1)


def approx_solution(x, mu):
    """
    近似解: Taylor-Green Stokes 衰减 (忽略非线性).
    对于 incompressible TGV: u = u0 * exp(-3μt)
    这里3D Burgers不完全一样但短时间内相当接近.
    """
    t = x[:, 0:1]
    decay = torch.exp(-3 * mu * t)
    u0 = exact_ic(x)
    return u0 * decay


# ========== 采样 ==========
def sample_interior(N, T_end):
    x = torch.zeros(N, 4, dtype=dtype, device=cuda)
    x[:, 0] = torch.rand(N, device=cuda) * T_end
    x[:, 1] = torch.rand(N, device=cuda) * 2 * pi - pi
    x[:, 2] = torch.rand(N, device=cuda) * 2 * pi - pi
    x[:, 3] = torch.rand(N, device=cuda) * 2 * pi - pi
    x.requires_grad_(True)
    return x


def sample_ic(N):
    x = torch.zeros(N, 4, dtype=dtype, device=cuda)
    x[:, 1] = torch.rand(N, device=cuda) * 2 * pi - pi
    x[:, 2] = torch.rand(N, device=cuda) * 2 * pi - pi
    x[:, 3] = torch.rand(N, device=cuda) * 2 * pi - pi
    x.requires_grad_(True)
    return x


def sample_periodic_bc(N, T_end):
    """
    周期BC: u(-π,...) = u(π,...) 对每个空间维度.
    每个维度采N//3对点.
    """
    pairs = []
    N_per_dim = N // 3
    for dim in [1, 2, 3]:
        x_left = torch.zeros(N_per_dim, 4, dtype=dtype, device=cuda)
        x_left[:, 0] = torch.rand(N_per_dim, device=cuda) * T_end
        for d in [1, 2, 3]:
            if d == dim:
                x_left[:, d] = -pi
            else:
                x_left[:, d] = torch.rand(N_per_dim, device=cuda) * 2 * pi - pi

        x_right = x_left.clone()
        x_right[:, dim] = pi
        pairs.append((x_left, x_right))
    return pairs


# ========== Loss函数 ==========
def loss_pde(net, x_int):
    """Standard PINN: nested autograd for Laplacian"""
    y = net(x_int)
    u, v, w = y[:, 0:1], y[:, 1:2], y[:, 2:3]

    # 一阶导
    du = gradients(u, x_int)[0]
    u_t, u_x, u_y, u_z = du[:, 0:1], du[:, 1:2], du[:, 2:3], du[:, 3:4]
    dv = gradients(v, x_int)[0]
    v_t, v_x, v_y, v_z = dv[:, 0:1], dv[:, 1:2], dv[:, 2:3], dv[:, 3:4]
    dw = gradients(w, x_int)[0]
    w_t, w_x, w_y, w_z = dw[:, 0:1], dw[:, 1:2], dw[:, 2:3], dw[:, 3:4]

    # 二阶导 (nested autograd — the bottleneck!)
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

    return (res_u**2).mean() + (res_v**2).mean() + (res_w**2).mean()


def loss_ic(net, x_ic, ic_true):
    y = net(x_ic)
    return ((y - ic_true)**2).mean()


def loss_bc(net, bc_pairs):
    loss = torch.tensor(0.0, device=cuda)
    for x_left, x_right in bc_pairs:
        y_l = net(x_left)
        y_r = net(x_right)
        loss = loss + ((y_l - y_r)**2).mean()
    return loss


# ========== 训练 ==========
print(f'\n3D Viscous Burgers PINN Baseline')
print(f'Net: {args.Nl}x{args.Nn}, mu={mu}, T=[0,{T_end}]')
print(f'N_int={args.N_int}, N_ic={args.N_ic}, N_bc={args.N_bc}')

net = build_net(args.Nl, args.Nn).to(dtype).to(cuda)
n_params = sum(p.numel() for p in net.parameters())
print(f'Parameters: {n_params}')

# 固定采样点
x_int = sample_interior(args.N_int, T_end)
x_ic = sample_ic(args.N_ic)
ic_true = exact_ic(x_ic).detach()
bc_pairs = sample_periodic_bc(args.N_bc, T_end)

# loss 权重
w_pde = 1.0
w_ic = 10.0
w_bc = 5.0

# --- Adam ---
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.adam_epochs, eta_min=1e-5)

history = {'total': [], 'pde': [], 'ic': [], 'bc': [], 'ms_per_ep': []}
t_start = time.time()

print(f'\n===== Adam: {args.adam_epochs} epochs =====')
for ep in range(1, args.adam_epochs + 1):
    optimizer.zero_grad()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    l_pde = loss_pde(net, x_int)
    l_ic = loss_ic(net, x_ic, ic_true)
    l_bc = loss_bc(net, bc_pairs)
    loss = w_pde * l_pde + w_ic * l_ic + w_bc * l_bc

    loss.backward()
    optimizer.step()
    scheduler.step()

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    history['total'].append(loss.item())
    history['pde'].append(l_pde.item())
    history['ic'].append(l_ic.item())
    history['bc'].append(l_bc.item())
    history['ms_per_ep'].append((t1 - t0) * 1000)

    if ep % 1000 == 0 or ep == 1:
        avg_ms = np.mean(history['ms_per_ep'][-100:])
        print(f'  ep={ep:6d}  loss={loss.item():.4e}  pde={l_pde.item():.4e}  '
              f'ic={l_ic.item():.4e}  bc={l_bc.item():.4e}  {avg_ms:.1f}ms/ep')

    # 每3000 epoch重新采样
    if ep % 3000 == 0 and ep < args.adam_epochs:
        x_int = sample_interior(args.N_int, T_end)
        x_ic = sample_ic(args.N_ic)
        ic_true = exact_ic(x_ic).detach()
        bc_pairs = sample_periodic_bc(args.N_bc, T_end)
        print(f'  [Resampled collocation points at epoch {ep}]')

adam_time = time.time() - t_start
print(f'Adam done: {adam_time:.1f}s')

# --- LBFGS ---
# 固定点用于LBFGS
x_int_lbfgs = sample_interior(args.N_int, T_end)
x_ic_lbfgs = sample_ic(args.N_ic)
ic_true_lbfgs = exact_ic(x_ic_lbfgs).detach()
bc_pairs_lbfgs = sample_periodic_bc(args.N_bc, T_end)

optimizer_lbfgs = torch.optim.LBFGS(net.parameters(), lr=1.0,
                                     max_iter=20, line_search_fn='strong_wolfe')

print(f'\n===== LBFGS: {args.lbfgs_epochs} epochs =====')
t_lbfgs_start = time.time()

for ep in range(1, args.lbfgs_epochs + 1):
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    def closure():
        optimizer_lbfgs.zero_grad()
        l_pde = loss_pde(net, x_int_lbfgs)
        l_ic = loss_ic(net, x_ic_lbfgs, ic_true_lbfgs)
        l_bc = loss_bc(net, bc_pairs_lbfgs)
        loss = w_pde * l_pde + w_ic * l_ic + w_bc * l_bc
        loss.backward()
        return loss

    loss = optimizer_lbfgs.step(closure)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    if loss is not None:
        history['total'].append(loss.item())
        history['ms_per_ep'].append((t1 - t0) * 1000)

    if ep % 200 == 0 or ep == 1:
        # evaluate current losses
        with torch.no_grad():
            y_ic = net(x_ic_lbfgs)
            l_ic_val = ((y_ic - ic_true_lbfgs)**2).mean().item()
        loss_val = loss.item() if loss is not None else float('nan')
        avg_ms = np.mean(history['ms_per_ep'][-50:])
        print(f'  ep={ep:5d}  loss={loss_val:.4e}  {avg_ms:.1f}ms/ep')

lbfgs_time = time.time() - t_lbfgs_start
total_time = time.time() - t_start
print(f'LBFGS done: {lbfgs_time:.1f}s, Total: {total_time:.1f}s')


# ========== 验证 ==========
print(f'\n===== Validation =====')
net.eval()

# 在几个时间点验证
for t_val in [0.0, 0.1, 0.25, 0.5]:
    if t_val > T_end:
        continue
    N_val = 5000
    x_val = torch.zeros(N_val, 4, dtype=dtype, device=cuda)
    x_val[:, 0] = t_val
    x_val[:, 1] = torch.rand(N_val, device=cuda) * 2 * pi - pi
    x_val[:, 2] = torch.rand(N_val, device=cuda) * 2 * pi - pi
    x_val[:, 3] = torch.rand(N_val, device=cuda) * 2 * pi - pi

    with torch.no_grad():
        y_pred = net(x_val)
        y_approx = approx_solution(x_val, mu)

    err = (y_pred - y_approx).abs()
    rel_err = err / (y_approx.abs() + 1e-8)
    print(f'  t={t_val:.2f}: |u_pred - u_approx| mean={err[:, 0].mean():.4e}, '
          f'max={err[:, 0].max():.4e}, rel_mean={rel_err[:, 0].mean():.4e}')

# IC验证
x_ic_val = sample_ic(5000)
with torch.no_grad():
    y_ic_pred = net(x_ic_val)
    y_ic_exact = exact_ic(x_ic_val)
    ic_err = (y_ic_pred - y_ic_exact).abs()
print(f'\n  IC error: mean={ic_err.mean():.4e}, max={ic_err.max():.4e}')


# ========== 绘图 ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss vs epoch
ax = axes[0, 0]
ax.semilogy(history['total'], alpha=0.6, label='total')
ax.semilogy(history['pde'], alpha=0.6, label='pde')
ax.semilogy(history['ic'], alpha=0.6, label='ic')
ax.semilogy(history['bc'], alpha=0.6, label='bc')
ax.axvline(x=args.adam_epochs, color='k', linestyle='--', alpha=0.3, label='Adam→LBFGS')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('Training Loss'); ax.legend(); ax.grid(True, alpha=0.3)

# ms/epoch
ax = axes[0, 1]
ax.plot(history['ms_per_ep'], alpha=0.3)
# moving average
window = 100
if len(history['ms_per_ep']) > window:
    ma = np.convolve(history['ms_per_ep'], np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, window-1+len(ma)), ma, 'r-', linewidth=2)
ax.axvline(x=args.adam_epochs, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Epoch'); ax.set_ylabel('ms/epoch')
ax.set_title(f'Timing (avg Adam: {np.mean(history["ms_per_ep"][:args.adam_epochs]):.1f}ms)')
ax.grid(True, alpha=0.3)

# 解场 at t=0 (z=0 slice)
ax = axes[1, 0]
Ng = 80
xs = np.linspace(-pi, pi, Ng)
ys = np.linspace(-pi, pi, Ng)
X, Y = np.meshgrid(xs, ys)
x_grid = torch.zeros(Ng*Ng, 4, dtype=dtype, device=cuda)
x_grid[:, 0] = 0.0
x_grid[:, 1] = torch.tensor(X.flatten(), dtype=dtype, device=cuda)
x_grid[:, 2] = torch.tensor(Y.flatten(), dtype=dtype, device=cuda)
x_grid[:, 3] = 0.0

with torch.no_grad():
    u_pred = net(x_grid)[:, 0].cpu().numpy().reshape(Ng, Ng)
    u_exact = exact_ic(x_grid)[:, 0].cpu().numpy().reshape(Ng, Ng)

cf = ax.contourf(X, Y, u_pred, levels=30, cmap='RdBu_r')
plt.colorbar(cf, ax=ax)
ax.set_title('u(t=0, x, y, z=0) — PINN')
ax.set_xlabel('x'); ax.set_ylabel('y')

# 解场 at t=T_end/2
ax = axes[1, 1]
x_grid2 = x_grid.clone()
x_grid2[:, 0] = T_end / 2
with torch.no_grad():
    u_pred2 = net(x_grid2)[:, 0].cpu().numpy().reshape(Ng, Ng)
cf2 = ax.contourf(X, Y, u_pred2, levels=30, cmap='RdBu_r')
plt.colorbar(cf2, ax=ax)
ax.set_title(f'u(t={T_end/2:.2f}, x, y, z=0) — PINN')
ax.set_xlabel('x'); ax.set_ylabel('y')

fig.suptitle(f'3D Viscous Burgers Baseline — {args.Nl}×{args.Nn}, μ={mu}\n'
             f'Total: {total_time:.0f}s, Final loss: {history["total"][-1]:.2e}',
             fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, f'burgers3d_{args.tag}.png'), dpi=150)
plt.close(fig)

# 保存模型和history
torch.save({
    'net_state': net.state_dict(),
    'history': history,
    'args': vars(args),
    'total_time': total_time,
}, os.path.join(output_dir, f'burgers3d_{args.tag}.pt'))

avg_adam_ms = np.mean(history['ms_per_ep'][:args.adam_epochs])
print(f'\nSaved to {output_dir}/burgers3d_{args.tag}.png')
print(f'Avg ms/epoch (Adam): {avg_adam_ms:.1f}')
print(f'Final loss: {history["total"][-1]:.4e}')
print('Done!')
