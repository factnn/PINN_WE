"""
3D Viscous Burgers — Mixed PINN (First-Order Formulation)

核心创新: 网络直接输出速度梯度作为辅助变量, 将二阶PDE降为一阶PDE组.
         彻底消灭 nested autograd (create_graph=True 的嵌套).

标准PINN:
  输出: (u, v, w)
  需要: u_xx, u_yy, u_zz, ... (9个二阶导, 需 nested autograd)

Mixed PINN:
  输出: (u, v, w, σ1..σ9)  where σ = 速度梯度张量
  σ1=u_x, σ2=u_y, σ3=u_z
  σ4=v_x, σ5=v_y, σ6=v_z
  σ7=w_x, σ8=w_y, σ9=w_z

  PDE (只需一阶导!):
    u_t + u*σ1 + v*σ2 + w*σ3 = μ*(∂σ1/∂x + ∂σ2/∂y + ∂σ3/∂z)
    v_t + u*σ4 + v*σ5 + w*σ6 = μ*(∂σ4/∂x + ∂σ5/∂y + ∂σ6/∂z)
    w_t + u*σ7 + v*σ8 + w*σ9 = μ*(∂σ7/∂x + ∂σ8/∂y + ∂σ9/∂z)

  本构约束 (也只需一阶导):
    σ1 = ∂u/∂x, σ2 = ∂u/∂y, ... (用autograd一阶导验证)

  所有导数都是一阶 → 不需要 create_graph=True 的嵌套!
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
parser.add_argument('--w_pde', type=float, default=1.0)
parser.add_argument('--w_const', type=float, default=1.0)
parser.add_argument('--w_ic', type=float, default=10.0)
parser.add_argument('--w_bc', type=float, default=5.0)
parser.add_argument('--tag', type=str, default='mixed')
args = parser.parse_args()

cuda = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
print(f'Device: {cuda}, dtype: {dtype}')

mu = args.mu
T_end = args.T_end

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(output_dir, exist_ok=True)


# ========== 网络 ==========
def build_net(Nl, Nn, n_in=4, n_out=12):
    """
    输出12个量:
    [0:3]  = u, v, w  (速度)
    [3:6]  = σ1, σ2, σ3 = u_x, u_y, u_z
    [6:9]  = σ4, σ5, σ6 = v_x, v_y, v_z
    [9:12] = σ7, σ8, σ9 = w_x, w_y, w_z
    """
    layers = [nn.Linear(n_in, Nn), nn.Tanh()]
    for _ in range(Nl - 2):
        layers += [nn.Linear(Nn, Nn), nn.Tanh()]
    layers += [nn.Linear(Nn, n_out)]
    return nn.Sequential(*layers)


def gradients_no_graph(out, inp):
    """一阶导, 不建高阶图 — 这是Mixed PINN的关键优势!"""
    return torch.autograd.grad(out, inp, grad_outputs=torch.ones_like(out),
                               create_graph=True)[0]
    # 注意: 仍然需要 create_graph=True, 因为loss.backward()需要梯度流到网络参数.
    # 但关键区别: 这里只有一层grad, 不需要对grad结果再求grad!
    # 计算图深度: 1层 vs baseline的2层嵌套.


# ========== IC ==========
def exact_ic(x):
    u = torch.sin(x[:, 1]) * torch.cos(x[:, 2]) * torch.cos(x[:, 3])
    v = -torch.cos(x[:, 1]) * torch.sin(x[:, 2]) * torch.cos(x[:, 3])
    w = torch.zeros_like(u)
    return torch.stack([u, v, w], dim=1)


def exact_ic_grads(x):
    """IC对应的速度梯度精确值"""
    sx = torch.sin(x[:, 1]); cx = torch.cos(x[:, 1])
    sy = torch.sin(x[:, 2]); cy = torch.cos(x[:, 2])
    sz = torch.sin(x[:, 3]); cz = torch.cos(x[:, 3])

    # u = sin(x)*cos(y)*cos(z)
    u_x = cx * cy * cz
    u_y = -sx * sy * cz
    u_z = -sx * cy * sz

    # v = -cos(x)*sin(y)*cos(z)
    v_x = sx * sy * cz
    v_y = -cx * cy * cz
    v_z = cx * sy * sz

    # w = 0
    w_x = torch.zeros_like(sx)
    w_y = torch.zeros_like(sx)
    w_z = torch.zeros_like(sx)

    return torch.stack([u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z], dim=1)


def approx_solution(x, mu):
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
def loss_mixed_pde(net, x_int):
    """
    Mixed PINN: PDE + 本构关系, 只需一阶导!

    关键: 不需要 nested autograd!
    - 动量方程中的粘性项 ∂σ/∂x 只是对辅助变量的一阶导
    - 本构关系 σ = ∂u/∂x 只是对速度的一阶导
    - 两种一阶导独立计算, 不嵌套
    """
    y = net(x_int)

    # 解包网络输出
    u = y[:, 0:1]; v = y[:, 1:2]; w = y[:, 2:3]
    sig_ux = y[:, 3:4]; sig_uy = y[:, 4:5]; sig_uz = y[:, 5:6]   # u的梯度
    sig_vx = y[:, 6:7]; sig_vy = y[:, 7:8]; sig_vz = y[:, 8:9]   # v的梯度
    sig_wx = y[:, 9:10]; sig_wy = y[:, 10:11]; sig_wz = y[:, 11:12]  # w的梯度

    # === 动量方程 (用辅助变量, 只需一阶导!) ===
    # u_t 仍然需要对u求时间导数
    du_dt = gradients_no_graph(u, x_int)[:, 0:1]
    dv_dt = gradients_no_graph(v, x_int)[:, 0:1]
    dw_dt = gradients_no_graph(w, x_int)[:, 0:1]

    # 粘性项: μ*(∂σ_ux/∂x + ∂σ_uy/∂y + ∂σ_uz/∂z) — 对辅助变量的一阶导
    d_sig_ux = gradients_no_graph(sig_ux, x_int)
    d_sig_uy = gradients_no_graph(sig_uy, x_int)
    d_sig_uz = gradients_no_graph(sig_uz, x_int)
    lap_u = d_sig_ux[:, 1:2] + d_sig_uy[:, 2:3] + d_sig_uz[:, 3:4]

    d_sig_vx = gradients_no_graph(sig_vx, x_int)
    d_sig_vy = gradients_no_graph(sig_vy, x_int)
    d_sig_vz = gradients_no_graph(sig_vz, x_int)
    lap_v = d_sig_vx[:, 1:2] + d_sig_vy[:, 2:3] + d_sig_vz[:, 3:4]

    d_sig_wx = gradients_no_graph(sig_wx, x_int)
    d_sig_wy = gradients_no_graph(sig_wy, x_int)
    d_sig_wz = gradients_no_graph(sig_wz, x_int)
    lap_w = d_sig_wx[:, 1:2] + d_sig_wy[:, 2:3] + d_sig_wz[:, 3:4]

    # 动量残差
    res_u = du_dt + u*sig_ux + v*sig_uy + w*sig_uz - mu*lap_u
    res_v = dv_dt + u*sig_vx + v*sig_vy + w*sig_vz - mu*lap_v
    res_w = dw_dt + u*sig_wx + v*sig_wy + w*sig_wz - mu*lap_w

    l_momentum = (res_u**2).mean() + (res_v**2).mean() + (res_w**2).mean()

    # === 本构关系 (也只需一阶导!) ===
    # σ_ux should = ∂u/∂x, σ_uy should = ∂u/∂y, etc.
    du = gradients_no_graph(u, x_int)  # (N, 4)
    dv = gradients_no_graph(v, x_int)
    dw = gradients_no_graph(w, x_int)

    l_const = (
        (sig_ux - du[:, 1:2])**2 + (sig_uy - du[:, 2:3])**2 + (sig_uz - du[:, 3:4])**2 +
        (sig_vx - dv[:, 1:2])**2 + (sig_vy - dv[:, 2:3])**2 + (sig_vz - dv[:, 3:4])**2 +
        (sig_wx - dw[:, 1:2])**2 + (sig_wy - dw[:, 2:3])**2 + (sig_wz - dw[:, 3:4])**2
    ).mean()

    return l_momentum, l_const


def loss_ic(net, x_ic, ic_true, ic_grads_true):
    y = net(x_ic)
    l_vel = ((y[:, 0:3] - ic_true)**2).mean()
    l_grad = ((y[:, 3:12] - ic_grads_true)**2).mean()
    return l_vel + l_grad


def loss_bc(net, bc_pairs):
    loss = torch.tensor(0.0, device=cuda)
    for x_left, x_right in bc_pairs:
        y_l = net(x_left)
        y_r = net(x_right)
        # 速度和辅助变量都要周期
        loss = loss + ((y_l - y_r)**2).mean()
    return loss


# ========== 训练 ==========
print(f'\n3D Viscous Burgers Mixed PINN (First-Order Formulation)')
print(f'Net: {args.Nl}x{args.Nn}, mu={mu}, T=[0,{T_end}]')
print(f'N_int={args.N_int}, N_ic={args.N_ic}, N_bc={args.N_bc}')
print(f'Weights: pde={args.w_pde}, const={args.w_const}, ic={args.w_ic}, bc={args.w_bc}')

net = build_net(args.Nl, args.Nn).to(dtype).to(cuda)
n_params = sum(p.numel() for p in net.parameters())
print(f'Parameters: {n_params} (output: 12 = 3 vel + 9 grad)')

# 固定采样点
x_int = sample_interior(args.N_int, T_end)
x_ic = sample_ic(args.N_ic)
ic_true = exact_ic(x_ic).detach()
ic_grads_true = exact_ic_grads(x_ic).detach()
bc_pairs = sample_periodic_bc(args.N_bc, T_end)

# --- Adam ---
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.adam_epochs, eta_min=1e-5)

history = {'total': [], 'momentum': [], 'const': [], 'ic': [], 'bc': [], 'ms_per_ep': []}
t_start = time.time()

print(f'\n===== Adam: {args.adam_epochs} epochs =====')
for ep in range(1, args.adam_epochs + 1):
    optimizer.zero_grad()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    l_mom, l_const = loss_mixed_pde(net, x_int)
    l_ic = loss_ic(net, x_ic, ic_true, ic_grads_true)
    l_bc = loss_bc(net, bc_pairs)
    loss = args.w_pde * l_mom + args.w_const * l_const + args.w_ic * l_ic + args.w_bc * l_bc

    loss.backward()
    optimizer.step()
    scheduler.step()

    torch.cuda.synchronize()
    t1 = time.perf_counter()

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

    # 重新采样
    if ep % 3000 == 0 and ep < args.adam_epochs:
        x_int = sample_interior(args.N_int, T_end)
        x_ic = sample_ic(args.N_ic)
        ic_true = exact_ic(x_ic).detach()
        ic_grads_true = exact_ic_grads(x_ic).detach()
        bc_pairs = sample_periodic_bc(args.N_bc, T_end)
        print(f'  [Resampled at epoch {ep}]')

adam_time = time.time() - t_start
print(f'Adam done: {adam_time:.1f}s')

# --- LBFGS ---
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
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    def closure():
        optimizer_lbfgs.zero_grad()
        l_mom, l_const = loss_mixed_pde(net, x_int_lbfgs)
        l_ic = loss_ic(net, x_ic_lbfgs, ic_true_lbfgs, ic_grads_lbfgs)
        l_bc = loss_bc(net, bc_pairs_lbfgs)
        loss = args.w_pde * l_mom + args.w_const * l_const + args.w_ic * l_ic + args.w_bc * l_bc
        loss.backward()
        return loss

    loss = optimizer_lbfgs.step(closure)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    if loss is not None:
        history['total'].append(loss.item())
        history['ms_per_ep'].append((t1 - t0) * 1000)

    if ep % 200 == 0 or ep == 1:
        loss_val = loss.item() if loss is not None else float('nan')
        avg_ms = np.mean(history['ms_per_ep'][-50:])
        print(f'  ep={ep:5d}  loss={loss_val:.4e}  {avg_ms:.1f}ms/ep')

lbfgs_time = time.time() - t_lbfgs_start
total_time = time.time() - t_start
print(f'LBFGS done: {lbfgs_time:.1f}s, Total: {total_time:.1f}s')


# ========== 验证 ==========
print(f'\n===== Validation =====')
net.eval()

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
        y_pred = net(x_val)[:, 0:3]
        y_approx = approx_solution(x_val, mu)

    err = (y_pred - y_approx).abs()
    print(f'  t={t_val:.2f}: |u_pred - u_approx| mean={err[:, 0].mean():.4e}, '
          f'max={err[:, 0].max():.4e}')

# IC
x_ic_val = sample_ic(5000)
with torch.no_grad():
    y_ic_pred = net(x_ic_val)[:, 0:3]
    y_ic_exact = exact_ic(x_ic_val)
    ic_err = (y_ic_pred - y_ic_exact).abs()
print(f'\n  IC error: mean={ic_err.mean():.4e}, max={ic_err.max():.4e}')


# ========== 绘图 ==========
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Loss vs epoch
ax = axes[0, 0]
ax.semilogy(history['total'], alpha=0.6, label='total')
ax.semilogy(history['momentum'], alpha=0.6, label='momentum')
ax.semilogy(history['const'], alpha=0.6, label='constitutive')
ax.semilogy(history['ic'], alpha=0.6, label='ic')
ax.semilogy(history['bc'], alpha=0.6, label='bc')
ax.axvline(x=args.adam_epochs, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('Training Loss'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ms/epoch
ax = axes[0, 1]
ax.plot(history['ms_per_ep'], alpha=0.3)
window = 100
if len(history['ms_per_ep']) > window:
    ma = np.convolve(history['ms_per_ep'], np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, window-1+len(ma)), ma, 'r-', linewidth=2)
ax.axvline(x=args.adam_epochs, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Epoch'); ax.set_ylabel('ms/epoch')
ax.set_title(f'Timing (avg Adam: {np.mean(history["ms_per_ep"][:args.adam_epochs]):.1f}ms)')
ax.grid(True, alpha=0.3)

# Constitutive loss (关键: 辅助变量是否学准了)
ax = axes[0, 2]
ax.semilogy(history['const'], alpha=0.8, color='orange')
ax.axvline(x=args.adam_epochs, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Epoch'); ax.set_ylabel('Constitutive Loss')
ax.set_title('Constitutive Relation Error\n(σ_ij should ≈ ∂u_i/∂x_j)')
ax.grid(True, alpha=0.3)

# 解场 at t=0
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
    y_pred_grid = net(x_grid)

ax = axes[1, 0]
u_pred = y_pred_grid[:, 0].cpu().numpy().reshape(Ng, Ng)
cf = ax.contourf(X, Y, u_pred, levels=30, cmap='RdBu_r')
plt.colorbar(cf, ax=ax)
ax.set_title('u(t=0, x, y, z=0)'); ax.set_xlabel('x'); ax.set_ylabel('y')

# 辅助变量 σ1=u_x at t=0
ax = axes[1, 1]
sig_ux_pred = y_pred_grid[:, 3].cpu().numpy().reshape(Ng, Ng)
u_x_exact = (torch.cos(x_grid[:, 1]) * torch.cos(x_grid[:, 2]) *
             torch.cos(x_grid[:, 3])).cpu().numpy().reshape(Ng, Ng)
cf = ax.contourf(X, Y, sig_ux_pred - u_x_exact, levels=30, cmap='RdBu_r')
plt.colorbar(cf, ax=ax)
ax.set_title('σ_ux - exact(u_x) at t=0\n(should be ~0)')
ax.set_xlabel('x'); ax.set_ylabel('y')

# 解场 at t=T/2
ax = axes[1, 2]
x_grid2 = x_grid.clone()
x_grid2[:, 0] = T_end / 2
with torch.no_grad():
    u_pred2 = net(x_grid2)[:, 0].cpu().numpy().reshape(Ng, Ng)
cf2 = ax.contourf(X, Y, u_pred2, levels=30, cmap='RdBu_r')
plt.colorbar(cf2, ax=ax)
ax.set_title(f'u(t={T_end/2:.2f}, x, y, z=0)')
ax.set_xlabel('x'); ax.set_ylabel('y')

fig.suptitle(f'3D Viscous Burgers Mixed PINN — {args.Nl}×{args.Nn}, μ={mu}\n'
             f'Total: {total_time:.0f}s, Final loss: {history["total"][-1]:.2e}',
             fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, f'burgers3d_{args.tag}.png'), dpi=150)
plt.close(fig)

# 保存
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
