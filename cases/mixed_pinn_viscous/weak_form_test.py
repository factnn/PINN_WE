"""
弱形式 (积分形式) 正则化 for PINN

核心idea:
  主loss: 微分形式 (用FD/Hutchinson算Laplacian, 快)
  正则项: 弱形式 (只需一阶导, 保证守恒性)

弱形式推导 (3D viscous Burgers):
  强形式: u_t + u*u_x + v*u_y + w*u_z = μ*(u_xx + u_yy + u_zz)

  乘以 test function φ(x,y,z), 在体积V上积分:
  ∫_V (u_t + u*u_x + v*u_y + w*u_z) φ dV = μ ∫_V (u_xx + u_yy + u_zz) φ dV

  对右边分部积分:
  μ ∫_V ∇²u · φ dV = -μ ∫_V ∇u·∇φ dV + μ ∮_S (∇u·n)φ dS

  周期BC → 边界项消失 (左右互消)!

  最终:
  ∫_V (u_t + u*u_x + v*u_y + w*u_z) φ dV + μ ∫_V ∇u·∇φ dV = 0
                                                    ↑ 只需一阶导!

  Test function选择: 用简单的Fourier基 φ_k(x) = sin(kx)cos(ly)cos(mz)
  这样 ∇φ 也是解析的.

实验:
  1. 纯微分形式 (baseline autograd)
  2. 纯微分形式 (FD Laplacian)
  3. 微分形式 (FD) + 弱形式正则 (每10 epoch)
  4. 纯弱形式 (只用积分loss)
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
mu = 0.01
T_end = 0.5


def build_net(Nl=6, Nn=128, n_in=4, n_out=3):
    layers = [nn.Linear(n_in, Nn), nn.Tanh()]
    for _ in range(Nl - 2):
        layers += [nn.Linear(Nn, Nn), nn.Tanh()]
    layers += [nn.Linear(Nn, n_out)]
    return nn.Sequential(*layers)


def gradients(out, inp):
    return torch.autograd.grad(out, inp, grad_outputs=torch.ones_like(out),
                               create_graph=True)[0]


def exact_ic(x):
    u = torch.sin(x[:, 1]) * torch.cos(x[:, 2]) * torch.cos(x[:, 3])
    v = -torch.cos(x[:, 1]) * torch.sin(x[:, 2]) * torch.cos(x[:, 3])
    w = torch.zeros_like(u)
    return torch.stack([u, v, w], dim=1)


# ========== 采样 ==========
def sample_interior(N):
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


# ========== FD Laplacian ==========
def laplacian_fd(net, x, h=0.1):
    """FD batched Laplacian for spatial dims [1,2,3]"""
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
    lap = torch.zeros(N, 3, device=cuda, dtype=dtype)
    for i, dim in enumerate([1, 2, 3]):
        yp = y_batch[(2*i+1)*N:(2*i+2)*N]
        ym = y_batch[(2*i+2)*N:(2*i+3)*N]
        for j in range(3):
            lap[:, j] += (yp[:, j] - 2*y0[:, j] + ym[:, j]) / (h*h)
    return lap


# ========== 微分形式 Loss ==========
def loss_strong_autograd(net, x):
    """标准PINN: nested autograd"""
    y = net(x); u, v, w = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    du = gradients(u, x); u_t, u_x, u_y, u_z = du[:, 0:1], du[:, 1:2], du[:, 2:3], du[:, 3:4]
    dv = gradients(v, x); v_t, v_x, v_y, v_z = dv[:, 0:1], dv[:, 1:2], dv[:, 2:3], dv[:, 3:4]
    dw = gradients(w, x); w_t, w_x, w_y, w_z = dw[:, 0:1], dw[:, 1:2], dw[:, 2:3], dw[:, 3:4]
    u_xx = gradients(u_x, x)[:, 1:2]; u_yy = gradients(u_y, x)[:, 2:3]; u_zz = gradients(u_z, x)[:, 3:4]
    v_xx = gradients(v_x, x)[:, 1:2]; v_yy = gradients(v_y, x)[:, 2:3]; v_zz = gradients(v_z, x)[:, 3:4]
    w_xx = gradients(w_x, x)[:, 1:2]; w_yy = gradients(w_y, x)[:, 2:3]; w_zz = gradients(w_z, x)[:, 3:4]
    res_u = u_t + u*u_x + v*u_y + w*u_z - mu*(u_xx + u_yy + u_zz)
    res_v = v_t + u*v_x + v*v_y + w*v_z - mu*(v_xx + v_yy + v_zz)
    res_w = w_t + u*w_x + v*w_y + w*w_z - mu*(w_xx + w_yy + w_zz)
    return (res_u**2).mean() + (res_v**2).mean() + (res_w**2).mean()


def loss_strong_fd(net, x, h=0.1):
    """微分形式 + FD Laplacian"""
    y = net(x); u, v, w = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    du = gradients(u, x); u_t, u_x, u_y, u_z = du[:, 0:1], du[:, 1:2], du[:, 2:3], du[:, 3:4]
    dv = gradients(v, x); v_t, v_x, v_y, v_z = dv[:, 0:1], dv[:, 1:2], dv[:, 2:3], dv[:, 3:4]
    dw = gradients(w, x); w_t, w_x, w_y, w_z = dw[:, 0:1], dw[:, 1:2], dw[:, 2:3], dw[:, 3:4]
    lap = laplacian_fd(net, x, h=h)
    res_u = u_t + u*u_x + v*u_y + w*u_z - mu*lap[:, 0:1]
    res_v = v_t + u*v_x + v*v_y + w*v_z - mu*lap[:, 1:2]
    res_w = w_t + u*w_x + v*w_y + w*w_z - mu*lap[:, 2:3]
    return (res_u**2).mean() + (res_v**2).mean() + (res_w**2).mean()


# ========== 弱形式 Loss ==========
def loss_weak_form(net, x_quad, w_quad=None):
    """
    弱形式 loss — 只需一阶导!

    对于 u 方程:
    ∫ (u_t + u*u_x + v*u_y + w*u_z) * φ dV + μ ∫ (u_x*φ_x + u_y*φ_y + u_z*φ_z) dV = 0

    用Monte Carlo积分: ∫ f dV ≈ V/N * Σ f(x_i)

    Test functions: 用几个低阶Fourier mode
      φ_1 = sin(x)cos(y)cos(z)   (和IC同mode)
      φ_2 = cos(x)sin(y)cos(z)
      φ_3 = sin(2x)cos(y)cos(z)  (高一阶mode)
      φ_4 = 1 (constant — 总守恒)
    """
    N = x_quad.shape[0]
    V = T_end * (2*pi)**3  # 域体积

    y = net(x_quad)
    u, v, w = y[:, 0:1], y[:, 1:2], y[:, 2:3]

    # 一阶导 (不需要create_graph的嵌套!)
    du = gradients(u, x_quad)
    u_t, u_x, u_y, u_z = du[:, 0:1], du[:, 1:2], du[:, 2:3], du[:, 3:4]
    dv = gradients(v, x_quad)
    v_t, v_x, v_y, v_z = dv[:, 0:1], dv[:, 1:2], dv[:, 2:3], dv[:, 3:4]
    dw = gradients(w, x_quad)
    w_t, w_x, w_y, w_z = dw[:, 0:1], dw[:, 1:2], dw[:, 2:3], dw[:, 3:4]

    # 对流项
    conv_u = u_t + u*u_x + v*u_y + w*u_z
    conv_v = v_t + u*v_x + v*v_y + w*v_z
    conv_w = w_t + u*w_x + v*w_y + w*w_z

    t = x_quad[:, 0:1]
    xx = x_quad[:, 1:2]; yy = x_quad[:, 2:3]; zz = x_quad[:, 3:4]

    # Test functions and their gradients
    test_funcs = [
        # (φ, φ_x, φ_y, φ_z)
        (torch.sin(xx)*torch.cos(yy)*torch.cos(zz),
         torch.cos(xx)*torch.cos(yy)*torch.cos(zz),
         -torch.sin(xx)*torch.sin(yy)*torch.cos(zz),
         -torch.sin(xx)*torch.cos(yy)*torch.sin(zz)),

        (torch.cos(xx)*torch.sin(yy)*torch.cos(zz),
         -torch.sin(xx)*torch.sin(yy)*torch.cos(zz),
         torch.cos(xx)*torch.cos(yy)*torch.cos(zz),
         -torch.cos(xx)*torch.sin(yy)*torch.sin(zz)),

        (torch.sin(2*xx)*torch.cos(yy)*torch.cos(zz),
         2*torch.cos(2*xx)*torch.cos(yy)*torch.cos(zz),
         -torch.sin(2*xx)*torch.sin(yy)*torch.cos(zz),
         -torch.sin(2*xx)*torch.cos(yy)*torch.sin(zz)),

        # constant test function → 总守恒
        (torch.ones_like(xx), torch.zeros_like(xx),
         torch.zeros_like(xx), torch.zeros_like(xx)),
    ]

    loss = torch.tensor(0.0, device=cuda)

    for phi, phi_x, phi_y, phi_z in test_funcs:
        # u方程弱形式: ∫ conv_u * φ dV + μ ∫ (u_x*φ_x + u_y*φ_y + u_z*φ_z) dV = 0
        integral_u = (conv_u * phi + mu * (u_x*phi_x + u_y*phi_y + u_z*phi_z)).mean()
        integral_v = (conv_v * phi + mu * (v_x*phi_x + v_y*phi_y + v_z*phi_z)).mean()
        integral_w = (conv_w * phi + mu * (w_x*phi_x + w_y*phi_y + w_z*phi_z)).mean()

        loss = loss + integral_u**2 + integral_v**2 + integral_w**2

    return loss


# ========== 训练框架 ==========
def train_method(name, loss_fn_strong, loss_fn_weak=None, weak_interval=10,
                 w_weak=0.1, N_epochs=3000):
    """
    训练PINN.
    loss_fn_strong: 每epoch都调用的微分形式loss
    loss_fn_weak: 每weak_interval epoch调用的弱形式正则 (None=不用)
    """
    torch.manual_seed(42)
    net = build_net().to(dtype).to(cuda)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_epochs, eta_min=1e-5)

    N_int = 10000
    N_ic = 3000
    x_int = sample_interior(N_int)
    x_ic = sample_ic(N_ic)
    ic_true = exact_ic(x_ic).detach()

    # 弱形式用的积分点 (可以和PDE点不同)
    x_quad = sample_interior(N_int) if loss_fn_weak else None

    losses = []
    pde_losses = []
    weak_losses = []
    times = []

    for ep in range(1, N_epochs + 1):
        optimizer.zero_grad()
        torch.cuda.synchronize(); t0 = time.perf_counter()

        l_pde = loss_fn_strong(net, x_int)
        l_ic = ((net(x_ic) - ic_true)**2).mean()
        loss = l_pde + 10 * l_ic

        # 弱形式正则
        l_weak = torch.tensor(0.0, device=cuda)
        if loss_fn_weak and ep % weak_interval == 0:
            l_weak = loss_fn_weak(net, x_quad)
            loss = loss + w_weak * l_weak

        loss.backward()
        optimizer.step()
        scheduler.step()

        torch.cuda.synchronize(); t1 = time.perf_counter()

        losses.append(loss.item())
        pde_losses.append(l_pde.item())
        weak_losses.append(l_weak.item() if isinstance(l_weak, torch.Tensor) else l_weak)
        times.append((t1 - t0) * 1000)

        if ep % 500 == 0 or ep == 1:
            avg_ms = np.mean(times[-100:])
            weak_str = f'  weak={l_weak.item():.4e}' if loss_fn_weak and ep % weak_interval == 0 else ''
            print(f'  [{name}] ep={ep:5d}  loss={loss.item():.4e}  '
                  f'pde={l_pde.item():.4e}{weak_str}  {avg_ms:.1f}ms/ep')

        if ep % 3000 == 0 and ep < N_epochs:
            x_int = sample_interior(N_int)
            x_ic = sample_ic(N_ic)
            ic_true = exact_ic(x_ic).detach()
            if x_quad is not None:
                x_quad = sample_interior(N_int)

    avg_ms = np.mean(times)
    print(f'  [{name}] Done: avg={avg_ms:.1f}ms/ep, final_loss={losses[-1]:.4e}')

    del net, optimizer
    torch.cuda.empty_cache()
    return {
        'losses': losses, 'pde': pde_losses, 'weak': weak_losses,
        'times': times, 'avg_ms': avg_ms
    }


# ========== 运行实验 ==========
N_epochs = 3000

print('='*70)
print('Experiment: Weak Form Regularization for PINN')
print('='*70)

results = {}

# 1. Baseline autograd
print('\n--- 1. Baseline (autograd nested) ---')
results['autograd'] = train_method('autograd', loss_strong_autograd, N_epochs=N_epochs)

# 2. FD only
print('\n--- 2. FD Laplacian only ---')
results['FD'] = train_method('FD', loss_strong_fd, N_epochs=N_epochs)

# 3. FD + weak form regularizer (every 10 epochs)
print('\n--- 3. FD + Weak Reg (every 10 ep) ---')
results['FD+weak10'] = train_method('FD+weak10', loss_strong_fd, loss_weak_form,
                                      weak_interval=10, w_weak=0.1, N_epochs=N_epochs)

# 4. FD + weak form regularizer (every 50 epochs)
print('\n--- 4. FD + Weak Reg (every 50 ep) ---')
results['FD+weak50'] = train_method('FD+weak50', loss_strong_fd, loss_weak_form,
                                      weak_interval=50, w_weak=0.5, N_epochs=N_epochs)

# 5. Pure weak form (no differential form at all)
print('\n--- 5. Pure Weak Form ---')
results['weak_only'] = train_method('weak_only',
                                     lambda net, x: loss_weak_form(net, x),
                                     N_epochs=N_epochs)


# ========== 绘图 ==========
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(output_dir, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss vs epoch
ax = axes[0]
for name, r in results.items():
    ax.semilogy(r['losses'], label=f'{name} ({r["avg_ms"]:.1f}ms)', alpha=0.7)
ax.set_xlabel('Epoch'); ax.set_ylabel('Total Loss')
ax.set_title('Loss vs Epoch'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Loss vs wall time
ax = axes[1]
for name, r in results.items():
    cum_t = np.cumsum(r['times']) / 1000
    ax.semilogy(cum_t, r['losses'], label=name, alpha=0.7)
ax.set_xlabel('Wall Time (s)'); ax.set_ylabel('Total Loss')
ax.set_title('Loss vs Wall Time'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# PDE loss vs epoch
ax = axes[2]
for name, r in results.items():
    ax.semilogy(r['pde'], label=name, alpha=0.7)
ax.set_xlabel('Epoch'); ax.set_ylabel('PDE Residual')
ax.set_title('PDE (Strong Form) Residual'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

fig.suptitle('Weak Form Regularization: FD + Integral Form\n'
             '3D Viscous Burgers, 6×128 network',
             fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'weak_form_comparison.png'), dpi=150)
plt.close()

# Summary
print(f'\n{"="*70}')
print('SUMMARY')
print('='*70)
base_ms = results['autograd']['avg_ms']
for name, r in results.items():
    sp = base_ms / r['avg_ms']
    print(f'{name:<15} {r["avg_ms"]:<8.1f}ms  {sp:.2f}x  final={r["losses"][-1]:.4e}')

print(f'\nPlot: {output_dir}/weak_form_comparison.png')
print('Done!')
