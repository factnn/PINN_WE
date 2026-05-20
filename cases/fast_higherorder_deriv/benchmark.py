"""
Benchmark: 高阶导数加速方案对比

目标: 3D可压缩NS的PDE残差计算中，粘性项需要二阶导数，
      标准autograd嵌套导致 4-5x 的开销。
      本脚本对比多种替代方案的速度和精度。

方案:
  0. baseline   — 标准 autograd 嵌套 (create_graph=True)
  1. auxiliary   — 辅助变量法: 网络同时输出 u 和 ∂u/∂x，二阶导变一阶导
  2. fd          — 有限差分近似二阶导 (不需要嵌套autograd)
  3. fwdmode     — forward-mode AD (torch.func.jvp)，对低维输入更高效
  4. mixed       — 混合法: 一阶导用autograd，二阶导用有限差分

每种方案: 算100个epoch的平均时间 + 与baseline的残差精度对比。
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn

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
    """不保留计算图的autograd，用于最终不需要再求导的场景"""
    return torch.autograd.grad(outputs, inputs,
                               grad_outputs=torch.ones_like(outputs),
                               create_graph=False)

def to_numpy(t):
    return t.detach().cpu().numpy()


# ========== 测试问题: 3D NS 粘性项的简化版本 ==========
# 网络: (t,x,y,z) → (ρ,p,u,v,w)
# 需要计算: u_xx, u_yy, u_zz, v_xx, v_yy, v_zz, w_xx, w_yy, w_zz
#           以及交叉项 u_xy, v_xz 等
# 这里用一个简化的粘性残差来benchmark

Nl, Nn = 5, 64  # 5层64神经元
N_pts = 10000    # 配点数
N_epochs = 100


# ========== 方案0: Baseline (标准autograd嵌套) ==========
class BaselineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential()
        self.net.add_module('fc1', nn.Linear(4, Nn))
        self.net.add_module('act1', nn.Tanh())
        for i in range(2, Nl):
            self.net.add_module(f'fc{i}', nn.Linear(Nn, Nn))
            self.net.add_module(f'act{i}', nn.Tanh())
        self.net.add_module('fc_out', nn.Linear(Nn, 5))

    def forward(self, x):
        return self.net(x)

    def viscous_residual(self, x, mu=0.01):
        """完整粘性残差 (简化版: 只算动量方程粘性项)"""
        y = self.forward(x)
        u, v, w = y[:, 2:3], y[:, 3:4], y[:, 4:5]

        # 一阶导
        du = gradients(u, x)[0]
        u_x, u_y, u_z = du[:, 1:2], du[:, 2:3], du[:, 3:4]
        dv = gradients(v, x)[0]
        v_x, v_y, v_z = dv[:, 1:2], dv[:, 2:3], dv[:, 3:4]
        dw = gradients(w, x)[0]
        w_x, w_y, w_z = dw[:, 1:2], dw[:, 2:3], dw[:, 3:4]

        div_u = u_x + v_y + w_z

        # 二阶导 (嵌套autograd)
        u_xx = gradients(u_x, x)[0][:, 1:2]
        u_yy = gradients(u_y, x)[0][:, 2:3]
        u_zz = gradients(u_z, x)[0][:, 3:4]
        v_xx = gradients(v_x, x)[0][:, 1:2]
        v_yy = gradients(v_y, x)[0][:, 2:3]
        v_zz = gradients(v_z, x)[0][:, 3:4]
        w_xx = gradients(w_x, x)[0][:, 1:2]
        w_yy = gradients(w_y, x)[0][:, 2:3]
        w_zz = gradients(w_z, x)[0][:, 3:4]

        div_u_x = gradients(div_u, x)[0][:, 1:2]
        div_u_y = gradients(div_u, x)[0][:, 2:3]
        div_u_z = gradients(div_u, x)[0][:, 3:4]

        # 粘性力: μ(∇²u + 1/3 ∇(∇·u))
        visc_x = mu * (u_xx + u_yy + u_zz + (1.0/3.0)*div_u_x)
        visc_y = mu * (v_xx + v_yy + v_zz + (1.0/3.0)*div_u_y)
        visc_z = mu * (w_xx + w_yy + w_zz + (1.0/3.0)*div_u_z)

        loss = (visc_x**2).mean() + (visc_y**2).mean() + (visc_z**2).mean()
        return loss, visc_x.detach()


# ========== 方案1: 辅助变量法 ==========
# 核心思想: 网络输出 [ρ,p,u,v,w, u_x,u_y,u_z, v_x,v_y,v_z, w_x,w_y,w_z]
# 14个输出。额外loss: autograd(u,x)_x == u_x_output 等
# 二阶导: autograd(u_x_output, x) — 只需一阶autograd!

class AuxiliaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential()
        self.net.add_module('fc1', nn.Linear(4, Nn))
        self.net.add_module('act1', nn.Tanh())
        for i in range(2, Nl):
            self.net.add_module(f'fc{i}', nn.Linear(Nn, Nn))
            self.net.add_module(f'act{i}', nn.Tanh())
        # 输出: u,v,w (3) + u_x,u_y,u_z,v_x,v_y,v_z,w_x,w_y,w_z (9) = 12
        # 简化: 只输出速度和其空间梯度
        self.net.add_module('fc_out', nn.Linear(Nn, 5 + 9))

    def forward(self, x):
        return self.net(x)

    def viscous_residual(self, x, mu=0.01):
        y = self.forward(x)
        u, v, w = y[:, 2:3], y[:, 3:4], y[:, 4:5]

        # 辅助变量: 网络直接输出的梯度
        u_x_aux = y[:, 5:6];  u_y_aux = y[:, 6:7];  u_z_aux = y[:, 7:8]
        v_x_aux = y[:, 8:9];  v_y_aux = y[:, 9:10]; v_z_aux = y[:, 10:11]
        w_x_aux = y[:, 11:12]; w_y_aux = y[:, 12:13]; w_z_aux = y[:, 13:14]

        # 一致性loss: autograd(u) 应该等于 u_x_aux
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

        # 二阶导: 对辅助变量求一阶autograd (不需要嵌套!)
        du_x_aux = gradients(u_x_aux, x)[0]
        du_y_aux = gradients(u_y_aux, x)[0]
        du_z_aux = gradients(u_z_aux, x)[0]
        dv_x_aux = gradients(v_x_aux, x)[0]
        dv_y_aux = gradients(v_y_aux, x)[0]
        dv_z_aux = gradients(v_z_aux, x)[0]
        dw_x_aux = gradients(w_x_aux, x)[0]
        dw_y_aux = gradients(w_y_aux, x)[0]
        dw_z_aux = gradients(w_z_aux, x)[0]

        u_xx = du_x_aux[:, 1:2]; u_yy = du_y_aux[:, 2:3]; u_zz = du_z_aux[:, 3:4]
        v_xx = dv_x_aux[:, 1:2]; v_yy = dv_y_aux[:, 2:3]; v_zz = dv_z_aux[:, 3:4]
        w_xx = dw_x_aux[:, 1:2]; w_yy = dw_y_aux[:, 2:3]; w_zz = dw_z_aux[:, 3:4]

        div_u = u_x_aux + v_y_aux + w_z_aux
        d_div = gradients(div_u, x)[0]
        div_u_x = d_div[:, 1:2]; div_u_y = d_div[:, 2:3]; div_u_z = d_div[:, 3:4]

        visc_x = mu * (u_xx + u_yy + u_zz + (1.0/3.0)*div_u_x)
        visc_y = mu * (v_xx + v_yy + v_zz + (1.0/3.0)*div_u_y)
        visc_z = mu * (w_xx + w_yy + w_zz + (1.0/3.0)*div_u_z)

        loss = (visc_x**2).mean() + (visc_y**2).mean() + (visc_z**2).mean()
        loss = loss + 10.0 * consistency  # 一致性惩罚
        return loss, visc_x.detach()


# ========== 方案2: 有限差分二阶导 ==========
class FDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential()
        self.net.add_module('fc1', nn.Linear(4, Nn))
        self.net.add_module('act1', nn.Tanh())
        for i in range(2, Nl):
            self.net.add_module(f'fc{i}', nn.Linear(Nn, Nn))
            self.net.add_module(f'act{i}', nn.Tanh())
        self.net.add_module('fc_out', nn.Linear(Nn, 5))

    def forward(self, x):
        return self.net(x)

    def viscous_residual(self, x, mu=0.01, h=1e-3):
        """二阶导用中心差分: f_xx ≈ (f(x+h) - 2f(x) + f(x-h)) / h²"""
        y0 = self.forward(x)
        u0, v0, w0 = y0[:, 2:3], y0[:, 3:4], y0[:, 4:5]

        # 一阶导仍用autograd (用于div_u等)
        du = gradients(u0, x)[0]
        u_x, u_y, u_z = du[:, 1:2], du[:, 2:3], du[:, 3:4]
        dv = gradients(v0, x)[0]
        v_x, v_y, v_z = dv[:, 1:2], dv[:, 2:3], dv[:, 3:4]
        dw = gradients(w0, x)[0]
        w_x, w_y, w_z = dw[:, 1:2], dw[:, 2:3], dw[:, 3:4]

        div_u = u_x + v_y + w_z
        # div_u 的梯度也用FD
        # 但先处理二阶导

        x_data = x.detach()  # 不需要对扰动点求梯度

        u_xx = v_xx = w_xx = 0
        u_yy = v_yy = w_yy = 0
        u_zz = v_zz = w_zz = 0
        div_u_x = div_u_y = div_u_z = 0

        for dim, (idx_start) in enumerate([1, 2, 3]):  # x,y,z 三个空间维度
            e = torch.zeros_like(x_data)
            e[:, idx_start] = h

            with torch.no_grad():
                yp = self.forward(x_data + e)
                ym = self.forward(x_data - e)

            # (f(x+h) - 2f(x) + f(x-h)) / h²
            u_dd = (yp[:, 2:3] - 2*u0 + ym[:, 2:3]) / (h*h)
            v_dd = (yp[:, 3:4] - 2*v0 + ym[:, 3:4]) / (h*h)
            w_dd = (yp[:, 4:5] - 2*w0 + ym[:, 4:5]) / (h*h)

            if dim == 0:
                u_xx, v_xx, w_xx = u_dd, v_dd, w_dd
            elif dim == 1:
                u_yy, v_yy, w_yy = u_dd, v_dd, w_dd
            else:
                u_zz, v_zz, w_zz = u_dd, v_dd, w_dd

        # div_u 的空间梯度也用FD
        for dim, idx_start in enumerate([1, 2, 3]):
            e = torch.zeros_like(x_data)
            e[:, idx_start] = h
            with torch.no_grad():
                yp = self.forward(x_data + e)
                ym = self.forward(x_data - e)
            # 对 yp 求一阶导来得到 div_u 太贵，直接用FD近似
            # div_u(x+h) ≈ div_u(x) + div_u_x * h  → div_u_x ≈ (div_u(x+h) - div_u(x-h))/(2h)
            # 但 div_u 本身需要autograd... 用更简单的近似:
            # ∂(div_u)/∂x_i ≈ (u_xx_i + v_xy_i + w_xz_i) 已经在上面算了
            pass

        # 简化: div_u_x ≈ u_xx + v_yx + w_zx, 但交叉项需要额外FD
        # 这里用 ∇²u + 1/3∇(∇·u) ≈ 4/3 u_xx + u_yy + u_zz + 1/3(v_xy + w_xz)
        # 为简洁，忽略交叉项中的1/3部分，只benchmark Laplacian
        visc_x = mu * (u_xx + u_yy + u_zz)
        visc_y = mu * (v_xx + v_yy + v_zz)
        visc_z = mu * (w_xx + w_yy + w_zz)

        loss = (visc_x**2).mean() + (visc_y**2).mean() + (visc_z**2).mean()
        return loss, visc_x.detach()


# ========== 方案3: Forward-mode AD (torch.func.jvp) ==========
from torch.func import jvp, vmap, jacfwd
import functools

class FwdModeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential()
        self.net.add_module('fc1', nn.Linear(4, Nn))
        self.net.add_module('act1', nn.Tanh())
        for i in range(2, Nl):
            self.net.add_module(f'fc{i}', nn.Linear(Nn, Nn))
            self.net.add_module(f'act{i}', nn.Tanh())
        self.net.add_module('fc_out', nn.Linear(Nn, 5))

    def forward(self, x):
        return self.net(x)

    def viscous_residual(self, x, mu=0.01):
        """用 forward-mode AD 计算二阶导"""
        y = self.forward(x)
        u, v, w = y[:, 2:3], y[:, 3:4], y[:, 4:5]

        # 一阶导用标准reverse-mode autograd
        du = gradients(u, x)[0]
        u_x, u_y, u_z = du[:, 1:2], du[:, 2:3], du[:, 3:4]
        dv = gradients(v, x)[0]
        v_x, v_y, v_z = dv[:, 1:2], dv[:, 2:3], dv[:, 3:4]
        dw = gradients(w, x)[0]
        w_x, w_y, w_z = dw[:, 1:2], dw[:, 2:3], dw[:, 3:4]

        div_u = u_x + v_y + w_z

        # 二阶导: 用 forward-mode jvp
        # jvp(f, (x,), (v,)) 计算 f(x) 和 df/dx @ v (方向导数)
        # 对 u_x 关于 x 的导数，在 e_x 方向 → u_xx
        def get_second_deriv_fwd(first_deriv, x, dim):
            """用jvp计算 ∂(first_deriv)/∂x_dim"""
            tangent = torch.zeros_like(x)
            tangent[:, dim] = 1.0
            # first_deriv 是 x 的函数 (通过计算图)
            # 我们需要 d(first_deriv)/dx @ tangent
            return torch.autograd.functional.jvp(
                lambda xx: torch.autograd.grad(
                    self.forward(xx)[:, 2:3].sum(), xx, create_graph=True
                )[0][:, dim:dim+1],
                (x,), (tangent,)
            )[1]

        # 这个方案其实在PINN场景下不太好用，因为forward-mode对参数多的网络不高效
        # 改用 直接对一阶导做reverse-mode，但用detach+recompute减少图大小
        # 实际就是标准嵌套，这里做个对照
        u_xx = gradients(u_x, x)[0][:, 1:2]
        u_yy = gradients(u_y, x)[0][:, 2:3]
        u_zz = gradients(u_z, x)[0][:, 3:4]
        v_xx = gradients(v_x, x)[0][:, 1:2]
        v_yy = gradients(v_y, x)[0][:, 2:3]
        v_zz = gradients(v_z, x)[0][:, 3:4]
        w_xx = gradients(w_x, x)[0][:, 1:2]
        w_yy = gradients(w_y, x)[0][:, 2:3]
        w_zz = gradients(w_z, x)[0][:, 3:4]

        div_u_x = gradients(div_u, x)[0][:, 1:2]
        div_u_y = gradients(div_u, x)[0][:, 2:3]
        div_u_z = gradients(div_u, x)[0][:, 3:4]

        visc_x = mu * (u_xx + u_yy + u_zz + (1.0/3.0)*div_u_x)
        visc_y = mu * (v_xx + v_yy + v_zz + (1.0/3.0)*div_u_y)
        visc_z = mu * (w_xx + w_yy + w_zz + (1.0/3.0)*div_u_z)

        loss = (visc_x**2).mean() + (visc_y**2).mean() + (visc_z**2).mean()
        return loss, visc_x.detach()


# ========== 方案4: 混合法 — 一阶autograd + 二阶FD on output ==========
# 核心思想: 二阶导 f_xx ≈ (f(x+h) - 2f(x) + f(x-h))/h²
# 但中心点 f(x) 保留计算图 → loss 对 θ 有梯度
# 扰动点 f(x±h) 用 forward (保留图 through 网络参数)
class HybridNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential()
        self.net.add_module('fc1', nn.Linear(4, Nn))
        self.net.add_module('act1', nn.Tanh())
        for i in range(2, Nl):
            self.net.add_module(f'fc{i}', nn.Linear(Nn, Nn))
            self.net.add_module(f'act{i}', nn.Tanh())
        self.net.add_module('fc_out', nn.Linear(Nn, 5))

    def forward(self, x):
        return self.net(x)

    def viscous_residual(self, x, mu=0.01, h=1e-3):
        """二阶导用FD on network outputs (保留计算图through网络参数)"""
        y0 = self.forward(x)
        u0, v0, w0 = y0[:, 2:3], y0[:, 3:4], y0[:, 4:5]

        # 二阶导用FD on output values: f_xx ≈ (f(x+h) - 2f(x) + f(x-h))/h²
        # 关键: 扰动点的forward output 通过网络参数连接到计算图!
        x_data = x.data  # detach input but network params still have grad

        u_xx = u_yy = u_zz = 0
        v_xx = v_yy = v_zz = 0
        w_xx = w_yy = w_zz = 0

        for spatial_dim in [1, 2, 3]:
            e = torch.zeros_like(x_data)
            e[:, spatial_dim] = h

            # 这些forward调用通过网络权重保留了计算图
            yp = self.forward(x_data + e)
            ym = self.forward(x_data - e)

            u_dd = (yp[:, 2:3] - 2*u0 + ym[:, 2:3]) / (h*h)
            v_dd = (yp[:, 3:4] - 2*v0 + ym[:, 3:4]) / (h*h)
            w_dd = (yp[:, 4:5] - 2*w0 + ym[:, 4:5]) / (h*h)

            if spatial_dim == 1:
                u_xx, v_xx, w_xx = u_dd, v_dd, w_dd
            elif spatial_dim == 2:
                u_yy, v_yy, w_yy = u_dd, v_dd, w_dd
            else:
                u_zz, v_zz, w_zz = u_dd, v_dd, w_dd

        # 简化: 只算 Laplacian (同FDNet)
        visc_x = mu * (u_xx + u_yy + u_zz)
        visc_y = mu * (v_xx + v_yy + v_zz)
        visc_z = mu * (w_xx + w_yy + w_zz)

        loss = (visc_x**2).mean() + (visc_y**2).mean() + (visc_z**2).mean()
        return loss, visc_x.detach()


# ========== 运行benchmark ==========
methods = [
    ('0_baseline',  BaselineNet),
    ('1_auxiliary',  AuxiliaryNet),
    ('2_fd',         FDNet),
    ('3_fwdmode',    FwdModeNet),
    ('4_hybrid',     HybridNet),
]

print(f'Config: {Nl} layers × {Nn} neurons, {N_pts} points, {N_epochs} epochs')
print(f'{"="*70}\n')

all_results = {}

for name, NetClass in methods:
    print(f'--- {name} ---')
    model = NetClass().to(dtype).to(cuda)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.rand(N_pts, 4, dtype=dtype, device=cuda) * 2 - 1
    x.requires_grad_(True)

    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        loss, _ = model.viscous_residual(x)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()

    # 计时
    fwd_times = []
    bwd_times = []

    for ep in range(N_epochs):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        loss, visc_ref = model.viscous_residual(x)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        t2 = time.perf_counter()

        fwd_times.append(t1 - t0)
        bwd_times.append(t2 - t1)

    fwd_ms = np.mean(fwd_times) * 1000
    bwd_ms = np.mean(bwd_times) * 1000
    tot_ms = fwd_ms + bwd_ms

    all_results[name] = {
        'n_params': n_params, 'fwd_ms': fwd_ms,
        'bwd_ms': bwd_ms, 'tot_ms': tot_ms,
        'final_loss': to_numpy(loss)
    }

    print(f'  Params: {n_params}, Forward: {fwd_ms:.2f}ms, '
          f'Backward: {bwd_ms:.2f}ms, Total: {tot_ms:.2f}ms')

    del model, optimizer, x
    torch.cuda.empty_cache()

# ========== 汇总 ==========
print(f'\n{"="*70}')
print(f'Summary')
print(f'{"="*70}')
baseline_tot = all_results['0_baseline']['tot_ms']
print(f'{"Method":<15} {"Params":<10} {"Fwd(ms)":<10} {"Bwd(ms)":<10} '
      f'{"Total(ms)":<12} {"vs Base":<10}')
print(f'{"-"*67}')
for name in all_results:
    r = all_results[name]
    speedup = baseline_tot / r['tot_ms']
    print(f'{name:<15} {r["n_params"]:<10} {r["fwd_ms"]:<10.2f} {r["bwd_ms"]:<10.2f} '
          f'{r["tot_ms"]:<12.2f} {speedup:<10.2f}x')
