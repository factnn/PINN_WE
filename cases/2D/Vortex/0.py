# ========== 路径设置（必须放在最前面）==========
import sys
import os

# 自动检测并添加PINNsrc路径
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# 从 cases/2D/Vortex 向上三级到 PINN_WE，然后进入 PINNsrc
pinn_src_path = os.path.join(current_file_dir, '..', '..', '..', 'PINNsrc')
pinn_src_path = os.path.abspath(pinn_src_path)
if pinn_src_path not in sys.path:
    sys.path.insert(0, pinn_src_path)

# ========== 导入模块 ==========
from PINNs import *
from utility import select_gpu, create_output_dir, save_results
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from smt.sampling_methods import LHS

dtype = torch.float64
setup_seed(2)

# ========== 文献参数设置 ==========
# 等熵涡传播问题（文献参数）
# 平均流：(ρ, U, V, p) = (1, 1, 1, 1)
# 涡强度：ε = 5
# 计算域：[-5, 5] × [-5, 5]
# 周期边界条件
Ts = 0
Te = 10.0  # 终止时间
Xs, Xe = -5.0, 5.0  # x方向：[-5, 5]
Ys, Ye = -5.0, 5.0  # y方向：[-5, 5]

setup_seed(7)

# ========== GPU模式选择 ==========
gpu_id = select_gpu()
if gpu_id is not None:
    torch.cuda.set_device(gpu_id)
    cuda = torch.device('cuda')
    print(f'使用 GPU {gpu_id}')
else:
    cuda = torch.device('cpu')
    print('未检测到可用 GPU，使用 CPU')

use_multi_gpu = os.environ.get('USE_MULTI_GPU', '0') == '1'
num_gpus = torch.cuda.device_count()

# ========== 初始条件：等熵涡（文献参数）==========
def IC_Vortex(x):
    """
    等熵涡初始条件（文献参数）
    - 平均流：(ρ, U, V, p) = (1, 1, 1, 1)
    - 涡强度：ε = 5
    - 涡中心：(0, 0)
    - 扰动：(δU, δV) = (ε/(2π)) * e^((1-r²)/2) * (-y, x)
    - δT = -(γ-1)ε²/(8γπ²) * e^(1-r²)
    """
    N = x.shape[0]
    rho_init = np.zeros(N)
    u_init = np.zeros(N)
    v_init = np.zeros(N)
    p_init = np.zeros(N)

    epsilon = 5.0  # 涡强度
    gamma = 1.4
    u_inf = 1.0
    v_inf = 1.0
    pi = math.pi

    for i in range(N):
        rx = x[i, 1]  # x坐标
        ry = x[i, 2]  # y坐标
        rsq = rx * rx + ry * ry

        # 速度扰动
        du = -(epsilon / (2.0 * pi)) * math.exp(0.5 * (1.0 - rsq)) * ry
        dv = (epsilon / (2.0 * pi)) * math.exp(0.5 * (1.0 - rsq)) * rx

        # 温度扰动
        dT = -((gamma - 1.0) * epsilon * epsilon) / (8.0 * gamma * pi * pi) * math.exp(1.0 - rsq)

        # 从温度和无熵扰动推导密度和压力
        T = 1.0 + dT
        rho_init[i] = math.pow(T, 1.0 / (gamma - 1.0))
        p_init[i] = rho_init[i] * T

        u_init[i] = u_inf + du
        v_init[i] = v_inf + dv

    return rho_init, u_init, v_init, p_init

# ========== 训练函数 ==========
def train(epoch, stage='Adam', total_epochs=None):
    """
    标准PINN训练函数（用于对比测试）
    - 损失函数：loss_pde + loss_ic + loss_bc（周期边界条件）
    - 权重函数k=0（标准PINN，无权重函数）
    """
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    # 标准PINN配置：k=0
    k = 0

    def closure():
        optimizer.zero_grad()
        loss_pde = actual_model.loss_pde(x_int, k=k)  # k=0，无权重函数
        loss_ic = actual_model.loss_ic(x_ic, rho_ic, u_ic, v_ic, p_ic)

        # 周期边界条件
        y_left = actual_model.net(x_bc_left)
        y_right = actual_model.net(x_bc_right)
        y_bottom = actual_model.net(x_bc_bottom)
        y_top = actual_model.net(x_bc_top)
        loss_bc = torch.mean((y_left - y_right)**2) + torch.mean((y_bottom - y_top)**2)

        # 标准PINN损失
        loss = loss_pde + 10*loss_ic + 10*loss_bc

        print(f'epoch {epoch} loss:{loss:.8f}, loss_pde:{loss_pde:.8f}, loss_ic:{loss_ic:.8f}, loss_bc:{loss_bc:.8f}')
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    return loss

# ========== 采样点设置 ==========
# 初始条件采样（t=0）
xlimits_ic = np.array([[0., 0.], [Xs, Xe], [Ys, Ye]])
sampling_ic = LHS(xlimits=xlimits_ic)
x_ic = sampling_ic(10000)

# 使用文献参数的初始条件
rho_ic, u_ic, v_ic, p_ic = IC_Vortex(x_ic)

# 内部PDE残差点采样
xlimits = np.array([[0., Te], [Xs, Xe], [Ys, Ye]])
sampling = LHS(xlimits=xlimits)
x_int = sampling(50000)

# 周期边界条件采样
Nbc = 5000
# 左边界 (x=-5) 和右边界 (x=5)
t_bc = np.random.uniform(0, Te, Nbc)
y_bc = np.random.uniform(Ys, Ye, Nbc)
x_bc_left = np.stack([t_bc, np.full(Nbc, Xs), y_bc], axis=1)
x_bc_right = np.stack([t_bc, np.full(Nbc, Xe), y_bc], axis=1)

# 下边界 (y=-5) 和上边界 (y=5)
t_bc2 = np.random.uniform(0, Te, Nbc)
x_bc2 = np.random.uniform(Xs, Xe, Nbc)
x_bc_bottom = np.stack([t_bc2, x_bc2, np.full(Nbc, Ys)], axis=1)
x_bc_top = np.stack([t_bc2, x_bc2, np.full(Nbc, Ye)], axis=1)

print(f'采样完成：初始条件{len(x_ic)}，内部点{len(x_int)}，边界条件{Nbc*4}')

# 保存numpy版本用于可视化
x_ic_np = x_ic.copy()
rho_ic_np = rho_ic.copy()
u_ic_np = u_ic.copy()
v_ic_np = v_ic.copy()
p_ic_np = p_ic.copy()

# 转换为Tensor
x_ic = torch.tensor(x_ic, requires_grad=True, dtype=dtype).to(cuda)
x_int = torch.tensor(x_int, requires_grad=True, dtype=dtype).to(cuda)
x_bc_left = torch.tensor(x_bc_left, requires_grad=True, dtype=dtype).to(cuda)
x_bc_right = torch.tensor(x_bc_right, requires_grad=True, dtype=dtype).to(cuda)
x_bc_bottom = torch.tensor(x_bc_bottom, requires_grad=True, dtype=dtype).to(cuda)
x_bc_top = torch.tensor(x_bc_top, requires_grad=True, dtype=dtype).to(cuda)

rho_ic = torch.tensor(rho_ic, dtype=dtype).to(cuda)
u_ic = torch.tensor(u_ic, dtype=dtype).to(cuda)
v_ic = torch.tensor(v_ic, dtype=dtype).to(cuda)
p_ic = torch.tensor(p_ic, dtype=dtype).to(cuda)

# ========== 网络结构 ==========
model = PINNs_WE_Euler_2D(Nl=6, Nn=90).to(cuda).double()

if use_multi_gpu and num_gpus > 1:
    print(f'检测到 {num_gpus} 个GPU，启用多GPU并行训练')
    model = torch.nn.DataParallel(model)
else:
    print(f'检测到 {num_gpus} 个GPU，使用单GPU模式（默认）')
    if num_gpus > 0:
        print(f'提示：如需使用多GPU，请设置环境变量 USE_MULTI_GPU=1')
        print(f'使用单GPU: {torch.cuda.get_device_name(gpu_id or 0)}')

print('Start training...')
epoch = 0
epochi = epoch
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 100000  # Adam阶段
loss_history = []
tic_total = time.time()

# ========== 创建输出目录 ==========
base_output_dir = create_output_dir('Vortex2D', gpu_count=num_gpus, tag='baseline_epsilon5')
print(f'\n[输出目录] {base_output_dir}')

# ========== Adam阶段训练 ==========
print('========== Adam阶段：标准PINN训练（k=0，无权重函数） ==========')
tic = time.time()
for epoch in range(1+epochi, epochs+epochi):
    loss = train(epoch, stage='Adam', total_epochs=epochs)
    loss_history.append(to_numpy(loss))

    # 早停条件
    if loss < 0.01:
        print(f'Loss < 0.01，提前停止训练')
        break

    # 每10000个epoch打印一次
    if epoch % 10000 == 0:
        print(f'Epoch {epoch}/{epochs}, Loss: {loss:.8f}')

toc = time.time()
print(f'Adam阶段训练完成，用时: {toc - tic:.2f}秒')
epochi = epoch
stage_boundary = len(loss_history)

# ========== LBFGS阶段训练 ==========
print('========== LBFGS阶段：精细化训练 ==========')
optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=30)
epochs_lbfgs = 1500
tic = time.time()
for epoch in range(1+epochi, epochs_lbfgs+epochi):
    loss = train(epoch, stage='LBFGS', total_epochs=epochs_lbfgs)
    loss_history.append(to_numpy(loss))

    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{epochs_lbfgs+epochi}, Loss: {loss:.8f}')

toc = time.time()
print(f'LBFGS阶段训练完成，用时: {toc - tic:.2f}秒')
training_time = time.time() - tic_total
print(f'总训练时间: {training_time:.2f}秒')

# ========== 预测 ==========
print('开始预测...')
actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model

# 生成评估网格（100x100）
nx = ny = 100
x_lin = np.linspace(Xs, Xe, nx)
y_lin = np.linspace(Ys, Ye, ny)
Xg, Yg = np.meshgrid(x_lin, y_lin, indexing='ij')
t_eval = np.full_like(Xg, Te)
x_test_np = np.stack([t_eval, Xg, Yg], axis=-1).reshape(-1, 3)
x_test = torch.tensor(x_test_np, dtype=dtype).to(cuda)

actual_model.eval()
with torch.no_grad():
    pred = actual_model(x_test)
rho_pred = to_numpy(pred[:, 0]).reshape(nx, ny)
p_pred = to_numpy(pred[:, 1]).reshape(nx, ny)
u_pred = to_numpy(pred[:, 2]).reshape(nx, ny)
v_pred = to_numpy(pred[:, 3]).reshape(nx, ny)

print('预测完成')

# ========== 计算精确解 ==========
print('计算精确解...')
x_exact_input = np.zeros((nx * ny, 3))
domain_width_x = Xe - Xs
domain_width_y = Ye - Ys

for i in range(nx):
    for j in range(ny):
        # 涡以速度(1,1)传播，考虑周期边界条件
        x_shifted = Xg[i, j] - Te
        y_shifted = Yg[i, j] - Te
        # 周期边界
        x_shifted = np.mod(x_shifted - Xs, domain_width_x) + Xs
        y_shifted = np.mod(y_shifted - Ys, domain_width_y) + Ys
        x_exact_input[i * ny + j, :] = [0.0, x_shifted, y_shifted]

rho_exact, u_exact, v_exact, p_exact = IC_Vortex(x_exact_input)
rho_exact = rho_exact.reshape(nx, ny)
u_exact = u_exact.reshape(nx, ny)
v_exact = v_exact.reshape(nx, ny)
p_exact = p_exact.reshape(nx, ny)

print('精确解计算完成')

# ========== 计算误差 ==========
print('计算误差...')
rho_err = rho_pred - rho_exact
p_err = p_pred - p_exact
u_err = u_pred - u_exact
v_err = v_pred - v_exact

rho_l2_error = np.sqrt(np.mean(rho_err**2))
p_l2_error = np.sqrt(np.mean(p_err**2))
u_l2_error = np.sqrt(np.mean(u_err**2))
v_l2_error = np.sqrt(np.mean(v_err**2))

rho_max_error = np.max(np.abs(rho_err))
p_max_error = np.max(np.abs(p_err))
u_max_error = np.max(np.abs(u_err))
v_max_error = np.max(np.abs(v_err))

# 相对误差
rho_rel_l2 = rho_l2_error / (np.sqrt(np.mean(rho_exact**2)) + 1e-10)
p_rel_l2 = p_l2_error / (np.sqrt(np.mean(p_exact**2)) + 1e-10)
u_rel_l2 = u_l2_error / (np.sqrt(np.mean(u_exact**2)) + 1e-10)
v_rel_l2 = v_l2_error / (np.sqrt(np.mean(v_exact**2)) + 1e-10)

print(f'\n========== 误差统计 ==========')
print(f'密度 - L2: {rho_l2_error:.6e}, 相对L2: {rho_rel_l2*100:.4f}%, Max: {rho_max_error:.6e}')
print(f'压力 - L2: {p_l2_error:.6e}, 相对L2: {p_rel_l2*100:.4f}%, Max: {p_max_error:.6e}')
print(f'速度u - L2: {u_l2_error:.6e}, 相对L2: {u_rel_l2*100:.4f}%, Max: {u_max_error:.6e}')
print(f'速度v - L2: {v_l2_error:.6e}, 相对L2: {v_rel_l2*100:.4f}%, Max: {v_max_error:.6e}')

# ========== 准备结果字典（2D问题，不使用evaluate_predictions）==========
pred_dict = {
    'x': Xg.flatten(),
    'y': Yg.flatten(),
    'rho': rho_pred.flatten(),
    'p': p_pred.flatten(),
    'u': u_pred.flatten(),
    'v': v_pred.flatten(),
    't': Te
}

# ========== 保存结果 ==========
print('\n保存结果...')
output_dir = create_output_dir('Vortex2D', gpu_count=num_gpus, tag='baseline_epsilon5')

# 保存模型
import torch
model_path = os.path.join(output_dir, 'model.pth')
if isinstance(model, torch.nn.DataParallel):
    torch.save(model.module.state_dict(), model_path)
else:
    torch.save(model.state_dict(), model_path)
print(f'✓ 模型已保存: {model_path}')

# 保存损失历史
if loss_history:
    loss_path = os.path.join(output_dir, 'loss_history.npy')
    np.save(loss_path, np.array(loss_history))
    print(f'✓ 损失历史已保存: {loss_path}')

# 保存预测数据（2D格式）
data_path = os.path.join(output_dir, 'predictions_2d.npz')
np.savez(data_path,
         x=Xg, y=Yg, t=Te,
         rho_pred=rho_pred, p_pred=p_pred, u_pred=u_pred, v_pred=v_pred,
         rho_exact=rho_exact, p_exact=p_exact, u_exact=u_exact, v_exact=v_exact,
         rho_error=rho_err, p_error=p_err, u_error=u_err, v_error=v_err)
print(f'✓ 预测数据已保存: {data_path}')

# 保存误差统计
metrics = {
    'rho_l2': float(rho_l2_error),
    'p_l2': float(p_l2_error),
    'u_l2': float(u_l2_error),
    'v_l2': float(v_l2_error),
    'rho_rel_l2': float(rho_rel_l2),
    'p_rel_l2': float(p_rel_l2),
    'u_rel_l2': float(u_rel_l2),
    'v_rel_l2': float(v_rel_l2),
    'rho_max': float(rho_max_error),
    'p_max': float(p_max_error),
    'u_max': float(u_max_error),
    'v_max': float(v_max_error),
    'training_time': float(training_time)
}
metrics_path = os.path.join(output_dir, 'metrics.json')
import json
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f'✓ 误差统计已保存: {metrics_path}')

print(f'\n结果已保存到: {output_dir}')

# ========== 2D可视化 ==========
print('\n生成2D可视化图像...')
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# 第一行：密度
im0 = axes[0, 0].imshow(rho_pred, origin='lower', extent=[Ys, Ye, Xs, Xe],
                        cmap='viridis', aspect='equal')
axes[0, 0].set_title('PINN ρ (t=%.1f)' % Te)
axes[0, 0].set_xlabel('y')
axes[0, 0].set_ylabel('x')
fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

im1 = axes[0, 1].imshow(rho_exact, origin='lower', extent=[Ys, Ye, Xs, Xe],
                        cmap='viridis', aspect='equal')
axes[0, 1].set_title('Exact ρ')
axes[0, 1].set_xlabel('y')
axes[0, 1].set_ylabel('x')
fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

im2 = axes[0, 2].imshow(rho_err, origin='lower', extent=[Ys, Ye, Xs, Xe],
                        cmap='RdBu_r', aspect='equal')
axes[0, 2].set_title('ρ Error (PINN - Exact)')
axes[0, 2].set_xlabel('y')
axes[0, 2].set_ylabel('x')
fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

# 密度误差统计
axes[0, 3].text(0.1, 0.7, f'ρ L2: {rho_l2_error:.6e}', fontsize=11)
axes[0, 3].text(0.1, 0.5, f'ρ Rel: {rho_rel_l2*100:.4f}%', fontsize=11)
axes[0, 3].text(0.1, 0.3, f'ρ Max: {rho_max_error:.6e}', fontsize=11)
axes[0, 3].axis('off')
axes[0, 3].set_title('Error Statistics')

# 第二行：压力
im3 = axes[1, 0].imshow(p_pred, origin='lower', extent=[Ys, Ye, Xs, Xe],
                        cmap='viridis', aspect='equal')
axes[1, 0].set_title('PINN p (t=%.1f)' % Te)
axes[1, 0].set_xlabel('y')
axes[1, 0].set_ylabel('x')
fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

im4 = axes[1, 1].imshow(p_exact, origin='lower', extent=[Ys, Ye, Xs, Xe],
                        cmap='viridis', aspect='equal')
axes[1, 1].set_title('Exact p')
axes[1, 1].set_xlabel('y')
axes[1, 1].set_ylabel('x')
fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

im5 = axes[1, 2].imshow(p_err, origin='lower', extent=[Ys, Ye, Xs, Xe],
                        cmap='RdBu_r', aspect='equal')
axes[1, 2].set_title('p Error (PINN - Exact)')
axes[1, 2].set_xlabel('y')
axes[1, 2].set_ylabel('x')
fig.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

# 压力和速度误差统计
axes[1, 3].text(0.1, 0.8, f'p L2: {p_l2_error:.6e}', fontsize=11)
axes[1, 3].text(0.1, 0.65, f'p Rel: {p_rel_l2*100:.4f}%', fontsize=11)
axes[1, 3].text(0.1, 0.5, f'u L2: {u_l2_error:.6e}', fontsize=11)
axes[1, 3].text(0.1, 0.35, f'u Rel: {u_rel_l2*100:.4f}%', fontsize=11)
axes[1, 3].text(0.1, 0.2, f'v L2: {v_l2_error:.6e}', fontsize=11)
axes[1, 3].text(0.1, 0.05, f'v Rel: {v_rel_l2*100:.4f}%', fontsize=11)
axes[1, 3].axis('off')
axes[1, 3].set_title('Error Statistics')

plt.tight_layout()
plot_path = os.path.join(output_dir, 'results_vortex_2d.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'✓ 2D结果图已保存: {plot_path}')

print(f"\n========== 训练完成 ==========")
print(f"输出目录: {output_dir}")
print(f"训练时间: {training_time:.2f}秒")
print(f"\n误差统计:")
print(f"  密度 - L2: {rho_l2_error:.6e}, 相对: {rho_rel_l2*100:.4f}%")
print(f"  压力 - L2: {p_l2_error:.6e}, 相对: {p_rel_l2*100:.4f}%")
print(f"  速度u - L2: {u_l2_error:.6e}, 相对: {u_rel_l2*100:.4f}%")
print(f"  速度v - L2: {v_l2_error:.6e}, 相对: {v_rel_l2*100:.4f}%")


