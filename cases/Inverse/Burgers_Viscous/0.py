# ========== 路径设置（必须放在最前面）==========
import sys
import os

# 自动检测并添加PINNsrc路径
current_file_dir = os.path.dirname(os.path.abspath(__file__))
pinn_src_path = os.path.join(current_file_dir, '..', '..', '..', 'PINNsrc')
pinn_src_path = os.path.abspath(pinn_src_path)
if pinn_src_path not in sys.path:
    sys.path.insert(0, pinn_src_path)

# ========== 导入模块 ==========
from PINNs import *
from utility import select_gpu, create_output_dir
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from smt.sampling_methods import LHS

dtype = torch.float64
setup_seed(2)

# ========== 反问题设置 ==========
# 目标：从测量数据反推粘性系数 nu
# 已知：稀疏的速度测量数据 u(x_i, t_i)
# 未知：粘性系数 nu（真实值 nu=0.01）
Ts = 0
Te = 1.0
Xs = 0
Xe = 2.0

# 初始猜测（故意设置错误的初值）
nu_init = 0.05  # 初始猜测（真实值是 0.01）

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

# ========== 加载测量数据 ==========
print('\n========== 加载测量数据 ==========')
data_file = os.path.join(current_file_dir, 'measurement_data.csv')
data = pd.read_csv(data_file)
x_data = data['x'].values
t_data = data['t'].values
u_data = data['u'].values

print(f'✓ 测量数据已加载: {len(x_data)} 个测量点')
print(f'  x 范围: [{x_data.min():.3f}, {x_data.max():.3f}]')
print(f'  t 范围: [{t_data.min():.3f}, {t_data.max():.3f}]')
print(f'  u 范围: [{u_data.min():.3f}, {u_data.max():.3f}]')

# 转换为 (t, x) 格式
x_data_input = np.stack([t_data, x_data], axis=1)

# ========== 初始条件 ==========
def IC_Burgers(x):
    """
    Burgers方程初始条件
    u(x,0) = -sin(π(x-1))
    """
    N = x.shape[0]
    u_init = np.zeros(N)
    for i in range(N):
        u_init[i] = -np.sin(np.pi * (x[i, 1] - 1))
    return u_init

# ========== 反问题专用网络（nu 可学习）==========
class PINNs_Burgers_Inverse(nn.Module):
    """
    反问题版本的 Burgers PINN
    - 网络预测 u(x,t)
    - nu 作为可学习参数
    """
    def __init__(self, Nl, Nn, nu_init=0.05):
        super(PINNs_Burgers_Inverse, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('Linear_layer_1', nn.Linear(2, Nn))
        self.net.add_module('Tanh_layer_1', nn.Tanh())

        for num in range(2, Nl):
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(Nn, Nn))
            self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())
        self.net.add_module('Linear_layer_final', nn.Linear(Nn, 1))

        # 可学习的粘性系数（关键！）
        # 使用 log 空间保证 nu > 0
        self.log_nu = nn.Parameter(torch.tensor(np.log(nu_init), dtype=torch.float64))

    def forward(self, x):
        return self.net(x)

    @property
    def nu(self):
        """返回正数的 nu（通过 exp 变换）"""
        return torch.exp(self.log_nu)

    def loss_pde_burgers(self, x, k=None):
        """
        Burgers 方程 PDE 损失
        使用可学习的 self.nu
        """
        u = self.net(x)

        # 一阶导数
        du_g = gradients(u, x)[0]
        u_t, u_x = du_g[:, :1], du_g[:, 1:]

        # 二阶导数（粘性项）
        du_xx_g = gradients(u_x, x)[0]
        u_xx = du_xx_g[:, 1:]

        # 权重函数
        if k is None or k == 0:
            lam = 1.0
        else:
            lam = 1.0 / (k * (abs(u_x) - u_x) + 1)

        # Burgers方程残差：u_t + u*u_x - nu*u_xx = 0
        f = (((u_t + u*u_x - self.nu*u_xx)/lam)**2).mean()

        return f

    def loss_ic_burgers(self, x_ic, u_ic):
        """初始条件损失"""
        u_ic_pred = self.net(x_ic)
        u_ic_nn = u_ic_pred[:, 0]
        loss_ic = ((u_ic_nn - u_ic) ** 2).mean()
        return loss_ic

    def loss_data(self, x_data, u_data):
        """数据拟合损失（关键！）"""
        u_pred = self.net(x_data)
        u_pred_flat = u_pred[:, 0]
        loss = ((u_pred_flat - u_data) ** 2).mean()
        return loss


# ========== 训练函数（反问题）==========
def train(epoch, stage='Adam', total_epochs=None):
    """
    反问题训练函数
    - 损失函数：loss_pde + loss_ic + loss_data
    - loss_data 是关键（拟合测量数据）
    """
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    k = 0  # 标准PINN

    def closure():
        optimizer.zero_grad()
        loss_pde = actual_model.loss_pde_burgers(x_int, k=k)
        loss_ic = actual_model.loss_ic_burgers(x_ic, u_ic)
        loss_data = actual_model.loss_data(x_data_tensor, u_data_tensor)

        # 反问题损失：PDE + IC + 数据拟合
        # 大幅增加 PDE 权重，防止网络"作弊"
        loss = 1000*loss_pde + 10*loss_ic + 1*loss_data  # PDE 权重1000，数据权重降到1

        # 获取当前的 nu 值
        nu_current = actual_model.nu.item()

        print(f'epoch {epoch} loss:{loss:.8f}, loss_pde:{loss_pde:.8f}, '
              f'loss_ic:{loss_ic:.8f}, loss_data:{loss_data:.8f}, nu:{nu_current:.6f}')
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    return loss


# ========== 采样点设置 ==========
# 初始条件采样（t=0）
xlimits_ic = np.array([[0., 0.], [Xs, Xe]])
sampling_ic = LHS(xlimits=xlimits_ic)
x_ic = sampling_ic(100)

# 使用Burgers初始条件
u_ic = IC_Burgers(x_ic)

# 内部PDE残差点采样
xlimits = np.array([[0., Te], [Xs, Xe]])
sampling = LHS(xlimits=xlimits)
x_int = sampling(5000)  # 减少内部点，因为有测量数据

print(f'\n========== 采样点统计 ==========')
print(f'初始条件点: {len(x_ic)}')
print(f'内部PDE点: {len(x_int)}')
print(f'测量数据点: {len(x_data)}')

# 保存numpy版本
x_ic_np = x_ic.copy()
u_ic_np = u_ic.copy()

# 转换为Tensor
x_ic = torch.tensor(x_ic, requires_grad=True, dtype=dtype).to(cuda)
x_int = torch.tensor(x_int, requires_grad=True, dtype=dtype).to(cuda)
x_data_tensor = torch.tensor(x_data_input, requires_grad=True, dtype=dtype).to(cuda)

u_ic = torch.tensor(u_ic, dtype=dtype).to(cuda)
u_data_tensor = torch.tensor(u_data, dtype=dtype).to(cuda)


# ========== 网络结构（反问题专用）==========
model = PINNs_Burgers_Inverse(Nl=6, Nn=60, nu_init=nu_init).to(cuda).double()

if use_multi_gpu and num_gpus > 1:
    print(f'检测到 {num_gpus} 个GPU，启用多GPU并行训练')
    model = torch.nn.DataParallel(model)
else:
    print(f'检测到 {num_gpus} 个GPU，使用单GPU模式（默认）')
    if num_gpus > 0:
        print(f'使用单GPU: {torch.cuda.get_device_name(gpu_id or 0)}')

print(f'\n========== 反问题设置 ==========')
print(f'初始猜测: nu = {nu_init}')
print(f'真实值: nu = 0.01 (未知，需要反推)')

print('\nStart training...')
epoch = 0
epochi = epoch
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 50000  # Adam阶段
loss_history = []
nu_history = []  # 记录 nu 的变化
tic_total = time.time()


# ========== 创建输出目录 ==========
base_output_dir = create_output_dir('Inverse_Burgers', gpu_count=num_gpus, tag=f'nu_init{nu_init}')
print(f'\n[输出目录] {base_output_dir}')

# ========== Adam阶段训练 ==========
print(f'\n========== Adam阶段：反问题训练（反推粘性系数 nu） ==========')
tic = time.time()
for epoch in range(1+epochi, epochs+epochi):
    loss = train(epoch, stage='Adam', total_epochs=epochs)
    loss_history.append(to_numpy(loss))
    
    # 记录当前的 nu 值
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    nu_history.append(actual_model.nu.item())

    # 早停条件
    if loss < 0.001:
        print(f'Loss < 0.001，提前停止训练')
        break

    # 每5000个epoch打印一次
    if epoch % 5000 == 0:
        print(f'Epoch {epoch}/{epochs}, Loss: {loss:.8f}, nu: {actual_model.nu.item():.6f}')

toc = time.time()
print(f'Adam阶段训练完成，用时: {toc - tic:.2f}秒')
epochi = epoch
stage_boundary = len(loss_history)


# ========== LBFGS阶段训练 ==========
print('\n========== LBFGS阶段：精细化训练 ==========')
optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=30)
epochs_lbfgs = 1000
tic = time.time()
for epoch in range(1+epochi, epochs_lbfgs+epochi):
    loss = train(epoch, stage='LBFGS', total_epochs=epochs_lbfgs)
    loss_history.append(to_numpy(loss))
    
    # 记录当前的 nu 值
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    nu_history.append(actual_model.nu.item())

    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{epochs_lbfgs+epochi}, Loss: {loss:.8f}, nu: {actual_model.nu.item():.6f}')

toc = time.time()
print(f'LBFGS阶段训练完成，用时: {toc - tic:.2f}秒')
training_time = time.time() - tic_total
print(f'总训练时间: {training_time:.2f}秒')


# ========== 结果分析 ==========
print('\n========== 反问题结果分析 ==========')
actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
nu_predicted = actual_model.nu.item()
nu_true = 0.1  # 真实值（改成0.1测试大粘性情况）

print(f'初始猜测: nu = {nu_init:.6f}')
print(f'反推结果: nu = {nu_predicted:.6f}')
print(f'真实值:   nu = {nu_true:.6f}')
print(f'相对误差: {abs(nu_predicted - nu_true) / nu_true * 100:.2f}%')


# ========== 预测完整流场 ==========
print('\n开始预测完整流场...')
actual_model.eval()

# 生成评估网格
nx = 200
x_test_np = np.linspace(Xs, Xe, nx)
t_test_np = np.full(nx, Te)
x_test_input = np.stack([t_test_np, x_test_np], axis=1)
x_test = torch.tensor(x_test_input, dtype=dtype).to(cuda)

with torch.no_grad():
    u_pred = to_numpy(actual_model(x_test)).flatten()

print('预测完成')


# ========== 可视化结果 ==========
print('\n生成可视化图像...')

# 图1：nu 的收敛历史
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

# 子图1：nu 的变化
ax1 = axes1[0]
epochs_array = np.arange(1, len(nu_history) + 1)
ax1.plot(epochs_array, nu_history, 'b-', linewidth=2, label='Predicted nu')
ax1.axhline(y=nu_true, color='r', linestyle='--', linewidth=2, label=f'True nu = {nu_true}')
ax1.axhline(y=nu_init, color='g', linestyle=':', linewidth=2, label=f'Initial guess = {nu_init}')
ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax1.set_ylabel('Viscosity ν', fontsize=13, fontweight='bold')
ax1.set_title('Convergence of Viscosity Coefficient', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, max(nu_init*1.2, nu_history[0]*1.2)])

# 子图2：损失函数
ax2 = axes1[1]
ax2.semilogy(epochs_array, loss_history, 'b-', linewidth=2)
if stage_boundary > 0 and stage_boundary < len(loss_history):
    ax2.axvline(x=stage_boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Stage Boundary')
ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax2.set_ylabel('Total Loss (log scale)', fontsize=13, fontweight='bold')
ax2.set_title('Training Loss History', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(fontsize=11)

plt.tight_layout()
fig1_path = os.path.join(base_output_dir, 'inverse_convergence.png')
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close(fig1)
print(f'✓ 收敛历史图已保存: {fig1_path}')


# 图2：流场对比（预测 vs 测量数据）
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# 子图1：u(x, t=Te) 预测结果 + 测量点
ax1 = axes2[0]
ax1.plot(x_test_np, u_pred, 'b-', linewidth=2.5, label='PINN Prediction', zorder=2)
# 绘制测量点（只显示 t=Te 附近的点）
mask_te = np.abs(t_data - Te) < 0.1
ax1.scatter(x_data[mask_te], u_data[mask_te], c='r', s=50, marker='o', 
           label='Measurements (t≈1.0)', zorder=3, alpha=0.7)
ax1.set_xlabel('Position x', fontsize=13, fontweight='bold')
ax1.set_ylabel('Velocity u', fontsize=13, fontweight='bold')
ax1.set_title(f'Velocity Field at t={Te}', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# 子图2：所有测量点的拟合情况
ax2 = axes2[1]
x_data_test = torch.tensor(x_data_input, dtype=dtype).to(cuda)
with torch.no_grad():
    u_data_pred = to_numpy(actual_model(x_data_test)).flatten()
ax2.scatter(u_data, u_data_pred, c='b', s=30, alpha=0.6)
ax2.plot([u_data.min(), u_data.max()], [u_data.min(), u_data.max()], 
        'r--', linewidth=2, label='Perfect fit')
ax2.set_xlabel('Measured u', fontsize=13, fontweight='bold')
ax2.set_ylabel('Predicted u', fontsize=13, fontweight='bold')
ax2.set_title('Data Fitting Quality', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
# 计算 R²
r2 = 1 - np.sum((u_data - u_data_pred)**2) / np.sum((u_data - u_data.mean())**2)
ax2.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax2.transAxes, 
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
fig2_path = os.path.join(base_output_dir, 'inverse_field_comparison.png')
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close(fig2)
print(f'✓ 流场对比图已保存: {fig2_path}')


# ========== 保存结果 ==========
print('\n保存结果...')

# 保存模型
model_path = os.path.join(base_output_dir, 'model.pth')
if isinstance(model, torch.nn.DataParallel):
    torch.save(model.module.state_dict(), model_path)
else:
    torch.save(model.state_dict(), model_path)
print(f'✓ 模型已保存: {model_path}')

# 保存 nu 的历史
nu_history_path = os.path.join(base_output_dir, 'nu_history.dat')
np.savetxt(nu_history_path, nu_history, fmt='%.8f', header='nu')
print(f'✓ nu历史已保存: {nu_history_path}')

# 保存损失历史
loss_path = os.path.join(base_output_dir, 'loss_history.dat')
np.savetxt(loss_path, loss_history, fmt='%.6e', header='loss')
print(f'✓ 损失历史已保存: {loss_path}')


# 保存反问题结果摘要
summary = {
    'nu_init': float(nu_init),
    'nu_predicted': float(nu_predicted),
    'nu_true': float(nu_true),
    'relative_error': float(abs(nu_predicted - nu_true) / nu_true * 100),
    'training_time': float(training_time),
    'n_measurements': int(len(x_data)),
    'final_loss': float(loss_history[-1]),
    'r2_score': float(r2)
}

import json
summary_path = os.path.join(base_output_dir, 'inverse_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=4)
print(f'✓ 结果摘要已保存: {summary_path}')


print(f'\n结果已保存到: {base_output_dir}')

# ========== 最终总结 ==========
print(f'\n========== 反问题求解完成 ==========')
print(f'训练时间: {training_time:.2f}秒')
print(f'\n粘性系数反推结果:')
print(f'  初始猜测: nu = {nu_init:.6f}')
print(f'  反推结果: nu = {nu_predicted:.6f}')
print(f'  真实值:   nu = {nu_true:.6f}')
print(f'  相对误差: {abs(nu_predicted - nu_true) / nu_true * 100:.2f}%')
print(f'\n数据拟合质量:')
print(f'  R² score: {r2:.4f}')
print(f'  测量点数: {len(x_data)}')
print(f'\n输出文件:')
print(f'  - inverse_convergence.png (nu收敛历史)')
print(f'  - inverse_field_comparison.png (流场对比)')
print(f'  - inverse_summary.json (结果摘要)')
print(f'  - model.pth (训练好的模型)')

