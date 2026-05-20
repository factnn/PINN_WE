# ========== 路径设置（必须放在最前面）==========
import sys
import os

# 自动检测并添加PINNsrc路径
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# 从 cases/1D/Blast 向上三级到 PINN_WE，然后进入 PINNsrc
pinn_src_path = os.path.join(current_file_dir, '..', '..', '..', 'PINNsrc')
pinn_src_path = os.path.abspath(pinn_src_path)
if pinn_src_path not in sys.path:
    sys.path.insert(0, pinn_src_path)

# ========== 导入模块 ==========
from PINNs import *
from utility import select_gpu, visualize_training_progress, create_output_dir, visualize_initial_condition
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from smt.sampling_methods import LHS
dtype=torch.float64
setup_seed(2)
Ts = 0
Te = 2.0  # Advection问题：终止时间2.0
Xs = 0
Xe = 2.0  # Advection问题：空间域[0, 2]

# ========== Advection问题参数 ==========
Nx = 100
Nt = 100
dt = 0.02
dx = 0.02

# Advection问题没有左右状态，只有初始条件
# ρ(x) = 1 + 0.2*sin(πx), U(x) = 1, p(x) = 1
setup_seed(7)

# ========== GPU模式选择 ==========
# 优先 GPU_ID；默认自动选显存占用最低；AUTO_GPU=0 可关闭；USE_MULTI_GPU=1 启用多卡
gpu_id = select_gpu()
if gpu_id is not None:
    torch.cuda.set_device(gpu_id)
    cuda = torch.device('cuda')
    print(f'使用 GPU {gpu_id}')
else:
    cuda = torch.device('cpu')
    print('未检测到可用 GPU，使用 CPU')

# 通过环境变量 USE_MULTI_GPU=1 来启用多GPU模式，默认单GPU
use_multi_gpu = os.environ.get('USE_MULTI_GPU', '0') == '1'
num_gpus = torch.cuda.device_count()

def train(epoch, stage='Adam', total_epochs=None):
    """
    标准PINN训练函数（用于对比测试）
    - 损失函数包含2项：loss_pde + loss_ic（标准原始PINN配置）
    - 权重：loss_pde权重1，loss_ic权重10
    - 权重函数k=0（相当于无权重函数，标准PINN）
    - 目的：测试原始PINN在激波问题上的效果
    """
    # ========== DataParallel兼容：获取实际模型 ==========
    # 如果使用DataParallel，需要通过 .module 访问实际模型
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    # ========== 标准PINN配置：k=0 ==========
    # k=0时，权重函数退化为d=1，相当于标准PINN（无权重函数）
    k = 0
    
    def closure():
        optimizer.zero_grad()
        loss_pde = actual_model.loss_pde(x_int, k=k)  # k=0，无权重函数
        loss_ic = actual_model.loss_ic(x_ic, rho_ic,u_ic,p_ic)   # 初始条件约束

        # ========== 周期边界条件约束 ==========
        # 在左边界 x=0 和右边界 x=2 处，物理量应该相等
        y_left = actual_model.net(x_bc_left)
        y_right = actual_model.net(x_bc_right)
        loss_bc = torch.mean((y_left - y_right)**2)

        # ========== PINN配置：PDE残差 + 初始条件 + 周期边界条件 ==========
        loss = loss_pde + 10*loss_ic + 10*loss_bc

        print(f'epoch {epoch} loss:{loss:.8f}, loss_pde:{loss_pde:.8f}, loss_ic:{loss_ic:.8f}, loss_bc:{loss_bc:.8f}')
        loss.backward()
        return loss
    loss = optimizer.step(closure)
    return loss

#x_ic,x_bc,x_int =  Mesh_Data(Nx,Nt,Ts,Te,Xs,Xe)

# ========== 采样点设置（原始仓库参数） ==========
# 论文参数（已注释）:
# x_int= sampling(5000)  # 论文: 5000个PDE残差点
xlimits = np.array([[0.,0],[0, Xe]])  #interal
sampling = LHS(xlimits=xlimits)
x_ic= sampling(100)

# ========== 初值设置：Advection问题的光滑初值 ==========
# ρ(x) = 1 + 0.2*sin(πx), U(x) = 1, p(x) = 1
# 提取空间坐标（x_ic是2D数组 [t, x]）
x_space = x_ic[:, 1]
rho_ic = 1.0 + 0.2 * np.sin(np.pi * x_space)
u_ic = np.ones_like(x_space)
p_ic = np.ones_like(x_space)

xlimits = np.array([[0.,Te],[0, Xe]])  #interal
sampling = LHS(xlimits=xlimits)
x_int= sampling(10000)  # 原始仓库: 10000个PDE残差点

# ========== 周期边界条件采样点 ==========
# 在时间域内均匀采样，左边界 x=0，右边界 x=2
Nbc = 100  # 边界条件采样点数
t_bc = np.linspace(Ts, Te, Nbc)
x_bc_left = np.hstack((t_bc[:, None], np.zeros((Nbc, 1))))  # [t, x=0]
x_bc_right = np.hstack((t_bc[:, None], np.full((Nbc, 1), Xe)))  # [t, x=2]

# 保存numpy版本的初始条件用于可视化（在转换为tensor之前）
x_ic_np = x_ic.copy()  # 保存numpy版本
rho_ic_np = rho_ic.copy()
u_ic_np = u_ic.copy()
p_ic_np = p_ic.copy()

x_ic = torch.tensor(x_ic,requires_grad=True, dtype=dtype).to(cuda)
x_int = torch.tensor(x_int,requires_grad=True, dtype=dtype).to(cuda)
x_bc_left = torch.tensor(x_bc_left, requires_grad=True, dtype=dtype).to(cuda)
x_bc_right = torch.tensor(x_bc_right, requires_grad=True, dtype=dtype).to(cuda)

rho_ic = torch.tensor(rho_ic, dtype=dtype).to(cuda)
u_ic = torch.tensor(u_ic, dtype=dtype).to(cuda)
p_ic = torch.tensor(p_ic, dtype=dtype).to(cuda)


# ========== 网络结构（原始仓库参数） ==========
# 论文参数（已注释）:
# model = PINNs_WE_Euler_1D(Nl=8, Nn=50).to(cuda).double()  # 论文: 7层隐藏层，每层50个神经元
model = PINNs_WE_Euler_1D(Nl=6,Nn=60).to(cuda).double()  # 原始仓库: 6层隐藏层，每层60个神经元

if use_multi_gpu and num_gpus > 1:
    print(f'检测到 {num_gpus} 个GPU，启用多GPU并行训练')
    model = torch.nn.DataParallel(model)
    # 注意：使用DataParallel后，访问模型属性需要用 model.module
else:
    print(f'检测到 {num_gpus} 个GPU，使用单GPU模式（默认）')
    if num_gpus > 0:
        print(f'提示：如需使用多GPU，请设置环境变量 USE_MULTI_GPU=1')
        print(f'使用单GPU: {torch.cuda.get_device_name(gpu_id or 0)}')

print('Start training...')
epoch = 0
epochi = epoch
lr = 0.001                                                           # Learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# ========== 训练参数（原始仓库参数） ==========
# 论文参数（已注释）:
# epochs = 100  # 论文: 100 epochs
# epochs = 3000  # 论文: LBFGS 3000 epochs
epochs = 100000  # 原始仓库: 100000 epochs (有早停 loss < 0.05)
loss_history=[]
tic_total = time.time()

# ========== 创建训练过程可视化目录 ==========
# 先创建主输出目录，然后在其中创建训练可视化子目录
base_output_dir = create_output_dir('Advection', gpu_count=num_gpus, tag='baseline')
training_vis_dir = os.path.join(base_output_dir, 'training_progress')
os.makedirs(training_vis_dir, exist_ok=True)
print(f'\n[训练过程可视化] 输出目录: {base_output_dir}')
print(f'[训练过程可视化] 检查点保存目录: {training_vis_dir}')

# ========== 可视化初始状态（第0个epoch） ==========
print('\n========== 可视化初始状态（第0个epoch） ==========')
# 提取空间坐标（x_ic_np是2D数组 [t, x]，需要提取x坐标）
if x_ic_np.ndim == 2 and x_ic_np.shape[1] == 2:
    x_ic_space = x_ic_np[:, 1]  # 提取空间坐标
else:
    x_ic_space = x_ic_np

# 生成更密集的网格来绘制真解（用于更平滑的曲线）
x_exact_dense = np.linspace(Xs, Xe, 200)  # 200个点用于绘制真解
# Advection问题的初值：ρ(x) = 1 + 0.2*sin(πx), U(x) = 1, p(x) = 1
rho_exact_dense = 1.0 + 0.2 * np.sin(np.pi * x_exact_dense)
u_exact_dense = np.ones_like(x_exact_dense)
p_exact_dense = np.ones_like(x_exact_dense)

exact_dict_ic = {
    'x': x_exact_dense,  # 使用密集网格的空间坐标
    'rho': rho_exact_dense,
    'p': p_exact_dense,
    'u': u_exact_dense
}
visualize_initial_condition(
    x_ic=x_ic_np, rho_ic=rho_ic_np, u_ic=u_ic_np, p_ic=p_ic_np,
    save_dir=training_vis_dir, Xs=Xs, Xe=Xe,
    exact_dict=exact_dict_ic,
    title_prefix="Initial Condition (Advection Problem)"
)

# ========== Adam阶段：标准PINN训练（k=0） ==========
print('========== Adam阶段：标准PINN训练（k=0，无权重函数） ==========')
checkpoint_interval_stage1 = 1000  # Adam阶段每1000个epoch保存一次
tic = time.time()
for epoch in range(1+epochi, epochs+epochi):
    loss = train(epoch, stage='Adam', total_epochs=epochs)
    print(f'loss_tot:{loss:.8f}')
    loss_history.append(to_numpy(loss))
    
    # ========== 训练过程可视化检查点（每1000个epoch） ==========
    if epoch % checkpoint_interval_stage1 == 0:
        visualize_training_progress(
            model=model, epoch=epoch, stage='Adam', save_dir=training_vis_dir,
            Xs=Xs, Xe=Xe, Te=Te, exact_file=None,  # Advection问题暂无精确解文件
            rhoref=1.0, uref=1.0, pref=1.0,
            device=cuda, dtype=dtype
        )
    
    if loss < 0.05:  # 原始仓库: 早停条件（与1.py对齐）
        print(f'达到早停条件 (loss < 0.05)，第一阶段训练结束')
        # 保存最后一次检查点
        if training_vis_dir is not None:
            visualize_training_progress(
                model=model, epoch=epoch, stage='Adam', save_dir=training_vis_dir,
                Xs=Xs, Xe=Xe, Te=Te, exact_file=None,  # Advection问题暂无精确解文件
                rhoref=1.0, uref=1.0, pref=1.0,
                device=cuda, dtype=dtype
            )
        break
toc = time.time()
adam_time = toc - tic
print(f'Adam training time: {adam_time}')

optimizer = torch.optim.LBFGS(model.parameters(),lr=0.1,max_iter=20)

epochi = 0

# ========== LBFGS阶段：标准PINN训练（k=0） ==========
print('========== LBFGS阶段：标准PINN训练（k=0，无权重函数） ==========')
epochs = 5000  # 原始仓库: 5000 epochs
checkpoint_interval_stage2 = 100  # LBFGS阶段每100个epoch保存一次
tic = time.time()
for epoch in range(epochi, epochs+epochi):
    loss = train(epoch, stage='LBFGS', total_epochs=epochs)
    print(f'loss_tot:{loss:.8f}')
    #loss_history.append(to_numpy(loss))
    
    # ========== 训练过程可视化检查点（每100个epoch） ==========
    if epoch % checkpoint_interval_stage2 == 0:
        visualize_training_progress(
            model=model, epoch=epoch, stage='LBFGS', save_dir=training_vis_dir,
            Xs=Xs, Xe=Xe, Te=Te, exact_file=None,  # Advection问题暂无精确解文件
            rhoref=1.0, uref=1.0, pref=1.0,
            device=cuda, dtype=dtype
        )
    #if loss < 0.01:
    #    break
toc = time.time()
lbfgs_time = toc - tic
print(f'LBFGS training time: {lbfgs_time}')
# 保存最后一次检查点
if training_vis_dir is not None:
    visualize_training_progress(
        model=model, epoch=epoch, stage='LBFGS', save_dir=training_vis_dir,
        Xs=Xs, Xe=Xe, Te=Te, exact_file=None,  # Advection问题暂无精确解文件
        rhoref=1.0, uref=1.0, pref=1.0,
        device=cuda, dtype=dtype
    )
total_training_time = time.time() - tic_total

# ========== 原代码（保留用于回溯） ==========
# torch.save(model.state_dict(), 'model.pth')
# print('模型已保存到: model.pth')

# ========== 使用通用结果保存系统（改进版） ==========
# 注意：所有算例都可以使用这个函数，无需重复修改

# ========== 可视化结果（带精确解对比）==========
print('\n开始生成可视化结果...')

# 创建测试网格（在最终时刻 t=Te）
x = np.linspace(Xs, Xe, 200)
t = np.full_like(x, Te)
x_test = np.hstack((t[:, None], x[:, None]))
x_test_tensor = torch.tensor(x_test, dtype=dtype).to(cuda)

# 预测
model.eval()
with torch.no_grad():
    # DataParallel会自动处理，但为了兼容性，直接调用即可
    u_pred = model(x_test_tensor)
    rho_pred = u_pred[:, 0].cpu().numpy()
    p_pred = u_pred[:, 1].cpu().numpy()
    u_vel_pred = u_pred[:, 2].cpu().numpy()

print("预测完成")

# 计算Advection问题的解析解
# 解析解：ρ(x,t) = 1 + 0.2*sin(π(x-t)), U(x,t) = 1, p(x,t) = 1
# 注意：需要处理周期边界条件
x_exact = x
x_shifted = x - Te  # x - U*t，其中U=1
# 处理周期边界条件：将x_shifted映射到[0, 2]区间
x_shifted = np.mod(x_shifted - Xs, Xe - Xs) + Xs
rho_exact = 1.0 + 0.2 * np.sin(np.pi * x_shifted)
u_exact = np.ones_like(x)
p_exact = np.ones_like(x)

# 用于插值的精确解（与预测网格相同）
rho_exact_interp = rho_exact
p_exact_interp = p_exact
u_exact_interp = u_exact

# 计算误差
l2_rho = np.sqrt(np.mean((rho_pred - rho_exact_interp)**2)) / np.sqrt(np.mean(rho_exact_interp**2))
l2_p = np.sqrt(np.mean((p_pred - p_exact_interp)**2)) / np.sqrt(np.mean(p_exact_interp**2))
l2_u = np.sqrt(np.mean((u_vel_pred - u_exact_interp)**2)) / np.sqrt(np.mean(u_exact_interp**2))

avg_error = (l2_rho + l2_p + l2_u) / 3
print(f"\n相对L2误差: 密度={l2_rho:.2%}, 压力={l2_p:.2%}, 速度={l2_u:.2%}")
print(f"平均误差: {avg_error:.2%}")

if avg_error < 0.05:
    print("✓✓✓ 结果优秀！")
elif avg_error < 0.10:
    print("✓✓ 结果良好")
elif avg_error < 0.20:
    print("✓ 结果一般")
else:
    print("✗ 需要改进")

# 绘图（带精确解对比）
fig = plt.figure(figsize=(18, 6))

# 密度
ax1 = plt.subplot(1, 3, 1)
ax1.plot(x, rho_pred, 'b-', linewidth=2.5, label='PINN Prediction', zorder=3)
if x_exact is not None:
    ax1.plot(x_exact, rho_exact, 'r--', linewidth=2, label='Exact', alpha=0.8, zorder=2)
    ax1.fill_between(x, rho_pred, rho_exact_interp, alpha=0.2, color='gray', label='Error')
    ax1.text(0.02, 0.98, f'Error: {l2_rho:.2%}', transform=ax1.transAxes, 
             fontsize=11, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax1.set_xlabel('Position x', fontsize=13, fontweight='bold')
ax1.set_ylabel('Density ρ', fontsize=13, fontweight='bold')
ax1.set_title(f'Density (t={Te})', fontsize=14, fontweight='bold')
ax1.set_xlim(Xs, Xe)  # 统一x轴范围为[0, 1]
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11, loc='best')

# 压力
ax2 = plt.subplot(1, 3, 2)
ax2.plot(x, p_pred, 'b-', linewidth=2.5, label='PINN Prediction', zorder=3)
if x_exact is not None:
    ax2.plot(x_exact, p_exact, 'r--', linewidth=2, label='Exact', alpha=0.8, zorder=2)
    ax2.fill_between(x, p_pred, p_exact_interp, alpha=0.2, color='gray', label='Error')
    ax2.text(0.02, 0.98, f'Error: {l2_p:.2%}', transform=ax2.transAxes, 
             fontsize=11, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax2.set_xlabel('Position x', fontsize=13, fontweight='bold')
ax2.set_ylabel('Pressure p', fontsize=13, fontweight='bold')
ax2.set_title(f'Pressure (t={Te})', fontsize=14, fontweight='bold')
ax2.set_xlim(Xs, Xe)  # 统一x轴范围为[0, 1]
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=11, loc='best')

# 速度
ax3 = plt.subplot(1, 3, 3)
ax3.plot(x, u_vel_pred, 'b-', linewidth=2.5, label='PINN Prediction', zorder=3)
if x_exact is not None:
    ax3.plot(x_exact, u_exact, 'r--', linewidth=2, label='Exact', alpha=0.8, zorder=2)
    ax3.fill_between(x, u_vel_pred, u_exact_interp, alpha=0.2, color='gray', label='Error')
    ax3.text(0.02, 0.98, f'Error: {l2_u:.2%}', transform=ax3.transAxes, 
             fontsize=11, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax3.set_xlabel('Position x', fontsize=13, fontweight='bold')
ax3.set_ylabel('Velocity u', fontsize=13, fontweight='bold')
ax3.set_title(f'Velocity (t={Te})', fontsize=14, fontweight='bold')
ax3.set_xlim(Xs, Xe)  # 统一x轴范围为[0, 1]
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(fontsize=11, loc='best')

plt.tight_layout()

# ========== 原代码（保留用于回溯） ==========
# plt.savefig('Sod_PINN_results.png', dpi=300, bbox_inches='tight')
# print('\n结果图片已保存到: Sod_PINN_results.png')
# results = np.column_stack([x, rho_pred, p_pred, u_vel_pred])
# np.savetxt('Sod_PINN_results.dat', results, header='x rho p u', fmt='%.6e')
# print('结果数据已保存到: Sod_PINN_results.dat')

# ========== 使用通用结果保存系统（改进版） ==========
# 准备数据
pred_dict = {
    'x': x,
    'rho': rho_pred,
    'p': p_pred,
    'u': u_vel_pred,
    't': Te
}

exact_dict = None
if x_exact is not None:
    exact_dict = {
        'x': x_exact,
        'rho': rho_exact,
        'p': p_exact,
        'u': u_exact
    }

# 配置信息
config = {
    'case': 'Advection_Density_Perturbation',
    'Ts': Ts,
    'Te': Te,
    'Xs': Xs,
    'Xe': Xe,
    'Nx': Nx,
    'Nt': Nt,
    'initial_condition': 'rho(x)=1+0.2*sin(pi*x), U(x)=1, p(x)=1',
    'boundary_condition': 'periodic',
    'model_layers': 6,  # 6层隐藏层
    'model_neurons': 60,  # 每层60个神经元
    'sampling_method': 'LHS',
    'ic_points': 100,  # 100个初始点
    'interior_points': 10000,  # 10000个PDE残差点
    'loss_config': 'PINN_IC',  # 标准PINN：loss_pde + loss_ic
    'k_parameter': 0.0  # 权重函数参数：k=0（标准PINN，无权重函数）
}

# 调用通用保存函数（所有算例都可以用）
from utility import save_results
# 使用之前创建的 base_output_dir，避免重复创建
output_dir, metrics = save_results(
    case_name='Advection',
    model=model,
    pred_dict=pred_dict,
    loss_history=loss_history,
    exact_dict=exact_dict,
    training_time=total_training_time,
    config=config,
    save_model=True,
    save_plot=True,
    gpu_count=num_gpus,
    stage_boundary=len(loss_history),  # 第一阶段（Adam）结束的epoch数
    output_dir=base_output_dir  # 使用之前创建的目录
)

print('\n========== 结果对比说明 ==========')
print('评估标准：')
print('  1. 损失函数 (loss_pde, loss_ic等) - 越小越好')
print('  2. 相对L2误差 - 与精确解对比，越小越好')
print('     - < 5%:  优秀')
print('     - 5-10%: 良好')
print('     - 10-20%: 一般')
print('     - > 20%: 需要改进')
print('  3. 光滑解质量 - 密度扰动传播准确、无数值耗散')
print(f'\n详细结果已保存到: {output_dir}')

# ========== 保存结果统计（与Burgers_Compare格式一致）==========
result_file = 'result_baseline.dat'
with open(result_file, 'w') as f:
    f.write('========== 实验配置 ==========\n')
    f.write(f'问题类型: Advection (密度扰动传播)\n')
    f.write(f'模式: Baseline (k=0)\n')
    f.write(f'权重函数k: 0\n')
    f.write(f'================================\n')
    
    f.write(f'L2_rho: {l2_rho:.6e}\n')
    f.write(f'L2_p: {l2_p:.6e}\n')
    f.write(f'L2_u: {l2_u:.6e}\n')
    f.write(f'L2_avg: {avg_error:.6e}\n')
    
    f.write(f'Adam_time: {adam_time:.6e}s\n')
    f.write(f'LBFGS_time: {lbfgs_time:.6e}s\n')
    f.write(f'Total_time: {total_training_time:.6e}s\n')
    
    f.write('================================\n')
    f.write(f'结果已保存到: {result_file}\n')
    f.write(f'详细结果已保存到: {output_dir}\n')

print(f'\n========== 完成 ==========')
print(f'模式: Baseline')
print(f'结果已保存到: {result_file}')
print(f'详细结果目录: {output_dir}')
print(f'===========================\n')