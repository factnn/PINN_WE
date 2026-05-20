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
Te = 0.2
Xs = 0
Xe = 1
# ========== 参数设置（原始仓库参数） ==========
# 论文参数（已注释）:
# Nt = 200  # 论文: 200
# x_int: 5000个点 (LHS) - 论文: 5000
Nx = 100
Nt = 100  # 原始仓库: 100
dt = 0.002
dx = 0.01

crhoL = 1.0
cuL = 0.0
cpL = 1.0

crhoR = 0.125
cuR = 0
cpR = 0.1
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
    训练函数（傅里叶特征消融实验）
    - 只使用 PDE 损失 + 初始条件损失
    - 不使用 Godunov 损失（纯消融实验，测试傅里叶特征效果）
    """
    # ========== DataParallel兼容：获取实际模型 ==========
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    # ========== 自适应权重：k=0.2 ==========
    k = 0.2

    def closure():
        optimizer.zero_grad()
        loss_pde = actual_model.loss_pde(x_int, k=k)
        loss_ic = actual_model.loss_ic(x_ic, rho_ic, u_ic, p_ic)

        # 总损失（只有 PDE + IC）
        loss = loss_pde + 10*loss_ic

        print(f'epoch {epoch} loss:{loss:.8f}, loss_pde:{loss_pde:.8f}, loss_ic:{loss_ic:.8f}')
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

# ========== 初值设置：使用Riemann问题的阶跃初值 ==========
rho_ic, u_ic, p_ic = IC_Riemann_1D(x_ic, crhoL, cuL, cpL, crhoR, cuR, cpR)

# rho_exact, u_exact, p_exact = IC_Riemann_1D(x_ic, crhoL, cuL, cpL, crhoR, cuR, cpR)

# 光滑扰动函数：使用高斯滤波实现光滑扰动（而不是每个点独立的高斯噪声）
# def smooth_perturbation(x, values, noise_level=0.05, smooth_scale=0.1, seed_offset=0, Xs=0, Xe=1):
#     '''
#     对初值添加光滑扰动
#     x: 空间坐标点（可以是2D数组 [t, x] 或1D数组）
#     values: 原始值（rho, u, 或 p）
#     noise_level: 扰动幅度（相对于值的比例）
#     smooth_scale: 光滑尺度（相对于空间域大小的比例，如0.1表示空间域的10%）
#     seed_offset: 随机种子偏移，用于生成不同的扰动场
#     Xs, Xe: 空间域范围，用于计算光滑尺度
#     '''
#     from scipy.ndimage import gaussian_filter1d
#     # 处理x的格式：如果是2D数组，提取空间坐标
#     if isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] == 2:
#         x_space = x[:, 1]  # 提取空间坐标
#     else:
#         x_space = np.array(x).flatten()
    
#     # 确保x_space是排序的（用于高斯滤波）
#     sort_idx = np.argsort(x_space)
#     x_sorted = x_space[sort_idx]
#     values_sorted = np.array(values)[sort_idx]
    
#     # 生成随机噪声场（使用不同的种子偏移，让每个物理量有不同的扰动）
#     np.random.seed(42 + seed_offset)  # 固定随机种子以便复现，但每个物理量不同
#     noise = np.random.randn(len(x_sorted))
    
#     # 计算高斯滤波的sigma：基于空间域大小，而不是点的数量
#     # smooth_scale是相对于空间域大小的比例（如0.1表示空间域的10%）
#     domain_size = Xe - Xs
#     sigma_pixels = smooth_scale * domain_size * len(x_sorted) / domain_size  # 转换为像素单位
#     # 更合理的计算：sigma应该基于空间距离，而不是点的数量
#     # 假设点均匀分布，每个点之间的平均距离
#     if len(x_sorted) > 1:
#         avg_spacing = domain_size / (len(x_sorted) - 1)
#         sigma_pixels = smooth_scale * domain_size / avg_spacing  # 基于空间尺度的sigma
#     else:
#         sigma_pixels = 1.0
    
#     # 使用高斯滤波使噪声光滑
#     smooth_noise = gaussian_filter1d(noise, sigma=sigma_pixels)
    
#     # 归一化并缩放到指定幅度
#     if np.std(smooth_noise) > 1e-10:
#         smooth_noise = smooth_noise / np.std(smooth_noise) * noise_level
#     else:
#         smooth_noise = noise_level * smooth_noise
    
#     # 添加扰动（相对扰动，保持物理量符号）
#     perturbed_sorted = values_sorted * (1.0 + smooth_noise)
    
#     # 恢复原始顺序
#     perturbed = np.zeros_like(values)
#     perturbed[sort_idx] = perturbed_sorted
    
#     return perturbed

# 对三个物理量分别添加光滑扰动
# noise_level = 0.05  # 5%的扰动幅度
# smooth_scale = 0.15  # 光滑尺度参数（相对于空间域大小的比例，0.15表示空间域的15%，更光滑）
# rho_ic = smooth_perturbation(x_ic, rho_exact, noise_level=noise_level, smooth_scale=smooth_scale, seed_offset=0, Xs=Xs, Xe=Xe)
# u_ic = smooth_perturbation(x_ic, u_exact, noise_level=noise_level, smooth_scale=smooth_scale, seed_offset=1, Xs=Xs, Xe=Xe)
# p_ic = smooth_perturbation(x_ic, p_exact, noise_level=noise_level, smooth_scale=smooth_scale, seed_offset=2, Xs=Xs, Xe=Xe)
# print(f"使用真解+光滑扰动模式: noise_level={noise_level}, smooth_scale={smooth_scale} (空间域的{smooth_scale*100:.0f}%)")



xlimits = np.array([[0.,Te],[0, Xe]])  #interal
sampling = LHS(xlimits=xlimits)
x_int= sampling(10000)  # 原始仓库: 10000个PDE残差点

xrh,xrhL,xrhR,xrhP,xrhPL,xrhPR = Pertur_1D(x_ic,Te,dt,dx)

x_en = Move_Time_1D(x_ic,Te)

# 保存numpy版本的初始条件用于可视化（在转换为tensor之前）
x_ic_np = x_ic.copy()  # 保存numpy版本
rho_ic_np = rho_ic.copy()
u_ic_np = u_ic.copy()
p_ic_np = p_ic.copy()

x_ic = torch.tensor(x_ic,requires_grad=True, dtype=dtype).to(cuda)
x_int = torch.tensor(x_int,requires_grad=True, dtype=dtype).to(cuda)
x_en = torch.tensor(x_en,  dtype=dtype).to(cuda)
xrh = torch.tensor(xrh,  dtype=dtype).to(cuda)
xrhL = torch.tensor(xrhL,  dtype=dtype).to(cuda)

rho_ic = torch.tensor(rho_ic, dtype=dtype).to(cuda)
u_ic = torch.tensor(u_ic, dtype=dtype).to(cuda)
p_ic = torch.tensor(p_ic, dtype=dtype).to(cuda)


# ========== 网络结构（傅里叶特征版本） ==========
# 使用带傅里叶特征的网络，其他参数保持不变
model = PINNs_WE_Euler_1D_Fourier(
    Nl=6,
    Nn=60,
    fourier_mapping_size=128,  # 傅里叶特征维度（输出256维）
    fourier_scale=10.0         # 频率尺度（适合一般激波）
).to(cuda).double()

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
base_output_dir = create_output_dir('Sod', gpu_count=num_gpus, tag='fourier')
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
x_exact_2d = np.hstack((np.zeros((len(x_exact_dense), 1)), x_exact_dense[:, None]))  # [t=0, x]
rho_exact_dense, u_exact_dense, p_exact_dense = IC_Riemann_1D(x_exact_2d, crhoL, cuL, cpL, crhoR, cuR, cpR)

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
    title_prefix="Initial Condition (True Solution + Perturbation)"
)

# ========== Adam阶段：傅里叶特征消融实验 (k=0.2) ==========
print('========== Adam阶段：傅里叶特征消融实验 (k=0.2) ==========')
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
            Xs=Xs, Xe=Xe, Te=Te, exact_file='sode.dat',
            rhoref=1.0, uref=1.0, pref=1.0,
            device=cuda, dtype=dtype
        )
    
    if loss < 0.05:  # 早停条件
        print(f'达到早停条件 (loss < 0.05)，第一阶段训练结束')
        if training_vis_dir is not None:
            visualize_training_progress(
                model=model, epoch=epoch, stage='Adam', save_dir=training_vis_dir,
                Xs=Xs, Xe=Xe, Te=Te, exact_file='sode.dat',
                rhoref=1.0, uref=1.0, pref=1.0,
                device=cuda, dtype=dtype
            )
        break
toc = time.time()
adam_time = toc - tic
print(f'Adam训练时间: {adam_time:.2f}s')

optimizer = torch.optim.LBFGS(model.parameters(),lr=0.1,max_iter=20)

epochi = 0

# ========== LBFGS阶段：傅里叶特征消融实验 (k=0.2) ==========
print('========== LBFGS阶段：傅里叶特征消融实验 (k=0.2) ==========')
epochs = 5000  # 原始仓库: 5000 epochs
checkpoint_interval_stage2 = 100  # LBFGS阶段每100个epoch保存一次
tic = time.time()
for epoch in range(epochi, epochs+epochi):
    loss = train(epoch, stage='LBFGS', total_epochs=epochs)
    print(f'loss_tot:{loss:.8f}')
    
    # ========== 训练过程可视化检查点（每100个epoch） ==========
    if epoch % checkpoint_interval_stage2 == 0:
        visualize_training_progress(
            model=model, epoch=epoch, stage='LBFGS', save_dir=training_vis_dir,
            Xs=Xs, Xe=Xe, Te=Te, exact_file='sode.dat',
            rhoref=1.0, uref=1.0, pref=1.0,
            device=cuda, dtype=dtype
        )
toc = time.time()
lbfgs_time = toc - tic
print(f'LBFGS训练时间: {lbfgs_time:.2f}s')

# 保存最后一次检查点
if training_vis_dir is not None:
    visualize_training_progress(
        model=model, epoch=epoch, stage='LBFGS', save_dir=training_vis_dir,
        Xs=Xs, Xe=Xe, Te=Te, exact_file='sode.dat',
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

# 尝试加载精确解进行对比
exact_file = 'sode.dat'
x_exact = None
if os.path.exists(exact_file):
    print(f"找到精确解文件: {exact_file}")
    exact_data = np.loadtxt(exact_file)
    x_exact = exact_data[:, 0]
    rho_exact = exact_data[:, 1]
    # 注意：sode.dat格式是 [x, rho, u, p]，但网络输出是 [rho, p, u]
    # 所以这里需要交换读取顺序，让 p_exact 读取第4列，u_exact 读取第3列
    p_exact = exact_data[:, 3]    # 第4列是p
    u_exact = exact_data[:, 2]    # 第3列是u
    
    # 插值到相同网格
    from scipy.interpolate import interp1d
    rho_exact_interp = interp1d(x_exact, rho_exact, kind='linear', 
                                bounds_error=False, fill_value='extrapolate')(x)
    p_exact_interp = interp1d(x_exact, p_exact, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')(x)
    u_exact_interp = interp1d(x_exact, u_exact, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')(x)
    
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
    'case': 'Sod_Shock_Tube',
    'Ts': Ts,
    'Te': Te,
    'Xs': Xs,
    'Xe': Xe,
    'Nx': Nx,
    'Nt': Nt,
    'crhoL': crhoL,
    'cuL': cuL,
    'cpL': cpL,
    'crhoR': crhoR,
    'cuR': cuR,
    'cpR': cpR,
    'model_layers': 6,  # 原始仓库: 6层隐藏层
    'model_neurons': 60,  # 原始仓库: 每层60个神经元
    'sampling_method': 'LHS',
    'ic_points': 100,  # 原始仓库: 100个初始点
    'interior_points': 10000,  # 原始仓库: 10000个PDE残差点
    'rh_points': 100,  # 原始仓库: 100个RH约束点（t=0.2）（已注释，未使用）
    'con_points': 100,  # 原始仓库: 100个守恒约束点（t=0和t=0.2各50个）（已注释，未使用）
    'loss_config': 'PINN_IC',  # 原始PINN测试：loss_pde + loss_ic
    'k_parameter': 0.0  # 权重函数参数：k=0（标准PINN，无权重函数）
}

# 调用通用保存函数（所有算例都可以用）
from utility import save_results
# 使用之前创建的 base_output_dir，避免重复创建
output_dir, metrics = save_results(
    case_name='Sod',
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

# ========== 保存结果统计（与Burgers_Compare格式一致）==========
from datetime import datetime
import os

result_file = 'result_torch.dat'
with open(result_file, 'w') as f:
    f.write('========== 实验配置 ==========\n')
    f.write(f'模式: Fourier (消融实验)\n')
    f.write(f'网络结构: 傅里叶特征映射 + MLP\n')
    f.write(f'傅里叶特征维度: 128 (输出256维)\n')
    f.write(f'傅里叶频率尺度: 10.0\n')
    f.write(f'权重函数k: 0.2\n')
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
print(f'模式: Fourier (消融实验)')
print(f'结果已保存到: {result_file}')
print(f'详细结果目录: {output_dir}')
print(f'===========================\n')

print('\n========== 结果对比说明 ==========')
print('评估标准：')
print('  1. 损失函数 (loss_pde, loss_ic等) - 越小越好')
print('  2. 相对L2误差 - 与精确解对比，越小越好')
print('     - < 5%:  优秀')
print('     - 5-10%: 良好')
print('     - 10-20%: 一般')
print('     - > 20%: 需要改进')
print('  3. 激波捕捉质量 - 激波位置准确、陡度足够、无振荡')
print(f'\n详细结果已保存到: {output_dir}')