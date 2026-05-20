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
from utility import select_gpu, create_output_dir, save_results, plot_burgers_results
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from smt.sampling_methods import LHS

dtype = torch.float64
setup_seed(2)

# ========== 粘性Burgers方程参数 ==========
# 方程：u_t + u*u_x = nu*u_xx
# 初始条件：u(x,0) = -sin(π(x-1))
# 边界条件：u(0,t) = u(2,t) = 0（Dirichlet边界条件）
# 精确解：可以通过Cole-Hopf变换得到
Ts = 0
Te = 1.0  # 终止时间
Xs = 0
Xe = 2.0  # 空间域[0, 2]

# 粘性系数（关键参数）
nu = 0.01  # 粘性系数，nu越小越接近无粘（激波），nu越大越光滑

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

# ========== 精确解（Cole-Hopf变换）==========
def Exact_Burgers(x, t, nu):
    """
    粘性Burgers方程的精确解（通过Cole-Hopf变换）
    u_t + u*u_x = nu*u_xx
    初始条件：u(x,0) = -sin(π(x-1))

    Cole-Hopf变换：u = -2*nu * (∂φ/∂x) / φ
    其中 φ 满足热方程，可以用积分表示
    """
    N = len(x)
    u_exact = np.zeros(N)

    # 积分点数（用于数值积分）
    n_int = 500
    eta = np.linspace(Xs, Xe, n_int)
    deta = eta[1] - eta[0]

    for i in range(N):
        xi = x[i]

        # 初始条件 u0(eta) = -sin(π(eta-1))
        u0 = -np.sin(np.pi * (eta - 1))

        # Cole-Hopf变换的核函数
        # φ(x,t) = ∫ exp(-(x-η)²/(4νt) - (1/(2ν))∫[0→η] u0(s) ds) dη

        # 计算 ∫[0→η] u0(s) ds（初始条件的积分）
        # u0(s) = -sin(π(s-1))，积分得：(1/π)*cos(π(s-1))
        integral_u0 = (1/np.pi) * (np.cos(np.pi * (eta - 1)) - np.cos(np.pi * (0 - 1)))

        # 计算指数项
        if t < 1e-10:
            # t=0 时，直接返回初始条件
            u_exact[i] = -np.sin(np.pi * (xi - 1))
        else:
            # 高斯核
            gaussian = np.exp(-(xi - eta)**2 / (4 * nu * t))
            # Cole-Hopf 指数项
            hopf_exp = np.exp(-integral_u0 / (2 * nu))

            # φ 的积分
            phi = np.sum(gaussian * hopf_exp * deta)

            # ∂φ/∂x 的积分（对高斯核求导）
            dphi_dx = np.sum(-(xi - eta) / (2 * nu * t) * gaussian * hopf_exp * deta)

            # u = -2*nu * (∂φ/∂x) / φ
            if phi > 1e-10:
                u_exact[i] = -2 * nu * dphi_dx / phi
            else:
                u_exact[i] = 0.0

    return u_exact

# ========== 训练函数 ==========
def train(epoch, stage='Adam', total_epochs=None):
    """
    带边界条件的PINN训练函数（粘性Burgers方程）
    - 损失函数：loss_pde + loss_ic + loss_bc
    - 边界条件：u(0,t) = u(2,t) = 0
    """
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    # 标准PINN配置：k=0
    k = 0

    def closure():
        optimizer.zero_grad()
        loss_pde = actual_model.loss_pde_burgers(x_int, nu=nu, k=k)
        loss_ic = actual_model.loss_ic_burgers(x_ic, u_ic)

        # 边界条件损失：u(0,t) = 0 和 u(2,t) = 0
        u_bc_left = actual_model(x_bc_left)
        u_bc_right = actual_model(x_bc_right)
        loss_bc = torch.mean(u_bc_left**2) + torch.mean(u_bc_right**2)

        # 带边界条件的PINN损失
        loss = loss_pde + 10*loss_ic + 10*loss_bc

        print(f'epoch {epoch} loss:{loss:.8f}, loss_pde:{loss_pde:.8f}, loss_ic:{loss_ic:.8f}, loss_bc:{loss_bc:.8f}')
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
x_int = sampling(10000)

# 边界条件采样（u(0,t) = 0 和 u(2,t) = 0）
Nbc = 100
t_bc = np.random.uniform(0, Te, Nbc)
# 左边界 x=0
x_bc_left = np.stack([t_bc, np.full(Nbc, Xs)], axis=1)
# 右边界 x=2
x_bc_right = np.stack([t_bc, np.full(Nbc, Xe)], axis=1)

print(f'采样完成：初始条件{len(x_ic)}，内部点{len(x_int)}，边界条件{Nbc*2}')

# 保存numpy版本用于可视化
x_ic_np = x_ic.copy()
u_ic_np = u_ic.copy()

# 转换为Tensor
x_ic = torch.tensor(x_ic, requires_grad=True, dtype=dtype).to(cuda)
x_int = torch.tensor(x_int, requires_grad=True, dtype=dtype).to(cuda)
x_bc_left = torch.tensor(x_bc_left, requires_grad=True, dtype=dtype).to(cuda)
x_bc_right = torch.tensor(x_bc_right, requires_grad=True, dtype=dtype).to(cuda)

u_ic = torch.tensor(u_ic, dtype=dtype).to(cuda)

# ========== 网络结构（使用Burgers专用模型）==========
model = PINNs_Burgers(Nl=6, Nn=60).to(cuda).double()

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
epochs = 100000
loss_history = []
tic_total = time.time()

# ========== 创建输出目录 ==========
base_output_dir = create_output_dir('Burgers_Viscous', gpu_count=num_gpus, tag=f'with_BC_nu{nu}')
print(f'\n[输出目录] {base_output_dir}')

# ========== Adam阶段训练 ==========
print(f'========== Adam阶段：带边界条件的PINN训练（k=0，粘性系数nu={nu}） ==========')
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

# 生成评估网格
nx = 200
x_test_np = np.linspace(Xs, Xe, nx)
t_test_np = np.full(nx, Te)
x_test_input = np.stack([t_test_np, x_test_np], axis=1)
x_test = torch.tensor(x_test_input, dtype=dtype).to(cuda)

actual_model.eval()
with torch.no_grad():
    u_pred = to_numpy(actual_model(x_test)).flatten()

print('预测完成')

# ========== 计算精确解 ==========
print('计算精确解...')
u_exact = Exact_Burgers(x_test_np, Te, nu)
print('精确解计算完成')

# ========== 使用Burgers专用绘图函数 ==========
print('\n生成Burgers结果图...')
burgers_metrics = plot_burgers_results(
    u_pred=u_pred,
    u_exact=u_exact,
    x=x_test_np,
    t=Te,
    nu=nu,
    output_dir=base_output_dir,
    check_boundary=True,
    plot_spacetime={
        'model': actual_model,
        'device': cuda,
        'dtype': dtype,
        'nx': 200,
        'nt': 100,
        't_max': Te,
        'x_min': Xs,
        'x_max': Xe
    }
)
)

# ========== 准备结果字典（用于save_results保存数据）==========
# Burgers方程只有u，为了兼容save_results，将u映射到rho
pred_dict = {
    'x': x_test_np,
    'rho': u_pred,  # 将u映射为rho以兼容save_results
    'p': np.ones_like(u_pred),  # 占位符
    'u': np.ones_like(u_pred),  # 占位符
    't': Te
}

exact_dict = {
    'x': x_test_np,
    'rho': u_exact,  # 将u映射为rho以兼容save_results
    'p': np.ones_like(u_exact),  # 占位符
    'u': np.ones_like(u_exact),  # 占位符
    't': Te
}

# ========== 保存结果 ==========
print('保存结果...')
output_dir, metrics = save_results(
    case_name='Burgers_Viscous',
    model=model,
    pred_dict=pred_dict,
    loss_history=loss_history,
    exact_dict=exact_dict,
    training_time=training_time,
    gpu_count=num_gpus,
    tag=f'with_BC_nu{nu}',
    stage_boundary=stage_boundary,
    save_plot=False,  # 不使用Euler通用绘图，只用Burgers专用绘图
    output_dir=base_output_dir  # 使用已创建的输出目录
)

print(f'结果已保存到: {output_dir}')
print(f'\n========== 训练完成 ==========')
print(f'粘性系数 nu = {nu}')
print(f'训练时间: {training_time:.2f}秒')

# 打印Burgers专用的误差统计
if burgers_metrics:
    print(f'\n误差统计:')
    print(f'  L2 误差: {burgers_metrics["l2_error"]:.6e}')
    print(f'  相对L2误差: {burgers_metrics["rel_l2_error"]*100:.4f}%')
    print(f'  最大误差: {burgers_metrics["max_error"]:.6e}')
    print(f'\n边界值检查:')
    print(f'  u(x=0) = {burgers_metrics["u_left_boundary"]:.6f} (应该≈0)')
    print(f'  u(x=2) = {burgers_metrics["u_right_boundary"]:.6f} (应该≈0)')





