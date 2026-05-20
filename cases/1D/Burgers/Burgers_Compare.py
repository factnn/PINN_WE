import torch
import torch.nn as nn
import numpy as np
import time
import scipy.io
import math
import sys
import os
import argparse

# 假设 Exact_burgers 在同一目录下，如果不是请调整
try:
    import Exact_burgers
except ImportError:
    print("Warning: Exact_burgers module not found. L2 error calculation might fail.")

# 添加PINNsrc路径以便导入riemann_solver
current_file_dir = os.path.dirname(os.path.abspath(__file__))
pinn_src_path = os.path.join(current_file_dir, '..', '..', '..', 'PINNsrc')
pinn_src_path = os.path.abspath(pinn_src_path)
if pinn_src_path not in sys.path:
    sys.path.insert(0, pinn_src_path)

# 尝试导入riemann_solver（因为riemann_solver.py在PINNsrc目录下）
try:
    from riemann_solver import BurgersSolver
except ImportError as e:
    print(f"Error: 无法导入riemann_solver: {e}")
    print(f"请确保riemann_solver.py在{pinn_src_path}目录下")
    sys.exit(1)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 尝试读取WENO数据，如果不存在则跳过（防止报错）
try:
    WENO = np.loadtxt('result50WENO.dat')
except OSError:
    WENO = None

# 添加matplotlib
import matplotlib.pyplot as plt

# ========== 参数解析 ==========
parser = argparse.ArgumentParser(description='Burgers PINN with/without Riemann Solver Comparison')
parser.add_argument('--mode', type=str, default='baseline',
                   choices=['baseline', 'riemann_torch', 'riemann_triton'],
                   help='Mode: baseline (no Riemann), riemann_torch, or riemann_triton')
parser.add_argument('--num_seed', type=int, default=1,
                   help='Number of random seeds to run')
parser.add_argument('--epochs', type=int, default=10,
                   help='Number of training epochs')
parser.add_argument('--nnode', type=int, default=26,
                   help='Number of spatial nodes')
parser.add_argument('--lambda_godunov', type=float, default=10.0,
                   help='Weight for Godunov loss (only for riemann modes)')
parser.add_argument('--output_dir', type=str, default=None,
                   help='Output directory (auto-generated if not specified)')
args = parser.parse_args()

mode = args.mode
num_seed = args.num_seed
epochs = args.epochs
nnode = args.nnode
lambda_godunov = args.lambda_godunov

print(f'\n========== 实验配置 ==========')
print(f'模式: {mode}')
print(f'种子数: {num_seed}')
print(f'Epochs: {epochs}')
print(f'节点数: {nnode}')
print(f'Godunov权重: {lambda_godunov}')
print(f'================================\n')

# ========== 输出文件名 ==========
# 创建时间戳文件夹（如果未指定）
if args.output_dir:
    output_dir = args.output_dir
else:
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"comparison_results_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

if mode == 'baseline':
    result_file = os.path.join(output_dir, 'result_baseline.dat')
elif mode == 'riemann_torch':
    result_file = os.path.join(output_dir, 'result_riemann_torch.dat')
else:  # riemann_triton
    result_file = os.path.join(output_dir, 'result_riemann_triton.dat')

f = open(result_file, "w+")
f.write('========== 实验配置 ==========\n')
f.write(f'模式: {mode}\n')
f.write(f'种子数: {num_seed}\n')
f.write(f'Epochs: {epochs}\n')
f.write(f'节点数: {nnode}\n')
f.write(f'Godunov权重: {lambda_godunov}\n')
f.write(f'================================\n')
f.write(f'###########nnode: {nnode} ########### \n')

L2_smooth_ave = []
L2_shock_ave = []
L2_ave = []
L2_max_ave = []
loss_ave = []
train_time_ave = []
godunov_time_ave = []  # 只在riemann模式下有
adam_train_time = 0.0
lbfgs_train_time = 0.0

starttime = time.time()

for i in range(num_seed):
    seed = i
    setup_seed(seed)
    f.write('****seed: %d ***** \n' % seed)

    def gradients(outputs, inputs):
        return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)

    def to_numpy(input):
        if isinstance(input, torch.Tensor):
            return input.detach().cpu().numpy()
        elif isinstance(input, np.ndarray):
            return input
        else:
            raise TypeError('Unknown type of input, expected torch.Tensor or ' \
                            'np.ndarray, but got {}'.format(type(input)))

    def IC(x):
        N = len(x)
        u_init = np.zeros((x.shape[0]))
        for i in range(N):
            u_init[i] = -np.sin(np.pi*(x[i,1]-1))
        return u_init

    # ========== 神经网络定义 ==========
    class DNN(nn.Module):
        def __init__(self):
            super(DNN, self).__init__()
            self.net = nn.Sequential()
            self.net.add_module('Linear_layer_1', nn.Linear(2, 30))
            self.net.add_module('Tanh_layer_1', nn.Tanh())

            for num in range(2, 4):
                self.net.add_module('Linear_layer_%d' % (num), nn.Linear(30, 30))
                self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())
            self.net.add_module('Linear_layer_final', nn.Linear(30, 1))

        def forward(self, x):
            return self.net(x)

        def loss_pde(self, x):
            """
            PDE残差损失（带自适应激波捕捉）
            
            使用自适应权重函数：
            - 光滑区域 (u_x >= 0): d ≈ 1, PDE loss权重正常
            - 压缩/激波区域 (u_x < 0): d > 1, PDE loss权重降低
            
            返回:
                loss_total: 加权后的总PDE loss
                loss_smooth: 光滑区域的PDE loss (u_x >= -epsilon)
                loss_shock: 激波区域的PDE loss (u_x < -epsilon)
            """
            y = self.net(x)
            u = y[:, 0:1] 

            U = u**2/2 

            dU_g = gradients(U, x)[0]
            U_x = dU_g[:, 1:]
            du_g = gradients(u, x)[0]
            u_t, u_x = du_g[:, :1], du_g[:, 1:]

            # ========== 自适应权重函数（从Blast复制） ==========
            # 参数k控制自适应强度：k=0时退化为标准PINN
            # 光滑区域(u_x >= 0): d = 1/(k*(abs(u_x)-u_x)+1) = 1
            # 压缩区域(u_x < 0): d = 1/(k*(abs(u_x)-u_x)+1) = 1/(k*(-2*u_x)+1) > 1
            k = 0.2  # 与Blast一致
            d = 1.0 / (k * (torch.abs(u_x) - u_x) + 1.0)

            # 加权PDE残差
            f_weighted = (((u_t + U_x) / d) ** 2).mean()

            # ========== 分别计算光滑和激波区域的loss（用于监控） ==========
            epsilon = 0.01  # 判断光滑/激波的阈值
            mask_smooth = (u_x >= -epsilon).flatten()
            mask_shock = (u_x < -epsilon).flatten()

            if mask_smooth.sum() > 0:
                f_smooth = (((u_t + U_x) ** 2).flatten()[mask_smooth]).mean()
            else:
                f_smooth = torch.tensor(0.0, device=x.device)

            if mask_shock.sum() > 0:
                f_shock = (((u_t + U_x) ** 2).flatten()[mask_shock]).mean()
            else:
                f_shock = torch.tensor(0.0, device=x.device)

            return f_weighted, f_smooth, f_shock

        def loss_godunov(self, x_int, x_sorted_np, x_int_np, mode):
            """
            Godunov黎曼求解器损失（带独立激波捕捉器，与Blast的RH loss逻辑一致）
            
            策略（与Blast的loss_rh一致）：
            - 只在激波区域（相邻点速度差大）使用Godunov loss
            - 使用eta作为权重函数：基于相邻点的速度差
            - eta = clamp(abs(u_left - u_right) - threshold, min=0)
            """
            # 获取网络预测的u值
            y = self.net(x_int)
            u_pred = y[:, 0:1] 

            # 将x_int转为numpy用于排序
            # 注意：频繁的CPU-GPU传输和排序会影响整体训练速度
            x_int_np_local = to_numpy(x_int) 

            # 按空间坐标排序
            sort_idx = np.argsort(x_int_np_local[:, 1])
            # x_sorted = x_int_np_local[sort_idx] # 这个变量暂时没用到
            u_sorted = u_pred[torch.tensor(sort_idx, device=x_int.device)] 

            # 构造相邻点对（左状态和右状态）
            n_points = len(u_sorted)
            if n_points < 2:
                return torch.tensor(0.0, device=x_int.device) 

            # 创建左状态和右状态
            u_left = u_sorted[:-1].flatten()
            u_right = u_sorted[1:].flatten() 

            # ========== 独立激波捕捉器（与Blast的loss_rh逻辑一致）==========
            # 基于相邻点的速度差，只在激波区域eta>0
            # 类比Blast: eta = clamp(abs(p-pl)-0.2, min=0) * clamp(abs(u-ul)-0.2, min=0)
            # Burgers只有u，所以简化为：eta = clamp(abs(u_left - u_right) - threshold, min=0)
            delta_u = torch.abs(u_left - u_right)
            threshold = 0.2  # 与Blast一致
            eta = torch.clamp(delta_u - threshold, min=0)

            # 使用Godunov求解器计算数值通量（根据模式选择后端）
            if mode == 'riemann_torch':
                flux_godunov = BurgersSolver.godunov_flux(u_left, u_right, backend='torch')
            elif mode == 'riemann_triton':
                flux_godunov = BurgersSolver.godunov_flux(u_left, u_right, backend='triton')
            else:
                # [修复] 修正了f-string缺少引号的问题
                raise ValueError(f"Unknown mode: {mode}") 

            # 计算网络预测的通量 f(u) = 0.5 * u^2
            flux_pred_left = 0.5 * u_left**2
            flux_pred_right = 0.5 * u_right**2 

            # 计算损失：(黎曼通量 - 平均通量)^2，应用激波捕捉器权重eta
            # 只在激波区域（eta>0）应用Godunov loss
            loss_unweighted = ((flux_godunov - 0.5*(flux_pred_left + flux_pred_right))**2)
            loss_weighted = (loss_unweighted * eta).mean() 

            return loss_weighted

        def loss_ic(self, x_ic, u_ic):
            y_ic = self.net(x_ic)
            u_ic_nn = y_ic[:, 0]
            loss_ics = ((u_ic_nn - u_ic) ** 2).mean()
            return loss_ics

    # ========== 数据准备 ==========
    device = torch.device('cuda')
    num_x = nnode
    num_t = nnode
    num_i_train = nnode
    num_f_train = nnode*nnode
    x = np.linspace(0, 2, num_x)
    t = np.linspace(0, 1, num_t)
    t_grid, x_grid = np.meshgrid(t, x)
    T = t_grid.flatten()[:, None]
    X = x_grid.flatten()[:, None]

    id_ic = np.random.choice(num_x, num_i_train, replace=False)
    id_f = np.random.choice(num_x*num_t, num_f_train, replace=False)

    x_ic = x_grid[id_ic, 0][:, None]
    t_ic = t_grid[id_ic, 0][:, None]
    x_ic_train = np.hstack((t_ic, x_ic))
    u_ic_train = IC(x_ic_train)

    x_int = X[:, 0][id_f, None]
    t_int = T[:, 0][id_f, None]
    x_int_train = np.hstack((t_int, x_int))

    u_ic_train = IC(x_ic_train)

    # 保存numpy版本用于排序
    x_int_np = x_int_train.copy()
    x_sorted_np = x_int_train.copy()

    x_ic = torch.tensor(x_ic_train, dtype=torch.float32).to(device)
    x_int = torch.tensor(x_int_train, requires_grad=True, dtype=torch.float32).to(device)
    u_ic = torch.tensor(u_ic_train, dtype=torch.float32).to(device)

    # ========== 模型创建 ==========
    model = DNN().to(device)

    # ========== 训练时间统计 ==========
    seed_train_time = 0.0
    seed_godunov_time = 0.0
    adam_train_time = 0.0
    lbfgs_train_time = 0.0

    # ========== 第一阶段：Adam训练 ==========
    print('='*60)
    print('第一阶段：Adam训练 (lr=0.001)')
    print('='*60)
    adam_epochs = 100000  # 最多100000 epochs
    adam_early_stop_loss = 0.05  # 早停条件
    checkpoint_interval_adam = 1000  # 每1000个epoch打印

    optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001)

    def train_adam(epoch):
        model.it = epoch
        optimizer_adam.zero_grad()
        loss_pde, loss_pde_smooth, loss_pde_shock = model.loss_pde(x_int)
        loss_ic = model.loss_ic(x_ic, u_ic)

        if mode == 'baseline':
            loss = loss_pde + 10*loss_ic
            loss_godunov_val = 0.0
        else:
            loss_godunov_val = model.loss_godunov(x_int, x_sorted_np, x_int_np, mode)
            loss = loss_pde + 10*loss_ic + lambda_godunov*loss_godunov_val

        loss.backward()
        optimizer_adam.step()
        return loss.item()

    tic_adam = time.time()
    for epoch in range(1, adam_epochs+1):
        loss_val = train_adam(epoch)

        if epoch % checkpoint_interval_adam == 0 or loss_val < adam_early_stop_loss:
            print(f'Adam epoch {epoch}: loss={loss_val:.6f}')

        if loss_val < adam_early_stop_loss:
            print(f'达到早停条件 (loss < {adam_early_stop_loss})，第一阶段训练结束')
            break

    adam_train_time = time.time() - tic_adam
    seed_train_time += adam_train_time
    print(f'Adam训练完成，耗时: {adam_train_time:.2f}s')

    # ========== 第二阶段：LBFGS训练 ==========
    print('='*60)
    print('第二阶段：LBFGS训练 (lr=0.1)')
    print('='*60)
    lbfgs_epochs = 5000  # 5000 epochs
    checkpoint_interval_lbfgs = 100  # 每100个epoch打印

    optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.1,
                                        max_iter=20,
                                        max_eval=None,
                                        tolerance_grad=1e-05,
                                        tolerance_change=1e-09,
                                        history_size=100,
                                        line_search_fn='strong_wolfe')

    def train_lbfgs(epoch):
        model.it = epoch
        epoch_godunov_time = [0.0]

        def closure():
            optimizer_lbfgs.zero_grad()
            loss_pde, loss_pde_smooth, loss_pde_shock = model.loss_pde(x_int)
            loss_ic = model.loss_ic(x_ic, u_ic)

            if mode == 'baseline':
                loss = loss_pde + 10*loss_ic
                loss_godunov_val = 0.0
                current_godunov_time = 0.0
            else:
                torch.cuda.synchronize()
                start_godunov = time.time()
                loss_godunov_val = model.loss_godunov(x_int, x_sorted_np, x_int_np, mode)
                torch.cuda.synchronize()
                current_godunov_time = time.time() - start_godunov
                epoch_godunov_time[0] += current_godunov_time

                loss = loss_pde + 10*loss_ic + lambda_godunov*loss_godunov_val

            if mode == 'baseline':
                print(f'  LBFGS epoch {epoch}: loss_pde={loss_pde:.6f}, loss_ic={loss_ic:.6f}, loss={loss:.6f}')
            else:
                print(f'  LBFGS epoch {epoch}: loss_pde={loss_pde:.6f}, loss_ic={loss_ic:.6f}, loss_g={loss_godunov_val:.6f}, total={loss:.6f}')

            loss.backward()
            return loss

        loss = optimizer_lbfgs.step(closure)
        return loss.item(), epoch_godunov_time[0]

    tic_lbfgs = time.time()
    for epoch in range(1, lbfgs_epochs+1):
        epoch_start = time.time()
        loss_val, godunov_time = train_lbfgs(epoch)
        epoch_time = time.time() - epoch_start

        lbfgs_train_time += epoch_time
        seed_godunov_time += godunov_time

        if epoch % checkpoint_interval_lbfgs == 0:
            print(f'LBFGS epoch {epoch}: loss={loss_val:.6f}, time={epoch_time:.3f}s')

    lbfgs_train_time = time.time() - tic_lbfgs
    seed_train_time += lbfgs_train_time
    print(f'LBFGS训练完成，耗时: {lbfgs_train_time:.2f}s')

    print('='*60)
    print(f'训练完成！总耗时: {seed_train_time:.2f}s (Adam: {adam_train_time:.2f}s + LBFGS: {lbfgs_train_time:.2f}s)')
    print('='*60)

    # ========== 评估 ==========
    x = np.linspace(0.0, 2.0, nnode)
    t = np.linspace(1.0, 1.0, 1)
    t_grid, x_grid = np.meshgrid(t, x)
    T = t_grid.flatten()[:, None]
    X = x_grid.flatten()[:, None]
    x_test = np.hstack((T, X))
    x_test = torch.tensor(x_test, requires_grad=True, dtype=torch.float32).to(device)
    u_pred = to_numpy(model(x_test))

    # 记录最终loss
    final_loss = seed_train_time  # 用训练时间作为"loss"的占位符，因为loss在两阶段训练中没有统一记录

    # 注意：res_pde 和 lambda_pde 方法在 DNN 类中没有定义，如果不需要可以注释掉
    # res = to_numpy(model.res_pde(x_test))
    # d = to_numpy(model.lambda_pde(x_test))

    # ========== 计算精度指标 ==========
    if 'Exact_burgers' in sys.modules:
        y_e = Exact_burgers.Exact_Burgers(x)
        L2_error = np.sqrt(np.sum((u_pred[:,0]-y_e)**2)/nnode)
        
        num = 0
        num_shock = 0
        sum_err = 0.0
        sum_shock = 0
        for i in range(nnode):
          if abs(x[i]-1)> 0.05:
            sum_err = sum_err +  np.sum((u_pred[i,0]-y_e[i])**2)
            num = num + 1
          else:
            sum_shock = sum_shock +  np.sum((u_pred[i,0]-y_e[i])**2)
            num_shock = num_shock + 1

        if num > 0: sum_err = sum_err/num
        if num_shock > 0: sum_shock = sum_shock/num_shock
        
        L2_error_smooth = np.sqrt(sum_err)
        L2_error_shock = np.sqrt(sum_shock)
        L2_Max = (np.max(np.abs(u_pred[:,0]-y_e)))

        L2_max_ave.append(L2_Max)
        L2_ave.append(L2_error)
        L2_smooth_ave.append(L2_error_smooth)
        L2_shock_ave.append(L2_error_shock)
        
        f.write(f'L2 error: {L2_error:.6e}\n')
        f.write(f'L2_smooth error: {L2_error_smooth:.6e}\n')
        f.write(f'L2_shock error: {L2_error_shock:.6e}\n')
        f.write(f'L2_max error: {L2_Max:.6e}\n')
    else:
        # 如果没有精确解文件，填0防止报错
        L2_ave.append(0)
        L2_max_ave.append(0)
        L2_smooth_ave.append(0)
        L2_shock_ave.append(0)
        print("Warning: Skipping L2 Error calc due to missing Exact_burgers")

    loss_ave.append(final_loss)
    train_time_ave.append(seed_train_time)

    if mode != 'baseline':
        godunov_time_ave.append(seed_godunov_time)

    # ========== 可视化结果（带WENO精确解对比）==========
    print('\n开始生成可视化结果...')

    # 创建密集测试网格用于绘图（在最终时刻 t=1.0）
    x_plot = np.linspace(0.0, 2.0, 200)
    t_plot = np.full_like(x_plot, 1.0)
    x_test_plot = np.hstack((t_plot[:, None], x_plot[:, None]))
    x_test_plot_tensor = torch.tensor(x_test_plot, dtype=torch.float32).to(device)

    # 预测
    model.eval()
    with torch.no_grad():
        u_pred_plot = to_numpy(model(x_test_plot_tensor)).flatten()

    print("预测完成")

    # 尝试加载WENO精确解进行对比
    weno_file = 'result50WENO.dat'
    x_weno = None
    u_weno = None
    u_weno_interp = None
    if os.path.exists(weno_file):
        print(f"找到WENO精确解文件: {weno_file}")
        weno_data = np.loadtxt(weno_file)
        x_weno = weno_data[:, 0]
        u_weno = weno_data[:, 1]

        # 插值到相同网格
        try:
            from scipy.interpolate import interp1d
            u_weno_interp = interp1d(x_weno, u_weno, kind='linear',
                                      bounds_error=False, fill_value='extrapolate')(x_plot)
        except Exception as e:
            print(f"插值失败: {e}")
            u_weno_interp = None

        # 计算绘图误差
        if u_weno_interp is not None:
            l2_error_plot = np.sqrt(np.mean((u_pred_plot - u_weno_interp)**2)) / np.sqrt(np.mean(u_weno_interp**2))
            max_error_plot = np.max(np.abs(u_pred_plot - u_weno_interp))
            print(f"绘图误差: L2={l2_error_plot:.2%}, Max={max_error_plot:.2%}")
        else:
            l2_error_plot = None
            max_error_plot = None
    else:
        u_weno_interp = None
        l2_error_plot = None
        max_error_plot = None

    # 绘图
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(x_plot, u_pred_plot, 'b-', linewidth=2.5, label=f'{mode.upper()} Prediction', zorder=3)
    if x_weno is not None:
        ax.plot(x_weno, u_weno, 'r--', linewidth=2, label='WENO Exact', alpha=0.8, zorder=2)
        ax.fill_between(x_plot, u_pred_plot, u_weno_interp, alpha=0.2, color='gray', label='Error')
        ax.text(0.02, 0.98, f'L2 Error: {l2_error_plot:.2%}\nMax Error: {max_error_plot:.2%}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 标记激波区域（x=1附近）
    ax.axvspan(0.95, 1.05, alpha=0.2, color='red', label='Shock Region (x=1)')

    ax.set_xlabel('Position x', fontsize=14, fontweight='bold')
    ax.set_ylabel('Velocity u', fontsize=14, fontweight='bold')
    ax.set_title(f'Burgers Equation (t=1.0, nnode={nnode}, epochs={epochs})\nMode: {mode}', fontsize=15, fontweight='bold')
    ax.set_xlim(0.0, 2.0)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')

    plt.tight_layout()

    # 保存图片到时间戳文件夹
    plot_filename = os.path.join(output_dir, f'burgers_{mode}_nnode{nnode}_epochs{epochs}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f'\n结果图片已保存到: {plot_filename}')
    plt.close()

    f.write(f'train_time: {seed_train_time:.6e}s\n')
    f.write(f'  - Adam time: {adam_train_time:.6e}s\n')
    f.write(f'  - LBFGS time: {lbfgs_train_time:.6e}s\n')
    if mode != 'baseline':
        f.write(f'godunov_time: {seed_godunov_time:.6e}s\n')

# ========== 统计结果 ==========
f.write('\n========== 统计结果 ==========\n')
f.write(f'L2_max_ave: {np.mean(L2_max_ave):.6e}, L2_max_top : {np.max(L2_max_ave):.6e}, L2_max_bottom {np.min(L2_max_ave):.6e}\n')
f.write(f'L2_ave: {np.mean(L2_ave):.6e}, L2_ave_top: {np.max(L2_ave):.6e}, L2_ave_bottom: {np.min(L2_ave):.6e} \n')
f.write(f'L2_smooth_ave: {np.mean(L2_smooth_ave):.6e}, L2_smooth_ave_top: {np.max(L2_smooth_ave):.6e}, L2_smooth_bottom: {np.min(L2_smooth_ave):.6e} \n')
f.write(f'L2_shock_ave: {np.mean(L2_shock_ave):.6e}, L2_shock_top: {np.max(L2_shock_ave):.6e}, L2_shock_bottom: {np.min(L2_shock_ave):.6e} \n')
f.write(f'train_time_ave: {np.mean(train_time_ave):.6e}s, train_time_top: {np.max(train_time_ave):.6e}s, train_time_bottom: {np.min(train_time_ave):.6e}s\n')
f.write(f'  - Adam time avg: {adam_train_time:.6e}s\n')
f.write(f'  - LBFGS time avg: {lbfgs_train_time:.6e}s\n')
if mode != 'baseline':
    f.write(f'godunov_time_ave: {np.mean(godunov_time_ave):.6e}s, godunov_time_top: {np.max(godunov_time_ave):.6e}s, godunov_time_bottom: {np.min(godunov_time_ave):.6e}s\n')
f.write('================================\n')

endtime = time.time()
total_time = (endtime - starttime)/num_seed
f.write(f'\n total_time: {total_time:.6e}s\n')
f.close()

print(f'\n========== 完成 ==========')
print(f'模式: {mode}')
print(f'结果已保存到: {result_file}')
print(f'===========================\n')