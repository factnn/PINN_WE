"""
PINN with Experimental Data Fusion
在原有1.py基础上添加实验数据融合功能
"""
from PINNs import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS
dtype=torch.float64
setup_seed(2)

# ========== 原有参数 ==========
Ts = 0
Te = 0.2
Xs = 0
Xe = 1
Nx = 100
Nt = 100
dt = 0.002
dx = 0.01

crhoL = 1.0
cuL = 0.0
cpL = 1.0

crhoR = 0.125
cuR = 0
cpR = 0.1
setup_seed(7)

# ========== 实验数据加载（示例）==========
def load_experimental_data(data_file=None):
    """
    加载实验数据
    
    如果data_file为None，生成模拟数据作为示例
    实际使用时，替换为你的真实数据加载函数
    """
    if data_file is None:
        # 示例：生成一些模拟的实验数据点
        # 实际使用时删除这部分，改为从文件读取
        print("使用模拟数据作为示例...")
        N_exp = 20  # 实验数据点数量
        
        # 在激波附近和关键位置采样
        x_exp = []
        rho_exp = []
        u_exp = []
        p_exp = []
        
        # 在几个关键时刻采样
        for t in [0.05, 0.1, 0.15, 0.2]:
            # 在空间上均匀采样
            x_positions = np.linspace(Xs, Xe, 5)
            for x in x_positions:
                x_exp.append([t, x])
                # 这里用简单的解析解或WENO结果作为"实验数据"
                # 实际使用时替换为真实观测值
                if x < 0.5:
                    rho_exp.append(1.0)
                    u_exp.append(0.0)
                    p_exp.append(1.0)
                else:
                    rho_exp.append(0.125)
                    u_exp.append(0.0)
                    p_exp.append(0.1)
        
        x_exp = np.array(x_exp)
        rho_exp = np.array(rho_exp)
        u_exp = np.array(u_exp)
        p_exp = np.array(p_exp)
    else:
        # 从文件读取数据
        # 假设格式：t, x, rho, u, p
        data = np.loadtxt(data_file)
        x_exp = data[:, [0, 1]]  # t, x
        rho_exp = data[:, 2]
        u_exp = data[:, 3]
        p_exp = data[:, 4]
    
    return x_exp, rho_exp, u_exp, p_exp

# 加载实验数据
x_exp, rho_exp, u_exp, p_exp = load_experimental_data()
print(f"加载了 {len(x_exp)} 个实验数据点")

# ========== 扩展PINNs类添加数据损失 ==========
class PINNs_WE_Euler_1D_WithData(PINNs_WE_Euler_1D):
    """带数据融合的PINN"""
    
    def loss_data(self, x_data, rho_data, u_data, p_data):
        """
        实验数据拟合损失
        
        Args:
            x_data: 数据点坐标 [t, x], shape: (N_data, 2)
            rho_data, u_data, p_data: 实验观测值, shape: (N_data,)
        """
        y_pred = self.net(x_data)
        rho_pred = y_pred[:, 0]
        p_pred = y_pred[:, 1]
        u_pred = y_pred[:, 2]
        
        loss_data = ((rho_pred - rho_data)**2).mean() + \
                    ((u_pred - u_data)**2).mean() + \
                    ((p_pred - p_data)**2).mean()
        
        return loss_data
    
    def loss_data_weighted(self, x_data, rho_data, u_data, p_data,
                          sigma_rho=0.01, sigma_u=0.01, sigma_p=0.01):
        """
        带不确定性的数据融合（考虑测量误差）
        
        Args:
            sigma: 测量误差的标准差
        """
        y_pred = self.net(x_data)
        rho_pred = y_pred[:, 0]
        p_pred = y_pred[:, 1]
        u_pred = y_pred[:, 2]
        
        # 加权损失（误差大的数据权重小）
        loss_data = ((rho_pred - rho_data)**2 / (sigma_rho**2 + 1e-6)).mean() + \
                    ((u_pred - u_data)**2 / (sigma_u**2 + 1e-6)).mean() + \
                    ((p_pred - p_data)**2 / (sigma_p**2 + 1e-6)).mean()
        
        return loss_data

# ========== 训练函数（带数据融合）==========
def train(epoch, lambda_data=100.0):
    """
    训练函数，包含数据融合
    
    Args:
        lambda_data: 数据损失的权重
    """
    def closure():
        optimizer.zero_grad()
        
        # 原有损失
        loss_pde = model.loss_pde(x_int)
        loss_ic = model.loss_ic(x_ic, rho_ic, u_ic, p_ic)
        loss_rh = model.loss_rh(xrh, xrhL)
        loss_con = model.loss_con(x_en, x_ic, crhoL, cuL, cpL, crhoR, cuR, cpR, Te-Ts)
        
        # 新增：数据拟合损失
        loss_data = model.loss_data(x_exp_tensor, rho_exp_tensor, u_exp_tensor, p_exp_tensor)
        
        # 总损失
        loss = loss_pde + 10*loss_ic + 10*loss_rh + 10*loss_con + lambda_data*loss_data
        
        if epoch % 10 == 0:
            print(f'epoch {epoch} loss_pde:{loss_pde:.8f}, loss_ic:{loss_ic:.8f}, '
                  f'loss_rh:{loss_rh:.8f}, loss_con:{loss_con:.8f}, loss_data:{loss_data:.8f}')
        
        loss.backward()
        return loss
    loss = optimizer.step(closure)
    return loss

# ========== 生成训练数据 ==========
xlimits = np.array([[0., 0], [0, Xe]])
sampling = LHS(xlimits=xlimits)
x_ic = sampling(100)
rho_ic, u_ic, p_ic = IC_Riemann_1D(x_ic, crhoL, cuL, cpL, crhoR, cuR, cpR)

xlimits = np.array([[0., Te], [0, Xe]])
sampling = LHS(xlimits=xlimits)
x_int = sampling(10000)

xrh, xrhL, xrhR, xrhP, xrhPL, xrhPR = Pertur_1D(x_ic, Te, dt, dx)
x_en = Move_Time_1D(x_ic, Te)

# 转换为tensor
x_ic = torch.tensor(x_ic, requires_grad=True, dtype=dtype).to(cuda)
x_int = torch.tensor(x_int, requires_grad=True, dtype=dtype).to(cuda)
x_en = torch.tensor(x_en, dtype=dtype).to(cuda)
xrh = torch.tensor(xrh, dtype=dtype).to(cuda)
xrhL = torch.tensor(xrhL, dtype=dtype).to(cuda)

rho_ic = torch.tensor(rho_ic, dtype=dtype).to(cuda)
u_ic = torch.tensor(u_ic, dtype=dtype).to(cuda)
p_ic = torch.tensor(p_ic, dtype=dtype).to(cuda)

# 实验数据转换为tensor
x_exp_tensor = torch.tensor(x_exp, dtype=dtype).to(cuda)
rho_exp_tensor = torch.tensor(rho_exp, dtype=dtype).to(cuda)
u_exp_tensor = torch.tensor(u_exp, dtype=dtype).to(cuda)
p_exp_tensor = torch.tensor(p_exp, dtype=dtype).to(cuda)

# ========== 创建模型 ==========
model = PINNs_WE_Euler_1D_WithData(Nl=6, Nn=60).to(cuda).double()
print('Start training with data fusion...')
print(f'实验数据点数量: {len(x_exp)}')
print(f'数据损失权重: lambda_data (可调整)')

# ========== 训练 ==========
epoch = 0
epochi = epoch
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 100000

# 自适应权重策略（可选）
# 训练初期更重视PDE，后期更重视数据
lambda_data_schedule = lambda epoch: 10.0 if epoch < 1000 else (50.0 if epoch < 10000 else 100.0)

loss_history = []
tic = time.time()
for epoch in range(1+epochi, epochs+epochi):
    # 使用自适应权重（或固定权重）
    lambda_data = lambda_data_schedule(epoch)  # 或固定值：lambda_data = 100.0
    loss = train(epoch, lambda_data)
    print(f'loss_tot:{loss:.8f}')
    loss_history.append(to_numpy(loss))
    if loss < 0.05:
        break
toc = time.time()
print(f'Total training time: {toc - tic}')

# LBFGS优化
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=20)
epochi = 0
epochs = 5000
tic = time.time()
for epoch in range(epochi, epochs+epochi):
    lambda_data = 100.0  # LBFGS阶段使用固定权重
    loss = train(epoch, lambda_data)
    print(f'loss_tot:{loss:.8f}')
toc = time.time()
print(f'Total training time: {toc - tic}')

# ========== 可视化：对比预测值和实验数据 ==========
print('\n生成可视化结果...')

# 在实验数据点处预测
model.eval()
with torch.no_grad():
    y_pred_at_data = model(x_exp_tensor)
    rho_pred_at_data = y_pred_at_data[:, 0].cpu().numpy()
    p_pred_at_data = y_pred_at_data[:, 1].cpu().numpy()
    u_pred_at_data = y_pred_at_data[:, 2].cpu().numpy()

# 绘制对比图
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 在最终时刻的预测
x = np.linspace(Xs, Xe, 200)
t = np.full_like(x, Te)
x_test = np.hstack((t[:, None], x[:, None]))
x_test_tensor = torch.tensor(x_test, dtype=dtype).to(cuda)

with torch.no_grad():
    u_pred = model(x_test_tensor)
    rho_pred = u_pred[:, 0].cpu().numpy()
    p_pred = u_pred[:, 1].cpu().numpy()
    u_vel_pred = u_pred[:, 2].cpu().numpy()

# 第一行：最终时刻的预测
axes[0, 0].plot(x, rho_pred, 'b-', linewidth=2, label='PINN Prediction')
axes[0, 0].scatter(x_exp[x_exp[:, 0] == Te, 1], rho_exp[x_exp[:, 0] == Te], 
                   c='r', s=50, marker='o', label='Experimental Data', zorder=5)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title(f'Density at t={Te}')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(x, p_pred, 'r-', linewidth=2, label='PINN Prediction')
axes[0, 1].scatter(x_exp[x_exp[:, 0] == Te, 1], p_exp[x_exp[:, 0] == Te], 
                   c='r', s=50, marker='o', label='Experimental Data', zorder=5)
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('Pressure')
axes[0, 1].set_title(f'Pressure at t={Te}')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(x, u_vel_pred, 'g-', linewidth=2, label='PINN Prediction')
axes[0, 2].scatter(x_exp[x_exp[:, 0] == Te, 1], u_exp[x_exp[:, 0] == Te], 
                   c='r', s=50, marker='o', label='Experimental Data', zorder=5)
axes[0, 2].set_xlabel('x')
axes[0, 2].set_ylabel('Velocity')
axes[0, 2].set_title(f'Velocity at t={Te}')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 第二行：预测值 vs 实验数据的散点图
axes[1, 0].scatter(rho_exp, rho_pred_at_data, alpha=0.6)
axes[1, 0].plot([rho_exp.min(), rho_exp.max()], [rho_exp.min(), rho_exp.max()], 
                'r--', linewidth=2, label='Perfect Match')
axes[1, 0].set_xlabel('Experimental Density')
axes[1, 0].set_ylabel('Predicted Density')
axes[1, 0].set_title('Density: Prediction vs Experiment')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(p_exp, p_pred_at_data, alpha=0.6)
axes[1, 1].plot([p_exp.min(), p_exp.max()], [p_exp.min(), p_exp.max()], 
                'r--', linewidth=2, label='Perfect Match')
axes[1, 1].set_xlabel('Experimental Pressure')
axes[1, 1].set_ylabel('Predicted Pressure')
axes[1, 1].set_title('Pressure: Prediction vs Experiment')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].scatter(u_exp, u_pred_at_data, alpha=0.6)
axes[1, 2].plot([u_exp.min(), u_exp.max()], [u_exp.min(), u_exp.max()], 
                'r--', linewidth=2, label='Perfect Match')
axes[1, 2].set_xlabel('Experimental Velocity')
axes[1, 2].set_ylabel('Predicted Velocity')
axes[1, 2].set_title('Velocity: Prediction vs Experiment')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results_with_data_fusion.png', dpi=300, bbox_inches='tight')
print('结果已保存到: results_with_data_fusion.png')
plt.show()

# 计算数据拟合误差
mse_rho = ((rho_pred_at_data - rho_exp)**2).mean()
mse_p = ((p_pred_at_data - p_exp)**2).mean()
mse_u = ((u_pred_at_data - u_exp)**2).mean()

print(f'\n数据拟合误差:')
print(f'  密度 MSE: {mse_rho:.6f}')
print(f'  压力 MSE: {mse_p:.6f}')
print(f'  速度 MSE: {mse_u:.6f}')

print('\n完成！')

