import torch
import torch.nn as nn
import numpy as np
import time
import scipy.io
import math
import Exact_burgers
import sys
import os

# 添加PINNsrc路径以便导入riemann_solver
current_file_dir = os.path.dirname(os.path.abspath(__file__))
pinn_src_path = os.path.join(current_file_dir, '..', '..', '..', 'PINNsrc')
pinn_src_path = os.path.abspath(pinn_src_path)
if pinn_src_path not in sys.path:
    sys.path.insert(0, pinn_src_path)

from riemann_solver import BurgersSolver

def test_riemann_solver():
    """
    测试黎曼求解器的正确性和性能
    """
    print('\n=== Test 1: Correctness Check ===')
    torch.manual_seed(42)
    N = 1000

    # 生成随机测试数据 (包含正负值，模拟激波和稀疏波)
    ul = torch.randn(N, device='cuda')
    ur = torch.randn(N, device='cuda')

    # 分别运行
    flux_torch = BurgersSolver.godunov_flux(ul, ur, backend='torch')
    flux_triton = BurgersSolver.godunov_flux(ul, ur, backend='triton')

    # 比较差异
    if torch.allclose(flux_torch, flux_triton, atol=1e-6):
        print("✅ Pass! Triton result matches PyTorch result.")
    else:
        print("❌ Fail! Results diverge.")
        diff = (flux_torch - flux_triton).abs().max()
        print(f"Max difference: {diff.item()}")

    print('\n=== Test 2: Speed Benchmark ===')
    # 数据量大一点，才能看出差距 (例如 1000万 个网格点)
    N = 10 * 1024 * 1024
    ul = torch.randn(N, device='cuda')
    ur = torch.randn(N, device='cuda')

    # 预热 (Warmup) - 防止第一次启动编译时间干扰
    for _ in range(10):
        BurgersSolver.godunov_flux(ul, ur, backend='triton')
        BurgersSolver.godunov_flux(ul, ur, backend='torch')

    # 1. 测 PyTorch 时间
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        BurgersSolver.godunov_flux(ul, ur, backend='torch')
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / 100
    print(f"PyTorch Average Time: {torch_time*1000:.3f} ms")

    # 2. 测 Triton 时间
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        BurgersSolver.godunov_flux(ul, ur, backend='triton')
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / 100
    print(f"Triton  Average Time: {triton_time*1000:.3f} ms")

    print(f"🚀 Speedup: {torch_time / triton_time:.2f}x")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
  #  random.seed(seed)
    torch.backends.cudnn.deterministic = True
WENO = np.loadtxt('result50WENO.dat')

num_seed = 1  # 快速测试：只运行1个种子

f= open("result_riemann.dat", "w+")
for nnode in [26]:  # 快速测试：只运行26个节点
  f.write('###########nnode: %d ########### \n' % nnode)

  L2_smooth_ave = []
  L2_shock_ave = []
  L2_ave = []
  L2_max_ave = []
  loss_ave = []
  godunov_time_ave = []  # 记录Godunov损失计算时间
  starttime = time.time()
  for i in range(num_seed):
    seed = i
    setup_seed(seed)
    f.write('****seed: %d ***** \n' % seed)

    # Godunov损失的权重
    lambda_godunov = 10.0

    # ========== 测试黎曼求解器（只测试一次）==========
    if i == 0:
        print('\n========== 测试黎曼求解器 ==========')
        test_riemann_solver()

    def train(epoch):
        model.it = epoch
        def closure():
            optimizer.zero_grad()
            loss_pde = model.loss_pde(x_int)
            loss_ic = model.loss_ic(x_ic, u_ic)

            # 计时：Godunov损失计算时间
            torch.cuda.synchronize()
            start_godunov = time.time()
            loss_godunov = model.loss_godunov(x_int, x_sorted_np, x_int_np)
            torch.cuda.synchronize()
            godunov_time = time.time() - start_godunov

            loss = loss_pde + 10*loss_ic + lambda_godunov*loss_godunov

            print(f'epoch {epoch} loss_pde:{loss_pde:.8f}, loss_ic:{loss_ic:.8f}, loss_godunov:{loss_godunov:.8f}, loss:{loss:.8f}, godunov_time:{godunov_time*1000:.3f}ms')
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        loss_value = loss.item()
        print(f'epoch {epoch}: loss {loss_value:.6f}')
        return loss

    def gradients(outputs, inputs):
        return torch.autograd.grad(outputs, inputs,grad_outputs=torch.ones_like(outputs), create_graph=True)

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
            y = self.net(x)
            u = y[:, 0:1]

            U = u**2/2

            dU_g = gradients(U, x)[0]
            U_x = dU_g[:, 1:]
            du_g = gradients(u, x)[0]
            u_t,u_x = du_g[:, :1],du_g[:,1:]
            d = 0.1*(abs(u_x)-u_x) + 1
            #d = 1

            f = (((u_t + U_x)/d)**2).mean()

            return f

        def loss_godunov(self, x_int, x_sorted_np, x_int_np):
            """
            Godunov黎曼求解器损失
            """
            # 获取网络预测的u值
            y = self.net(x_int)
            u_pred = y[:, 0:1]

            # 将x_int转为numpy用于排序
            x_int_np_local = to_numpy(x_int)

            # 按空间坐标排序
            sort_idx = np.argsort(x_int_np_local[:, 1])
            x_sorted = x_int_np_local[sort_idx]
            u_sorted = to_numpy(u_pred)[sort_idx]

            # 构造相邻点对（左状态和右状态）
            # 对于每个点，取它在空间上相邻的点
            n_points = len(x_sorted)
            if n_points < 2:
                return torch.tensor(0.0, device=x_int.device)

            # 创建左状态和右状态数组
            # 方式：对于每个位置i，左状态是u[i]，右状态是u[i+1]
            u_left_np = u_sorted[:-1].flatten()
            u_right_np = u_sorted[1:].flatten()

            # 转换为torch tensor
            u_left = torch.tensor(u_left_np, dtype=torch.float32).to(x_int.device)
            u_right = torch.tensor(u_right_np, dtype=torch.float32).to(x_int.device)

            # 使用Godunov求解器计算数值通量
            try:
                flux_godunov = BurgersSolver.godunov_flux(u_left, u_right, backend='triton')
            except:
                # 如果triton失败，回退到torch实现
                flux_godunov = BurgersSolver.godunov_flux(u_left, u_right, backend='torch')

            # 计算网络预测的通量 f(u) = 0.5 * u^2
            # 注意：这里需要对齐点，因为flux_godunov是相邻点之间的通量
            flux_pred_left = 0.5 * u_left**2
            flux_pred_right = 0.5 * u_right**2

            # 计算损失：使用左右通量的平均
            loss = ((flux_godunov - 0.5*(flux_pred_left + flux_pred_right))**2).mean()

            return loss

        def res_pde(self,x):
            y = self.net(x)
            Res = np.zeros((x.shape[0]))

            u = y[:, 0:1]
            U = u**2/2
            dU_g = gradients(U, x)[0]
            U_x = dU_g[:, 1:]
            du_g = gradients(u, x)[0]
            u_t,u_x = du_g[:, :1],du_g[:,1:]
            Res = (u_t + U_x)**2
            return Res

        def lambda_pde(self,x):
            y = self.net(x)
            Res = np.zeros((x.shape[0]))

            u = y[:, 0:1]
            du_g = gradients(u, x)[0]
            u_t,u_x = du_g[:, :1],du_g[:,1:]
            d = 0.1*(abs(u_x)-u_x) + 1
            return  d

        def loss_ic(self, x_ic, u_ic):
            y_ic = self.net(x_ic)
            u_ic_nn = y_ic[:, 0]
            loss_ics = ((u_ic_nn - u_ic) ** 2).mean()
            return loss_ics

    device = torch.device('cuda')
    #device = torch.device('cpu')
    num_x = nnode
    num_t =  nnode
    num_i_train = nnode
    num_f_train =  nnode*nnode
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

    model = DNN().to(device)

    #lr = 0.001

    optimizer = torch.optim.LBFGS(model.parameters(), lr=1,
                                  max_iter = 20,
                                  max_eval = None,
                                  tolerance_grad = 1e-05,
                                  tolerance_change = 1e-09,
                                  history_size = 100,
                                  line_search_fn = 'strong_wolfe')
    epochs = 10  # 快速测试：只运行10个epoch
    tic = time.time()

    loss_test = 100.0
    num_epoch = 0
    for epoch in range(1, epochs+1):
      loss=train(epoch)


    x = np.linspace(0.0, 2.0, nnode)
    t = np.linspace(1.0, 1.0, 1)
    t_grid, x_grid = np.meshgrid(t, x)
    T = t_grid.flatten()[:, None]
    X = x_grid.flatten()[:, None]
    x_test = np.hstack((T, X))
    x_test = torch.tensor(x_test, requires_grad=True, dtype=torch.float32).to(device)
    u_pred = to_numpy(model(x_test))
    res = to_numpy(model.res_pde(x_test))
    d   = to_numpy(model.lambda_pde(x_test))


    y_e = Exact_burgers.Exact_Burgers(x)
    L2_error = np.sqrt(np.sum((u_pred[:,0]-y_e)**2)/nnode)
    f.write('L2 error: %e\n' % L2_error)
    num = 0
    num_shock = 0
    sum = 0.0
    sum_shock = 0
    for i in range(nnode):
      if abs(x[i]-1)> 0.05:
        sum = sum +  np.sum((u_pred[i,0]-y_e[i])**2)
        num = num + 1
      else:
        sum_shock = sum_shock +  np.sum((u_pred[i,0]-y_e[i])**2)
        num_shock = num_shock + 1
    sum = sum/num
    sum_shock = sum_shock/num_shock
    L2_error_smooth = np.sqrt(sum)
    L2_error_shock = np.sqrt(sum_shock)
    L2_Max =  (np.max(np.abs(u_pred[:,0]-y_e)))
    L2_max_ave.append(L2_Max)
    L2_ave.append(L2_error)
    L2_smooth_ave.append(L2_error_smooth)
    L2_shock_ave.append(L2_error_shock)
    loss_ave.append(to_numpy(loss))

  f.write('L2_max_ave: %e, L2_max_top : %e, L2_max_bottom %e\n' %(np.mean(L2_max_ave),np.max(L2_max_ave),np.min(L2_max_ave)))
  f.write('L2_ave: %e, L2_ave_top: %e, L2_ave_bottom: %e \n' %(np.mean(L2_ave), np.max(L2_ave), np.min(L2_ave)))
  f.write('L2_smooth_ave: %e, L2_smooth_ave_top: %e, L2_smooth_bottom: %e \n' % (np.mean(L2_smooth_ave), np.max(L2_smooth_ave), np.min(L2_smooth_ave)))
  f.write('L2_shock_ave: %e, L2_shock_top: %e, L2_shock_bottom: %e \n' % (np.mean(L2_shock_ave),np.max(L2_shock_ave),np.min(L2_shock_ave)))
  f.write('loss_ave: %e , loss_top: %e, loss_bottom: %e \n' % (np.mean(loss_ave),np.max(loss_ave),np.min(loss_ave)))
  endtime = time.time()
  train_time = (endtime - starttime)/num_seed
  f.write('\n train_time: %e\n' % (train_time/num_seed))
f.close()
