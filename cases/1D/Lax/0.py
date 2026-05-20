# ========== 路径设置（必须放在最前面）==========
import sys
import os

# 自动检测并添加PINNsrc路径
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# 从 cases/1D/Lax 向上三级到 PINN_WE，然后进入 PINNsrc
pinn_src_path = os.path.join(current_file_dir, '..', '..', '..', 'PINNsrc')
pinn_src_path = os.path.abspath(pinn_src_path)
if pinn_src_path not in sys.path:
    sys.path.insert(0, pinn_src_path)

# ========== 导入模块 ==========
from utility import save_results, select_gpu, create_output_dir, visualize_training_progress, visualize_initial_condition

def train(epoch):
    """
    训练函数：定义损失函数和优化步骤
    
    当前配置（原始PINN测试）：
    - 损失函数包含2项：loss_pde + loss_ic（标准原始PINN配置）
    - 权重：loss_pde权重1，loss_ic权重10
    - 权重函数k=1（相当于权重函数影响较小，接近标准PINN）
    - 目的：测试原始PINN在激波问题上的效果
    """
    def closure():
        optimizer.zero_grad()                                                     
        loss_pde = model.loss_pde(x_int,x_screen2_L,x_screen2_R,0.01)
        loss_ic = model.loss_ic(x_ic, rho_ic,u_ic,p_ic)   # 初始条件约束
        
        # ========== 原始PINN配置：PDE残差 + 初始条件 ==========
        loss = loss_pde + 10*loss_ic
        
        # ========== 其他损失项（已注释，用于对比测试） ==========
        # loss_rh1 = model.loss_rh(x_screen2,x_screen2_L,x_screen2_R) # RH relation
        # loss_con3 = model.loss_con(x_screen3, x_con_points_tensor, T3) #Conservation laws
        # loss_con1 = model.loss_con(x_screen1, x_con_points_tensor, T1) #Conservation laws
        # loss = 1*loss_pde + 10*loss_ic + 10*loss_rh1 + 10*loss_con1  # 论文配置
        
        print(f'epoch {epoch} loss_pde:{loss_pde:.8f}, loss_ic:{loss_ic:.8f}')
        loss.backward()
        return loss
    loss = optimizer.step(closure)
    return loss

def Unit_var(rhoL,uL,pL,rhoR,uR,pR,t):
  rhoref = max(rhoL,rhoR)
  pref = max(pL,pR)
  uref = np.sqrt(pref/rhoref)
  
  uLn = uL/uref
  uRn = uR/uref
  pLn = pL/pref
  pRn = pR/pref
  rhoLn = rhoL/rhoref
  rhoRn = rhoR/rhoref
  
  tn = t*uref
  
  return rhoLn,uLn,pLn,rhoRn,uRn,pRn, tn,rhoref,uref,pref

import torch
import torch.nn as nn
import numpy as np
import time
import scipy.io
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS
dtype=torch.float32
dtype=torch.float64
# Seeds
crhoL = 0.445
cuL = 0.698
cpL = 3.528

crhoR = 0.5
cuR = 0
cpR = 0.571
Ts = 0
# ========== 时间参数配置 ==========
# 原始仓库配置：Te = 0.13（归一化前）
Te = 0.13

# ========== 论文配置（已注释） ==========
# 论文: Te = 1.4（归一化前）
# Te = 1.4  # 论文: 1.4 (归一化前)
# 注意：用户曾尝试过 Te = 0.14，但原始仓库是 0.13
crhoL,cuL,cpL,crhoR,cuR,cpR,Te,rhoref,uref,pref = Unit_var(crhoL,cuL,cpL,crhoR,cuR,cpR,Te)

###Ts = 0, Xs =0, Xe = 1

#crhoL = 0.89
#cuL = 0.098923
#cpL = 1
#
#crhoR = 1
#cuR = 0
#cpR = 0.16185
#
#Ts = 0
#Te = 0.91728

Xs = 0
Xe = 1
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(5)
   
def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,grad_outputs=torch.ones_like(outputs), create_graph=True)

# Convert torch tensor into np.array
def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or ' \
                        'np.ndarray, but got {}'.format(type(input)))

# Initial conditions
def IC(x):
    N = x.shape[0]
    rho_init = np.zeros((x.shape[0]))                                              
    u_init = np.zeros((x.shape[0]))                                                
    p_init = np.zeros((x.shape[0]))                                                

    # rho, p - initial condition
    for i in range(N):
        if (x[i,1] <= 0.5):
            rho_init[i] = crhoL
            u_init[i] = cuL
            p_init[i] = cpL
        else:
            rho_init[i] = crhoR
            u_init[i] = cuR
            p_init[i] = cpR

    return rho_init, u_init, p_init

def BC(x):
    N = x.shape[0]
    rho_init = np.zeros((x.shape[0]))                                              
    u_init = np.zeros((x.shape[0]))                                                
    p_init = np.zeros((x.shape[0]))                                                

    # rho, p - initial condition
    for i in range(N):
        if (x[i,1] <= 0.5):
            rho_init[i] = 1.0
            p_init[i] = 1.0
        else:
            rho_init[i] = 0.125
            p_init[i] = 0.1

    return rho_init, u_init, p_init


# Generate Neural Network
class DNN(nn.Module):

    def __init__(self):
        """
        神经网络结构定义
        
        原始仓库配置：
        - 总层数：6层（1输入层 + 5隐藏层 + 1输出层）
        - 每层神经元数：30个
        - 激活函数：Tanh
        
        论文配置（已注释）：
        - 总层数：9层（1输入层 + 7隐藏层 + 1输出层）
        - 每层神经元数：50个
        - 激活函数：Tanh
        """
        super(DNN, self).__init__()
        # ========== 原始仓库配置：6层，每层30个神经元 ==========
        self.net = nn.Sequential()                                                  
        self.net.add_module('Linear_layer_1', nn.Linear(2, 30))                     
        self.net.add_module('Tanh_layer_1', nn.Tanh())                              
        
        for num in range(2, 7):  # range(2,7) = [2,3,4,5,6] → 5层隐藏层
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(30, 30))       
            self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())                 
        self.net.add_module('Linear_layer_final', nn.Linear(30, 3))
        
        # ========== 论文配置（7层隐藏层，每层50个神经元，已注释） ==========
        # self.net = nn.Sequential()                                                  
        # self.net.add_module('Linear_layer_1', nn.Linear(2, 50))  # 论文: 50个神经元
        # self.net.add_module('Tanh_layer_1', nn.Tanh())                              
        # for num in range(2, 9):  # 论文: 7层隐藏层 (range(2,9) = 2,3,4,5,6,7,8)
        #     self.net.add_module('Linear_layer_%d' % (num), nn.Linear(50, 50))  # 论文: 50个神经元
        #     self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())                 
        # self.net.add_module('Linear_layer_final', nn.Linear(50, 3))  # 论文: 50个神经元
        # # 改进的权重初始化（Xavier uniform）
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)

    # Forward Feed
    def forward(self, x):
        return self.net(x)

    # Loss function for PDE
    def loss_pde(self, x,xL,xR,Dx):
        y = self.net(x)                                                
        rho,p,u = y[:, 0:1], y[:, 1:2], y[:, 2:]
        
        yR = self.net(xR)                                                
        rhoR,pR,uR = yR[:, 0:1], yR[:, 1:2], yR[:, 2:]
        yL = self.net(xL)                                                
        rhoL,pL,uL = yL[:, 0:1], yL[:, 1:2], yL[:, 2:]
        
        U2 = rho*u
        U3 = 0.5*rho*u**2 + p/0.4
        
        #F1 = U2
        F2 = rho*u**2+p
        F3 = u*(U3 + p)
        
        gamma = 1.4                                                    

        # Gradients and partial derivatives
        drho_g = gradients(rho, x)[0]                                  
        rho_t, rho_x = drho_g[:, :1], drho_g[:, 1:]             


        du_g = gradients(u, x)[0]                                      
        u_t, u_x = du_g[:, :1], du_g[:, 1:]                            
        
        dp_g = gradients(p, x)[0]                                      
        p_t, p_x = dp_g[:, :1], dp_g[:, 1:]                            

       # dp_g = gradients(p, x)[0]                                     
       # p_t, p_x = dp_g[:, :1], dp_g[:, 1:]                           
        
        dU2_g = gradients(U2, x)[0]
        U2_t,U2_x = dU2_g[:,:1], dU2_g[:,1:]
        dU3_g = gradients(U3, x)[0]
        U3_t,U3_x = dU3_g[:,:1], dU3_g[:,1:]
        dF2_g = gradients(F2, x)[0]
        F2_t,F2_x = dF2_g[:,:1], dF2_g[:,1:]
        dF3_g = gradients(F3, x)[0]
        F3_t,F3_x = dF3_g[:,:1], dF3_g[:,1:]

        # ========== 权重函数（WE参数）：用于PDE损失项的加权 ==========
        # 当前配置（论文配置）：在压缩区域降低权重
        # 论文参数: WE参数k1 = 0.2
        # 
        # 重要：损失函数中是除以d，即 loss = ((PDE_residual/d)**2).mean()
        # 所以如果 d = (k*(abs(u_x)-(u_x))+1)，那么实际权重是 1/d = 1/(k*(abs(u_x)-(u_x))+1)
        # 
        # 说明：
        # - 在压缩区域（u_x<0）时：abs(u_x)-(u_x) = 2*abs(u_x) > 0
        #   → d = (k*2*abs(u_x)+1) > 1
        #   → 1/d < 1，权重降低，有利于稳定训练
        # - 在膨胀区域（u_x>0）时：abs(u_x)-(u_x) = 0
        #   → d = 1
        #   → 1/d = 1，权重为1（正常）
        # 
        # ========== 原始PINN测试配置：k=1 ==========
        # k=1时，权重函数的影响较小，接近标准PINN（无权重函数时d=1）
        # 在压缩区域（u_x<0）时：d = (1*2*abs(u_x)+1) > 1，权重降低
        # 在膨胀区域（u_x>0）时：d = 1，权重正常
        # 注意：k=1时权重函数仍然有效，但影响比k=0.2时更大
        k = 0
        d1 = (k*(abs(u_x)-(u_x) )+1)
        d2 = (k*(abs(u_x)-(u_x) )+1)
        d3 = (k*(abs(u_x)-(u_x) )+1)
        
        # ========== 论文配置（已注释）：k=0.2 ==========
        # k = 0.2  # 论文配置：在压缩区域更大幅度降低权重
        
        # ========== 如果损失函数中是乘以d（已注释） ==========
        # 如果损失函数是 loss = ((PDE_residual*d)**2).mean()，那么应该用分数形式：
        # d1 = 1/(k*(abs(u_x)-(u_x) )+1)
        # d2 = 1/(k*(abs(u_x)-(u_x) )+1)
        # d3 = 1/(k*(abs(u_x)-(u_x) )+1)
        
        # ========== 原始仓库配置（线性形式，在压缩区域增加权重，已注释） ==========
        # 原始仓库配置：d = 0.4*(abs(u_x)-(u_x))+1
        # 说明：在压缩区域（u_x<0）时，abs(u_x)-(u_x) = 2*abs(u_x)，权重增加
        #       在膨胀区域（u_x>0）时，abs(u_x)-(u_x) = 0，权重为1
        #       这意味着在压缩区域（激波）增加权重，可能不利于稳定
        # d1 = 0.4*(abs(u_x)-(u_x) )+1 #+ (torch.sign(0.01 -abs(u_x))+1)*abs(rho_x))+ 1
        # d2 = 0.4*(abs(u_x)-(u_x) )+1 #+ (torch.sign(0.01 -abs(u_x))+1)*abs(rho_x))+ 1
        # d3 = 0.4*(abs(u_x)-(u_x) )+1 #+ (torch.sign(0.01 -abs(u_x))+1)*abs(rho_x))+ 1
        
        #d = 0.1*(abs(uR-uL)-(uR-uL))/Dx + 1
        #d = torch.exp(-10*u_x)+1
        #d1 = torch.clamp(d/5,min=1)
     
        f = (((rho_t + U2_x)/d1)**2).mean() + \
            (((U2_t  + F2_x)/d2)**2).mean() + \
            (((U3_t  + F3_x)/d3)**2).mean() #+\
            #((rho_t).mean())**2 +\
            #((U3_t).mean())**2 
    
        return f

    def loss_ic(self, x, rho, u, p):
        y = self.net(x)                                                      
        rho_nn, p_nn,u_nn = y[:, 0], y[:, 1], y[:, 2]            

        loss_ics = ((u_nn - u) ** 2).mean() + \
               ((rho_nn- rho) ** 2).mean()  + \
               ((p_nn - p) ** 2).mean()

        return loss_ics
    
    # Loss function for conservation
    def loss_con(self, x_en,x_in,t):
        y_en = self.net(x_en)                                       
        y_in = self.net(x_in)                                       
        rhoen, pen,uen = y_en[:, 0], y_en[:, 1], y_en[:, 2]         
        rhoin, pin,uin = y_in[:, 0], y_in[:, 1], y_in[:, 2]         

        U3en = 0.5*rhoen*uen**2 + pen/0.4
        U3in = 0.5*rhoin*uin**2 + pin/0.4
        gamma = 1.4
        cU3L = 0.5*crhoL*cuL**2 + cpL/0.4 
        cU3R = 0.5*crhoR*cuR**2 + cpR/0.4 
        # Loss function for the initial condition
        loss_en = ((rhoen - rhoin).mean() - t*(crhoL*cuL-crhoR*cuR))**2+ \
            ((-U3en+ U3in).mean() + t*(cU3L*cuL - cU3R*cuR) + (cpL*cuL - cpR*cuR)*t )**2 +\
            ((-rhoen*uen + rhoin*uin).mean()+(cpL-cpR)*t +(crhoL*cuL*cuL-crhoR*cuR*cuR)*t)**2
        return loss_en
    
    def loss_rh(self, x,x_l,x_r):
        """
        Rankine-Hugoniot关系损失函数：用于激波约束
        
        原始仓库配置：
        - eta过滤器：只在压力差和速度差都>0.1时才计算损失
        - 阈值：0.1（用于检测激波位置）
        - 说明：如果点不在激波附近，eta=0，loss_rh=0
        """
        y = self.net(x)                                    
        y_r = self.net(x_r)                                    
        y_l = self.net(x_l)                                    
        rho, p,u = y[:, 0], y[:, 1], y[:, 2]          
        rhol, pl,ul = y_l[:, 0], y_l[:, 1], y_l[:, 2]          
        rhor, pr,ur = y_r[:, 0], y_r[:, 1], y_r[:, 2]          

        du_g = gradients(u, x)[0]                                      
        u_t, u_x = du_g[:, :1], du_g[:, 1:]                            
        d = 1/(0.1*(abs(u_x)-u_x)  + 1)
        # ========== eta过滤器：只在激波附近（压力差和速度差都>0.1）才计算RH损失 ==========
        # 原始仓库配置：阈值0.1
        #eta =  torch.clamp(d-0.1,max=0)*torch.clamp(abs(p-pl)-0.01,min=0)*torch.clamp(abs(u-ul)-0.01,min=0)
        eta =  torch.clamp(abs(p-pl)-0.1,min=0)*torch.clamp(abs(u-ul)-0.1,min=0)
        #eta = 1  # 测试用：移除过滤器，看看loss_rh是否正常
        
        #loss_rh =  (((rho/rhol - (6*p+pl)/(6*pl+p))*eta)**2).mean()+\
        loss_rh = (((rhor/rhol - (6*pr+pl)/(6*pl+pr))*(ur-ul)*eta)**2).mean()+\
                   ((((ur-ul)**2 -2/rhor*(pr-pl)**2/(0.4*pr+2.4*pl))*eta)**2).mean()
           #        ((((ur-u)**2 -2/rho*(pr-p)**2/(0.4*pr+2.4*p))*eta)**2).max()
            
        #loss_rh =  (((pr/pl - (6*rhor-rhol)/(6*rhol-rhor))*(pr-pl)*eta)**2).max()+\
                   #((((u-ul)**2 -2/rho*(p-pl)**2/(0.4*p+2.4*pl))*eta)**2).max()+\
        return loss_rh
    
    def loss_character(self, x_l,x_r):
        y_r = self.net(x_r)                                                      # Initial condition
        y_l = self.net(x_l)                                                      # Initial condition
        rhol, pl,ul = y_l[:, 0], y_l[:, 1], y_l[:, 2]            # rho, u, p - initial condition
        rhor, pr,ur = y_r[:, 0], y_r[:, 1], y_r[:, 2]            # rho, u, p - initial condition

        #du_g = gradients(ul, x_l)[0]                                      
        #u_t, u_x = du_g[:, :1], du_g[:, 1:]                            
        #d = 1/(0.1*(abs(u_x)-u_x)  + 1)
        #eta =  torch.clamp(d-0.1,max=0)*torch.clamp(abs(pr-pl)-0.01,min=0)*torch.clamp(abs(ur-ul)-0.01,min=0)
        eta =  torch.clamp(abs(pr-pl)-0.01,min=0)*torch.clamp(abs(ur-ul)-0.01,min=0)
       # eta = 1
        # Loss function for the initial condition
        gamma = 1.4
        ss = 1.0e-10
        cL = torch.sqrt(gamma*abs(pl)/(abs(rhol)+ss))
        cR = torch.sqrt(gamma*abs(pr)/(abs(rhor)+ss))
        sR = torch.max(ul+cL,ur+cR)* (rhol-rhor)
        sL = torch.min(ul-cL,ur-cR)*(rhol-rhor)
        
        s = rhol*ul - rhor*ur
       # if (s.max() > 1000):
       #     print(rhol-rhor)
       #     print(s)
        #print(torch.clamp(s-sR,min=0))
       # print(eta)
       # sm = exp(-100*(s-sR))
        loss_s = (((s-sR)*(s-sL)*eta)**2).mean()  #torch.min((((,torch.tensor(1.0))  #+ ((torch.clamp(sL-s,min=0))**2).max()
        return loss_s
        
    
def X_entropy(x,T,dt,dx):
    N=x.shape[0]
    xs   = np.zeros((N,2)) 
    xsL  = np.zeros((N,2)) 
    xsR  = np.zeros((N,2)) 
    xsP  = np.zeros((N,2)) 
    xsPL = np.zeros((N,2)) 
    xsPR = np.zeros((N,2)) 
    
    for i in range(N):
        xs[i,1] = x[i,1]
        xs[i,0] = x[i,0] + T
        xsL[i,1] = xs[i,1] - dx
        xsL[i,0] = xs[i,0]
        xsR[i,1] = xs[i,1] + dx
        xsR[i,0] = xs[i,0]
        xsP[i,0] = xs[i,0] + dt
        xsP[i,1] = xs[i,1]
        xsPL[i,0] = xsP[i,0]
        xsPL[i,1] = xsP[i,1]+ dx
        xsPR[i,0] = xsP[i,0]
        xsPR[i,1] = xsP[i,1]- dx
        
    return xs,xsL,xsR,xsP,xsPL,xsPR


def X_right(x,dx):
    N=x.shape[0]
    xen =np.zeros((N,2)) 
    
    for i in range(N):
        xen[i,1] = x[i,1] + dx
        xen[i,0] = x[i,0] 
    return xen
def bc_data(N,Ts,Te,Xs,Xe):
    x =np.zeros((2*N,2)) 
    
    for i in range(N):
        x[i,0] = (Te - Ts)*i/N
        x[i,1] = Xs
        
    for i in range(N):
        x[i+N,0] = (Te - Ts)*i/N
        x[i+N,1] = Xe
    return x
def Mesh_Data(num_x,num_t,Tstart,Tend, Xstart,Xend):
    x_ic = np.zeros((num_x,2))
    x_int = np.zeros((num_x*(num_t-1),2))
    
    x_bc =np.zeros((2*(num_t-1),2)) 
    
    dt = (Tend - Tstart)/num_t
    x =   np.linspace(Xs, Xe, num_x) 
    x_ic[:,0] = 0
    x_ic[:,1] = x
    t = np.linspace(Tstart+dt, Tend, num_t-1)                                     
    x_bc[:num_t-1,0] = t
    x_bc[:num_t-1,1] = Xstart 
    x_bc[num_t-1:,0] = t
    x_bc[num_t-1:,1] = Xend

    
    t_grid, x_grid = np.meshgrid(t, x)                                 
    T = t_grid.flatten()[:, None]                                      
    X = x_grid.flatten()[:, None]                                      
    x_int = X[:, 0][:,None]                                        
    t_int = T[:, 0][:,None]                                        

    x_int = np.hstack((t_int, x_int))                            
    
    return x_ic,x_bc,x_int
    
    
# ========== GPU模式选择 ==========
# 优先 GPU_ID；默认自动选显存占用最低；AUTO_GPU=0 可关闭；USE_MULTI_GPU=1 启用多卡
gpu_id = select_gpu()
if gpu_id is not None:
    torch.cuda.set_device(gpu_id)
    device = torch.device('cuda')
    print(f'使用 GPU {gpu_id} (GPU_ID 手动指定，默认自动选最空闲；AUTO_GPU=0 关闭自动)')
else:
    device = torch.device('cpu')
    print('未检测到可用 GPU，使用 CPU')

# 通过环境变量 USE_MULTI_GPU=1 来启用多GPU模式，默认单GPU
use_multi_gpu = os.environ.get('USE_MULTI_GPU', '0') == '1'
num_gpus = torch.cuda.device_count()

# ========== 参数设置（论文参数） ==========
# ========== 采样点配置 ==========
# ========== 论文配置：LHS采样 + 随机选择 ==========
# 论文: 初始点1000个（LHS采样），PDE残差点50000个（从1000×5000网格中随机选取）
# RH和守恒约束点各100个，从x_ic中随机选择

# 1. 初始条件点 (x_ic): 1000个，LHS采样
xlimits = np.array([[0.,0],[0, Xe]])  # IC: t=0, x在[0, Xe]
sampling = LHS(xlimits=xlimits)
x_ic = sampling(1000)  # 论文: 1000个初始点（LHS）
rho_ic, u_ic, p_ic = IC(x_ic)

# 2. PDE残差点 (x_int): 50000个，从1000×5000均匀网格中随机选择
num_x_grid = 1000
num_t_grid = 5000
x_grid = np.linspace(Xs, Xe, num_x_grid)
t_grid = np.linspace(Ts, Te, num_t_grid)
t_mesh, x_mesh = np.meshgrid(t_grid, x_grid)
T_flat = t_mesh.flatten()[:, None]
X_flat = x_mesh.flatten()[:, None]
all_points = np.hstack((T_flat, X_flat))
np.random.seed(5)
id_f = np.random.choice(len(all_points), 50000, replace=False)
x_int = all_points[id_f]

# 3. RH和守恒约束点：各100个，从x_ic中随机选择
T1 = Te
T3 = Te
dx = 0.01  # 空间步长，用于生成左右邻居点
dt = 0.002  # 时间步长，用于生成时间推进点

np.random.seed(5)
idx_rh = np.random.choice(len(x_ic), 100, replace=False)
x_rh_points = x_ic[idx_rh]
x_screen2,x_screen2_L,x_screen2_R,x_screen2_P,x_screen2_PL,x_screen2_PR = X_entropy(x_rh_points,0.0,dt,dx)

x_con_points = x_rh_points  # 守恒约束点使用与RH相同的100个点
x_screen1,x_screen1_L,x_screen1_R,x_screen1_P,x_screen1_PL,x_screen1_PR = X_entropy(x_con_points,T1,dt,dx)
x_screen3,x_screen3_L,x_screen3_R,x_screen3_P,x_screen3_PL,x_screen3_PR = X_entropy(x_con_points,T3,dt,dx)
x_con_points_tensor = torch.tensor(x_con_points, requires_grad=True, dtype=dtype).to(device)

# ========== 原始仓库配置（已注释） ==========
# 原始仓库配置：使用Mesh_Data生成均匀网格点
# - Nx = 100, Nt = 100
# - x_ic: 100个初始条件点（t=0，均匀分布在[0,1]）
# - x_int: 9900个PDE残差点（100×99，均匀网格）
# - x_bc: 198个边界条件点（左右边界各99个）
# Nx = 100
# Nt = 100
# x_ic,x_bc,x_int =  Mesh_Data(Nx,Nt,Ts,Te,Xs,Xe)
# rho_ic, u_ic, p_ic= IC(x_ic)                    
# rho_bc, u_bc, p_bc= IC(x_bc)                    
# x_screen1,x_screen1_L,x_screen1_R,x_screen1_P,x_screen1_PL,x_screen1_PR = X_entropy(x_ic,T1,dt,dx)
# x_screen2,x_screen2_L,x_screen2_R,x_screen2_P,x_screen2_PL,x_screen2_PR = X_entropy(x_int,0.0,dt,dx)
# x_screen3,x_screen3_L,x_screen3_R,x_screen3_P,x_screen3_PL,x_screen3_PR = X_entropy(x_ic,T3,dt,dx)

x_screen1     = torch.tensor(x_screen1, requires_grad=True, dtype=dtype).to(device)
x_screen1_L   = torch.tensor(x_screen1_L, dtype=dtype).to(device) 
x_screen1_R   = torch.tensor(x_screen1_R, dtype=dtype).to(device)
x_screen1_P   = torch.tensor(x_screen1_P, requires_grad=True, dtype=dtype).to(device)
x_screen1_PL  = torch.tensor(x_screen1_PL, dtype=dtype).to(device)
x_screen1_PR  = torch.tensor(x_screen1_PR, dtype=dtype).to(device)

x_screen2     = torch.tensor(x_screen2, requires_grad=True, dtype=dtype).to(device)
x_screen2_L   = torch.tensor(x_screen2_L, dtype=dtype).to(device)
x_screen2_R   = torch.tensor(x_screen2_R, dtype=dtype).to(device)
x_screen2_P   = torch.tensor(x_screen2_P, requires_grad=True, dtype=dtype).to(device)
x_screen2_PL  = torch.tensor(x_screen2_PL, dtype=dtype).to(device)
x_screen2_PR  = torch.tensor(x_screen2_PR, dtype=dtype).to(device)

x_screen3     = torch.tensor(x_screen3, requires_grad=True, dtype=dtype).to(device)
x_screen3_L   = torch.tensor(x_screen3_L, dtype=dtype).to(device) 
x_screen3_R   = torch.tensor(x_screen3_R, dtype=dtype).to(device)
x_screen3_P   = torch.tensor(x_screen3_P, requires_grad=True, dtype=dtype).to(device)
x_screen3_PL  = torch.tensor(x_screen3_PL, dtype=dtype).to(device)
x_screen3_PR  = torch.tensor(x_screen3_PR, dtype=dtype).to(device)

# 保存numpy版本的初始条件用于可视化（在转换为tensor之前）
x_ic_np = x_ic.copy()  # 保存numpy版本
rho_ic_np = rho_ic.copy()
u_ic_np = u_ic.copy()
p_ic_np = p_ic.copy()

x_ic = torch.tensor(x_ic,requires_grad=True, dtype=dtype).to(device)
x_int = torch.tensor(x_int, requires_grad=True, dtype=dtype).to(device)

# ========== 原始仓库配置（已注释） ==========
# 论文配置不需要边界条件点
# x_bc = torch.tensor(x_bc,requires_grad=True, dtype=dtype).to(device)

rho_ic = torch.tensor(rho_ic, dtype=dtype).to(device)
u_ic = torch.tensor(u_ic, dtype=dtype).to(device)
p_ic = torch.tensor(p_ic, dtype=dtype).to(device)

# ========== 原始仓库配置（已注释） ==========
# 论文配置不需要边界条件点
# rho_bc = torch.tensor(rho_bc, dtype=dtype).to(device)
# u_bc = torch.tensor(u_bc, dtype=dtype).to(device)
# p_bc = torch.tensor(p_bc, dtype=dtype).to(device)


model = DNN().to(device).double()

print('Start training...')

# ========== 训练策略配置 ==========
# 当前配置：两阶段训练（Adam预热 + LBFGS精调）
# 第一阶段：Adam优化器
#   - 学习率：lr = 0.0005
#   - 训练轮数：100000 epochs（带早停，loss<0.05）
#   - 作用：初步搜索，找到大致的最优区域
# 第二阶段：LBFGS优化器
#   - 学习率：lr = 0.1
#   - max_iter = 20
#   - 训练轮数：5000 epochs
#   - 作用：精细优化，快速收敛到最优解

epoch = 0
epochi = epoch
loss_history=[]
interrupted = False
tic_total = time.time()

# ========== 提前创建输出目录（用于保存训练过程可视化） ==========
# 这样训练过程中的检查点图片可以保存在实验输出目录内
# 获取当前脚本名（去掉.py后缀）作为额外层级
script_name = os.path.splitext(os.path.basename(__file__))[0]  # 获取脚本名（如 '1' 或 '0'）
base_output_dir = os.path.join('output', script_name)  # output/1/ 或 output/0/

gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
output_dir = create_output_dir('Lax', base_dir=base_output_dir, gpu_count=gpu_count)
training_vis_dir = os.path.join(output_dir, 'training_progress')
os.makedirs(training_vis_dir, exist_ok=True)
print(f'\n[训练过程可视化] 输出目录: {output_dir}')
print(f'[训练过程可视化] 检查点保存目录: {training_vis_dir}')

# ========== 可视化初始状态（第0个epoch） ==========
print('\n========== 可视化初始状态（第0个epoch） ==========')
# 对于lax，初始条件是阶跃函数（Riemann问题），没有真解+扰动模式
# 所以只显示初始条件本身
visualize_initial_condition(
    x_ic=x_ic_np, rho_ic=rho_ic_np, u_ic=u_ic_np, p_ic=p_ic_np,
    save_dir=training_vis_dir, Xs=Xs, Xe=Xe,
    exact_dict=None,  # lax没有真解对比
    title_prefix="Initial Condition (Riemann Problem)"
)

# ========== 第一阶段：Adam (lr=0.0005, 100000 epochs，早停loss<0.05) ==========
print('========== 第一阶段：Adam (lr=0.0005, 100000 epochs，早停0.05) ==========')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
epochs_stage1 = 100000
checkpoint_interval_stage1 = 1000  # Adam阶段每1000个epoch保存一次
try:
    tic_stage1 = time.time()
    for epoch in range(1+epochi, epochs_stage1+epochi):
        loss = train(epoch)
        print(f'loss_tot:{loss:.8f}')
        loss_history.append(to_numpy(loss))
        
        # ========== 训练过程可视化检查点（每1000个epoch） ==========
        if epoch % checkpoint_interval_stage1 == 0:
            visualize_training_progress(
                model=model, epoch=epoch, stage='Adam', save_dir=training_vis_dir,
                Xs=Xs, Xe=Xe, Te=Te, exact_file='laxe.dat',
                rhoref=rhoref, uref=uref, pref=pref,
                device=device, dtype=dtype
            )
        
        if loss < 0.15:  # 早停条件：loss < 0.05
            print(f'达到早停条件 (loss < 0.05)，第一阶段训练结束')
            # 保存最后一次检查点
            if training_vis_dir is not None:
                visualize_training_progress(
                    model=model, epoch=epoch, stage='Adam', save_dir=training_vis_dir,
                    Xs=Xs, Xe=Xe, Te=Te, exact_file='laxe.dat',
                    rhoref=rhoref, uref=uref, pref=pref,
                    device=device, dtype=dtype
                )
            break
    toc_stage1 = time.time()
    print(f'第一阶段训练时间: {toc_stage1 - tic_stage1:.2f}秒')
    epochi = 0  # LBFGS 从0开始计数
    stage_boundary = len(loss_history)
except KeyboardInterrupt:
    interrupted = True
    print("\n检测到中断 (Ctrl+C)，将保存当前模型与结果...")
    stage_boundary = len(loss_history)

# ========== 第二阶段：LBFGS (lr=0.1, max_iter=20, 5000 epochs) ==========
if not interrupted:
    print('========== 第二阶段：LBFGS (lr=0.1, max_iter=20, 5000 epochs) ==========')
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=20)
    epochs_stage2 = 5000
    checkpoint_interval_stage2 = 100  # LBFGS阶段每100个epoch保存一次
    try:
        tic_stage2 = time.time()
        for epoch in range(epochi, epochs_stage2+epochi):
            loss = train(epoch)
            print(f'loss_tot:{loss:.8f}')
            loss_history.append(to_numpy(loss))
            
            # ========== 训练过程可视化检查点（每100个epoch） ==========
            if epoch % checkpoint_interval_stage2 == 0:
                visualize_training_progress(
                    model=model, epoch=epoch, stage='LBFGS', save_dir=training_vis_dir,
                    Xs=Xs, Xe=Xe, Te=Te, exact_file='laxe.dat',
                    rhoref=rhoref, uref=uref, pref=pref,
                    device=device, dtype=dtype
                )
        toc_stage2 = time.time()
        print(f'第二阶段训练时间: {toc_stage2 - tic_stage2:.2f}秒')
        # 保存最后一次检查点
        if training_vis_dir is not None:
            visualize_training_progress(
                model=model, epoch=epoch, stage='LBFGS', save_dir=training_vis_dir,
                Xs=Xs, Xe=Xe, Te=Te, exact_file='laxe.dat',
                rhoref=rhoref, uref=uref, pref=pref,
                device=device, dtype=dtype
            )
    except KeyboardInterrupt:
        interrupted = True
        print("\n检测到中断 (Ctrl+C)，将保存当前模型与结果...")

if 'stage_boundary' not in locals():
    stage_boundary = len(loss_history)

# ========== 计算总训练时间 ==========
total_training_time = time.time() - tic_total
print(f'Total training时间: {total_training_time:.2f}秒')

# ========== 原始仓库配置（单阶段训练，已注释） ==========
# 原始仓库配置：单阶段Adam优化器
# - 学习率：lr = 0.001
# - 优化器：Adam
# - 训练轮数：2000 epochs
# - 特点：简单直接，能快速得到一个基本形状
# 
# lr = 0.001
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# epoch = 0
# epochi = epoch
# epochs = 2000
# loss_history=[]
# tic = time.time()
# for epoch in range(1+epochi, epochs+epochi):
#     loss = train(epoch)
#     print(f'loss_tot:{loss:.8f}')
#     loss_history.append(to_numpy(loss))
# toc = time.time()
# print(f'Total training time: {toc - tic}')
# total_training_time = toc - tic

# ========== 可视化结果（带精确解对比）==========
print('\n开始生成可视化结果...')

# 创建测试网格（在最终时刻 t=Te）
x = np.linspace(Xs, Xe, 200)
t = np.full_like(x, Te)
x_test = np.hstack((t[:, None], x[:, None]))
x_test_tensor = torch.tensor(x_test, dtype=dtype).to(device)

# 预测（网络输出的是归一化值）
actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
actual_model.eval()
with torch.no_grad():
    u_pred = actual_model(x_test_tensor)
    rho_pred_norm = u_pred[:, 0].cpu().numpy()  # 归一化密度
    p_pred_norm = u_pred[:, 1].cpu().numpy()    # 归一化压力
    u_vel_pred_norm = u_pred[:, 2].cpu().numpy()  # 归一化速度

# ========== 量纲还原：将归一化值还原到物理量纲 ==========
# 注意：rhoref, uref, pref 在全局作用域中定义（由 Unit_var 函数返回）
rho_pred = rho_pred_norm * rhoref  # 还原密度
p_pred = p_pred_norm * pref         # 还原压力
u_vel_pred = u_vel_pred_norm * uref  # 还原速度

print("预测完成（已还原到物理量纲）")

# 尝试加载精确解进行对比
exact_file = 'laxe.dat'
x_exact = None
if os.path.exists(exact_file):
    print(f"找到精确解文件: {exact_file}")
    exact_data = np.loadtxt(exact_file)
    x_exact_raw = exact_data[:, 0]
    rho_exact_raw = exact_data[:, 1]
    u_exact_raw = exact_data[:, 2]
    p_exact_raw = exact_data[:, 3]
    
    # 裁剪精确解到 [0, 1] 范围（与测试点范围一致）
    # 注意：精确解文件的 x 范围可能超出 [0, 1]，需要裁剪
    mask = (x_exact_raw >= Xs) & (x_exact_raw <= Xe)
    x_exact = x_exact_raw[mask]
    rho_exact = rho_exact_raw[mask]
    u_exact = u_exact_raw[mask]
    p_exact = p_exact_raw[mask]
    
    print(f"精确解原始范围: x=[{x_exact_raw.min():.4f}, {x_exact_raw.max():.4f}]")
    print(f"精确解裁剪后: x=[{x_exact.min():.4f}, {x_exact.max():.4f}], 点数={len(x_exact)}")
    
    # 插值到相同网格（用于误差计算）
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
# 显示物理时间（还原后的时间）
t_physical = Te / uref if uref > 0 else Te
ax1.set_title(f'Density (t={t_physical:.4f})', fontsize=14, fontweight='bold')
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
ax2.set_title(f'Pressure (t={t_physical:.4f})', fontsize=14, fontweight='bold')
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
ax3.set_title(f'Velocity (t={t_physical:.4f})', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(fontsize=11, loc='best')

plt.tight_layout()

# ========== 使用通用结果保存系统 ==========
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
    'case': 'Lax',
    'Ts': Ts,
    'Te': Te,
    'Xs': Xs,
    'Xe': Xe,
    # ========== 论文配置 ==========
    # 论文配置不使用Mesh_Data，所以Nx和Nt不再使用
    # 'Nx': Nx,  # 原始仓库: 100
    # 'Nt': Nt,  # 原始仓库: 100
    'crhoL': crhoL,
    'cuL': cuL,
    'cpL': cpL,
    'crhoR': crhoR,
    'cuR': cuR,
    'cpR': cpR,
    'model_layers': 7,  # 论文: 7层隐藏层
    'model_neurons': 50,  # 论文: 每层50个神经元
    'sampling_method': 'LHS_Random',  # 论文: LHS采样 + 1000×5000网格随机选
    'ic_points': 1000,  # 论文: 1000个初始点（LHS）
    'pde_points': 50000,  # 论文: 50000个PDE残差点
    'rh_points': 100,  # 论文: 100个RH约束点（已生成但未使用）
    'con_points': 100,  # 论文: 100个守恒约束点（已生成但未使用）
    'loss_config': 'PINN_IC',  # 原始PINN测试：loss_pde + loss_ic
    'k_parameter': 1.0  # 权重函数参数：k=1（原始PINN测试）
}

# 调用通用保存函数
# ========== 创建训练过程可视化目录 ==========
# 在保存结果之前，先创建训练过程可视化目录
# 注意：这里需要先调用save_results来获取output_dir，但我们需要提前创建目录
# 所以我们在训练循环中延迟创建，或者在这里先创建一个临时目录
# 实际上，我们可以在第一次检查点时创建目录

output_dir, metrics = save_results(
    case_name='Lax',
    output_dir=output_dir,  # 传入已创建的输出目录
    model=model,
    pred_dict=pred_dict,
    loss_history=loss_history,
    exact_dict=exact_dict,
    training_time=total_training_time,
    config=config,
    save_model=True,
    save_plot=True,
    gpu_count=num_gpus,
    stage_boundary=stage_boundary
)

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

# ========== 训练完成提示 ==========
# 当前配置：两阶段训练，需要检查interrupted变量
if interrupted:
    print("\n训练被中断，已保存当前结果。")
else:
    print(f"\n训练完成，结果已保存到: {output_dir}")

# ========== 原始仓库配置（单阶段训练，已注释） ==========
# 原始仓库配置：单阶段训练，不需要interrupted变量
# print(f"\n训练完成，结果已保存到: {output_dir}")