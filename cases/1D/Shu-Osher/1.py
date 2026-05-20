import sys
import os

# ========== 路径设置（必须放在最前面）==========
current_file_dir = os.path.dirname(os.path.abspath(__file__))
pinn_src_path = os.path.abspath(os.path.join(current_file_dir, '..', '..', '..', 'PINNsrc'))
if pinn_src_path not in sys.path:
    sys.path.insert(0, pinn_src_path)

from utility import save_results, create_output_dir, evaluate_predictions, select_gpu  # 统一结果保存


def train(epoch):
    # DataParallel 兼容：获取实际模型
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    def closure():
        optimizer.zero_grad()
        loss_pde = actual_model.loss_pde(x_int, x_screen2_L, x_screen2_R, 0.01)
        loss_ic = actual_model.loss_ic(x_ic, rho_ic, u_ic, p_ic)
        loss_bc = actual_model.loss_ic(x_bc, rho_bc, u_bc, p_bc)

        loss_rh1 = actual_model.loss_rh(x_screen2, x_screen2_L, x_screen2_R)  # RH relation
        loss_s0 = actual_model.loss_character(x_screen2, x_screen2_R)  # Entropy condition
        loss_con3 = actual_model.loss_con(x_screen3, x_ic, T3)  # Conservation laws
        loss_con1 = actual_model.loss_con(x_screen1, x_ic, T1)  # Conservation laws

        # 保持原有权重（与旧脚本一致）
        loss = loss_pde + 100 * (loss_ic + loss_bc) + 10 * (loss_rh1 + loss_con1 + loss_con3) + 10 * loss_s0
        print(f'epoch {epoch} loss_pde:{loss_pde:.8f}, loss_ic:{loss_ic:.8f}, loss_bc:{loss_bc:.8f}, '
              f'loss_rh1:{loss_rh1:.8f}, loss_con1:{loss_con1:.8f}, loss_con3:{loss_con3:.8f}, loss_s:{loss_s0:.8f}')
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    return loss


def Unit_var(rhoL,uL,pL,rhoR,uR,pR,t):
  rhoref = max(rhoL,rhoR)
  pmax = max(pL,pR)
  umax = max(abs(uL),abs(uR))
  uref = max(np.sqrt(pmax/rhoref),umax)
  pref = uref**2*rhoref

  
  uLn = (uL)/uref
  uRn = (uR)/uref
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
dtype=torch.float32
dtype=torch.float64
# Seeds
#crhoL = 1
#cuL = -2
#cpL = 0.4
#
#crhoR =1
#cuR = 2
#cpR = 0.4
#Ts = 0
#Te = 0.1
rhoref = 1
uref = 1
pref = 1
crhoL = 27/7
cuL = 2.629369
cpL = 31/3

crhoR = 1
cuR = 0
cpR = 1

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

Ts = 0
Te = 0.18
Xs = 0
Xe = 1

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
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)



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
        if (x[i,1] <= 0.2):
            rho_init[i] = crhoL
            u_init[i] = cuL
            p_init[i] = cpL
        else:
            rho_init[i] = crhoR*(1+0.5*np.sin(15*x[i,1]))
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
        if (x[i,1] <= 0.2):
            rho_init[i] = crhoL
            u_init[i] = cuL
            p_init[i] = cpL
        else:
            rho_init[i] = crhoR*(1+0.2*np.sin(50))
            u_init[i] = cuR
            p_init[i] = cpR

    return rho_init, u_init, p_init


# Generate Neural Network
class DNN(nn.Module):

    def __init__(self):
        super(DNN, self).__init__()
        self.net = nn.Sequential()                                                  
        self.net.add_module('Linear_layer_1', nn.Linear(2, 30))                     
        #self.net.add_module('Tanh_layer_1', nn.Tanh())                              

        for num in range(2, 7):                                                     
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(30, 30))       
            self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())                 
        self.net.add_module('Linear_layer_final', nn.Linear(30, 3))                 

    # Forward Feed
    def forward(self, x):
        return self.net(x)

    # Loss function for PDE
    def loss_pde(self, x, xL, xR, Dx):
        # ========== 原公式（保留注释） ==========
        # y = self.net(x)
        # rho,p,u = y[:, 0:1], y[:, 1:2], y[:, 2:]
        # yR = self.net(xR)
        # rhoR,pR,uR = yR[:, 0:1], yR[:, 1:2], yR[:, 2:]
        # yL = self.net(xL)
        # rhoL,pL,uL = yL[:, 0:1], yL[:, 1:2], yL[:, 2:]
        # U2 = rho*u
        # U3 = 0.5*rho*u**2 + p/0.4
        # F2 = rho*u**2+p
        # F3 = u*(U3 + p)
        # d1 = 0.4*(abs(u_x)-(u_x) )+1
        # ...
        # f = (((rho_t + U2_x)/d1)**2 + ...).mean()

        # ========== 守恒律形式（与 Blast 对齐） ==========
        y = self.net(x)
        rho, p, u = y[:, 0:1], y[:, 1:2], y[:, 2:]

        gamma = 1.4
        E = 0.5 * u ** 2 + p / ((gamma - 1) * rho + 1e-8)  # 比总能
        U1 = rho
        U2 = rho * u
        U3 = rho * E
        F1 = U2
        F2 = rho * u ** 2 + p
        F3 = (U3 + p) * u

        # 梯度
        dU1_g = gradients(U1, x)[0]; U1_t, U1_x = dU1_g[:, :1], dU1_g[:, 1:]
        dU2_g = gradients(U2, x)[0]; U2_t, U2_x = dU2_g[:, :1], dU2_g[:, 1:]
        dU3_g = gradients(U3, x)[0]; U3_t, U3_x = dU3_g[:, :1], dU3_g[:, 1:]
        dF1_g = gradients(F1, x)[0]; F1_t, F1_x = dF1_g[:, :1], dF1_g[:, 1:]
        dF2_g = gradients(F2, x)[0]; F2_t, F2_x = dF2_g[:, :1], dF2_g[:, 1:]
        dF3_g = gradients(F3, x)[0]; F3_t, F3_x = dF3_g[:, :1], dF3_g[:, 1:]

        # 权重函数（与 Blast 相同形式）
        u_g = gradients(u, x)[0]; u_x = u_g[:, 1:]
        d = 1.0 / (0.2 * (torch.abs(u_x) - u_x) + 1.0)

        f = ((d * (U1_t + F1_x)) ** 2).mean() + \
            ((d * (U2_t + F2_x)) ** 2).mean() + \
            ((d * (U3_t + F3_x)) ** 2).mean()

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
        y = self.net(x)                                    
        y_r = self.net(x_r)                                    
        y_l = self.net(x_l)                                    
        rho, p,u = y[:, 0], y[:, 1], y[:, 2]          
        rhol, pl,ul = y_l[:, 0], y_l[:, 1], y_l[:, 2]          
        rhor, pr,ur = y_r[:, 0], y_r[:, 1], y_r[:, 2]          

        du_g = gradients(u, x)[0]                                      
        u_t, u_x = du_g[:, 0], du_g[:, 1]                            
        d = 1/(0.1*(abs(u_x)-u_x)  + 1)
        eta =  torch.clamp(abs(pr-pl)-0.1,min=0)*torch.clamp(abs(ur-ul)-0.1,min=0)
       # eta =  torch.clamp(abs(pr-pl)-0.1,min=0)*torch.clamp(abs(ur-ul)-0.1,min=0)
        #eta = 1
        
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
    
    x_bc =np.zeros(((num_t-1),2)) 
    
    dt = (Tend - Tstart)/num_t
    x =   np.linspace(Xs, Xe, num_x) 
    x_ic[:,0] = 0
    x_ic[:,1] = x
    t = np.linspace(Tstart+dt, Tend, num_t-1)                                     
    x_bc[:num_t-1,0] = t
    x_bc[:num_t-1,1] = Xstart 
    #x_bc[num_t-1:,0] = t
    #x_bc[num_t-1:,1] = Xend

    
    t_grid, x_grid = np.meshgrid(t, x)                                 
    T = t_grid.flatten()[:, None]                                      
    X = x_grid.flatten()[:, None]                                      
    x_int = X[:, 0][:,None]                                        
    t_int = T[:, 0][:,None]                                        

    x_int = np.hstack((t_int, x_int))                            
    
    return x_ic,x_bc,x_int
    
    
gpu_id = select_gpu()
if gpu_id is not None:
    torch.cuda.set_device(gpu_id)
    device = torch.device('cuda')
    print(f'使用 GPU {gpu_id} (可通过环境变量 GPU_ID 指定，AUTO_GPU=1 自动挑选)')
else:
    device = torch.device('cpu')
    print('未检测到可用 GPU，使用 CPU')

# ========== 参数设置（与notebook统一） ==========
# Notebook: Nx=100, Nt=200
Nx = 100
Nt = 200  # 改为200以匹配notebook（原为50）

x_ic,x_bc,x_int =  Mesh_Data(Nx,Nt,Ts,Te,Xs,Xe)
rho_ic, u_ic, p_ic= IC(x_ic)                    
rho_bc, u_bc, p_bc= BC(x_bc)                    

T1 = Te/2
T3 = Te
dx = 1/Nx/2
dt = 0.002
x_screen1,x_screen1_L,x_screen1_R,x_screen1_P,x_screen1_PL,x_screen1_PR = X_entropy(x_ic,T1,dt,dx)
x_screen2,x_screen2_L,x_screen2_R,x_screen2_P,x_screen2_PL,x_screen2_PR = X_entropy(x_int,0.0,dt,dx)
x_screen3,x_screen3_L,x_screen3_R,x_screen3_P,x_screen3_PL,x_screen3_PR = X_entropy(x_ic,T3,dt,dx)

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

x_ic = torch.tensor(x_ic,requires_grad=True, dtype=dtype).to(device)
x_bc = torch.tensor(x_bc,requires_grad=True, dtype=dtype).to(device)
x_int = torch.tensor(x_int, requires_grad=True, dtype=dtype).to(device)

rho_ic = torch.tensor(rho_ic, dtype=dtype).to(device)
u_ic = torch.tensor(u_ic, dtype=dtype).to(device)
p_ic = torch.tensor(p_ic, dtype=dtype).to(device)

rho_bc = torch.tensor(rho_bc, dtype=dtype).to(device)
u_bc = torch.tensor(u_bc, dtype=dtype).to(device)
p_bc = torch.tensor(p_bc, dtype=dtype).to(device)


print('Start training...')

# ========== GPU模式选择 ==========
use_multi_gpu = os.environ.get('USE_MULTI_GPU', '0') == '1'
num_gpus = torch.cuda.device_count()

model = DNN().to(device).double()
if use_multi_gpu and num_gpus > 1:
    print(f'检测到 {num_gpus} 个GPU，启用多GPU并行训练')
    model = torch.nn.DataParallel(model)
else:
    print(f'检测到 {num_gpus} 个GPU，使用单GPU模式（默认）')
    if num_gpus > 0:
        print(f'使用单GPU: {torch.cuda.get_device_name(0)}')

# ========== 训练 ==========
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epoch = 0
epochi = epoch
epochs = 100000
loss_history = []
interrupted = False
stage_boundary = None
tic_total = time.time()

try:
    tic = time.time()
    for epoch in range(1 + epochi, epochs + epochi):
        loss = train(epoch)
        print(f'loss_tot:{loss:.8f}')
        loss_history.append(to_numpy(loss))
        if loss < 0.05:
            break
    stage_boundary = len(loss_history)
    toc = time.time()
    print(f'Total training time (Adam): {toc - tic:.2f}秒')

    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01, max_iter=30)
    epochi = epoch
    epochs = 2000
    tic = time.time()
    for epoch in range(epochi, epochs + epochi):
        loss = train(epoch)
        print(f'loss_tot:{loss:.8f}')
        loss_history.append(to_numpy(loss))
        #if loss < 0.01:
        #    break
    toc = time.time()
    print(f'Total training time (LBFGS): {toc - tic:.2f}秒')
except KeyboardInterrupt:
    interrupted = True
    print("\n检测到中断 (Ctrl+C)，将保存当前模型与结果...")

training_time = time.time() - tic_total

# ========== 结果保存与评估 ==========
# 预测（保持与训练一致的无量纲/归一化尺度）
x = np.linspace(0.0, Xe, 400)
t = np.full_like(x, Te)
x_test_np = np.stack([t, x], axis=1)
x_test = torch.tensor(x_test_np, requires_grad=False, dtype=dtype).to(device)
actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
with torch.no_grad():
    pred = actual_model(x_test)
rho_pred = to_numpy(pred[:, 0])
p_pred = to_numpy(pred[:, 1])
u_pred = to_numpy(pred[:, 2])

pred_dict = {
    'x': x,
    'rho': rho_pred,
    'p': p_pred,
    'u': u_pred,
    't': Te
}

# 精确解：优先使用本地数值生成的 shuo_exact_sim.dat（compute_exact.py 产生）
# 若不存在则回退 laxe.dat
def load_exact(x_grid):
    for fname in ['shuo_exact_sim.dat', 'laxe.dat']:
        path = os.path.join(current_file_dir, fname)
        if not os.path.exists(path):
            continue
        try:
            data = np.loadtxt(path)
            if data.shape[1] < 4:
                continue
            x_raw = data[:, 0]
            mask = (x_raw >= Xs) & (x_raw <= Xe)
            if not np.any(mask):
                continue
            xx = x_raw[mask]
            rho_e = data[mask, 1]
            u_e   = data[mask, 2]
            p_e   = data[mask, 3]
            rho_i = np.interp(x_grid, xx, rho_e)
            u_i   = np.interp(x_grid, xx, u_e)
            p_i   = np.interp(x_grid, xx, p_e)
            print(f'Loaded exact solution from {fname}')
            return {'x': x_grid, 'rho': rho_i, 'u': u_i, 'p': p_i}
        except Exception as e:
            print(f'Warning: 读取 {fname} 失败: {e}')
    print('Warning: 未找到可用的精确解文件')
    return None

exact_dict = load_exact(x)

# 保存结果
case_name = 'Shu_Osher'
tag = 'baseline'
output_dir, metrics = save_results(
    case_name=case_name,
    model=model,
    pred_dict=pred_dict,
    loss_history=loss_history,
    exact_dict=exact_dict,
    training_time=training_time,
    gpu_count=num_gpus,
    tag=tag,
    stage_boundary=stage_boundary
)

if interrupted:
    print("\n训练被中断，已保存当前结果。")
else:
    print(f"\n训练完成，结果已保存到: {output_dir}")