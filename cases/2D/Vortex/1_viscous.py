# 预留：若未提供边界训练数据，保持 None
x_bcI_train = None
sin_bcI_train = None
cos_bcI_train = None
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import time
import scipy.io
import matplotlib.pyplot as plt
from numpy import arange, meshgrid
import math
from smt.sampling_methods import LHS

# ========== 路径设置（必须放在最前面）==========
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# 从 cases/2D/Vortex 回到仓库根，需要向上三级
repo_root = os.path.abspath(os.path.join(current_file_dir, '..', '..', '..'))
pinn_src_path = os.path.join(repo_root, 'PINNsrc')
# 确保 repo 根目录和 PINNsrc 都在 sys.path，避免环境切换导致找不到 utility
for p in [repo_root, pinn_src_path]:
    if p not in sys.path:
        sys.path.insert(0, p)

# utility中没有save_results等，这里不需要import

# Seeds
dtype = torch.float32

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(6)
 
def train(epoch):
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    def closure():
        optimizer_lbfgs.zero_grad()
        loss_pde = actual_model.loss_pde(x_int_train, mu=mu_visc)
        loss_ic = actual_model.loss_ic(x_ic_train, rho_ic_train, u_ic_train, v_ic_train, p_ic_train)
        loss_bdI = torch.tensor(0.0, device=device)
        if x_bcI_train is not None:
            loss_bdI = actual_model.bd_B(x_bcI_train, sin_bcI_train, cos_bcI_train)
        loss_bdL = actual_model.loss_bc(x_bcL_train, rho_bcL_train, u_bcL_train, v_bcL_train, p_bcL_train)

        loss_ib = loss_ic + loss_bdI  # + loss_bdL（原脚本注释）
        loss = loss_pde + 10 * loss_ib

        print(f'epoch {epoch} loss_pde:{loss_pde:.8f}, loss_ib:{loss_ib:.8f}')
        loss.backward()
        return loss

    loss = optimizer_lbfgs.step(closure)
    return loss
# Calculate gradients using torch.autograd.grad
def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,grad_outputs=torch.ones_like(outputs), create_graph=True)

# Convert torch tensor into np.array
def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '
                        'np.ndarray, but got {}'.format(type(input)))
def IC_Vortex(x):
  N =x.shape[0]
  rho_init = np.zeros((x.shape[0]))
  u_init = np.zeros((x.shape[0])) 
  v_init = np.zeros((x.shape[0]))
  p_init = np.zeros((x.shape[0]))
  x0 = 5.0
  y0 = 5.0
  NI = 10
  NJ = 10
  dx = 1.0
  dy = 1.0
  u_inf = 1.0
  v_inf = 1.0
  GAMMA = 1.4
  b = 1.0
  pi = math.pi


  for i in range(N):
    rx = x[i,1] - x0
    ry = x[i,2] - y0
    if rx < -5:
        rx += 10
    elif rx > 5:
        rx -= 10
    rsq = rx * rx + ry * ry
    rho_init[i] = math.pow(1.0 - ((GAMMA - 1.0) * b * b) / (8.0 * GAMMA * pi * pi) * math.exp(1.0 - rsq),
                   1.0 / (GAMMA - 1.0))
    p_init[i] = math.pow(rho_init[i], GAMMA)
    du = -b / (2.0 * pi) * math.exp(0.5 * (1.0 - rsq)) * ry
    dv = b / (2.0 * pi) * math.exp(0.5 * (1.0 - rsq)) * rx
    u_init[i] = u_inf + du
    v_init[i] = v_inf + dv
     # u0[p] = rho
     # u1[p] = rho * u
     # u2[p] = rho * v
     # u3[p] = P / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)
  return rho_init, u_init, v_init,p_init

def IC(x):
    N =x.shape[0]
    rho_init = np.zeros((x.shape[0]))                                              # rho - initial condition
    u_init = np.zeros((x.shape[0]))                                                # u - initial condition
    v_init = np.zeros((x.shape[0]))                                                # u - initial condition
    p_init = np.zeros((x.shape[0]))                                                # p - initial condition
    
    gamma = 1.4
    rho1 = 1.0
    p1 =  1.16
   # p1 =  1.458
    v1 = 0.0
    u1 = 1.0
    # rho, p - initial condition
    for i in range(N):
        #if x[i,1] < 0.5:
          rho_init[i] = rho1
          u_init[i] =   u1
          v_init[i] =  v1
          p_init[i] =  p1
        #else:
        #    rho_init[i] = rho2
        #    u_init[i] =   u2
        #    v_init[i] =  v2
        #    p_init[i] =  p2

    return rho_init, u_init, v_init,p_init

def BC_L(x):
    N =x.shape[0]
    rho_init = np.zeros((x.shape[0]))                                              # rho - initial condition
    u_init = np.zeros((x.shape[0]))                                                # u - initial condition
    v_init = np.zeros((x.shape[0]))                                                # u - initial condition
    p_init = np.zeros((x.shape[0]))                                                # p - initial condition
    
    gamma = 1.4
    #u1 = ms*npsqrt(gamma)
    # rho, p - initial condition
    rho1 = 1.0
    p1 =  1.116
    v1 = 0.0
    u1 = 1.0
    for i in range(N):
        rho_init[i] = rho1
        u_init[i] =  u1
        v_init[i] =  v1
        p_init[i] =  p1
    return rho_init, u_init, v_init,p_init
def BC_R(x):
    N =x.shape[0]
    rho_init = np.zeros((x.shape[0]))                                              # rho - initial condition
    u_init = np.zeros((x.shape[0]))                                                # u - initial condition
    v_init = np.zeros((x.shape[0]))                                                # u - initial condition
    p_init = np.zeros((x.shape[0]))                                                # p - initial condition
    
    gamma = 1.4
    ms = 2.0
    rho1 = 1.0
    p1 = 1.0
    v1 = 0.0
    u1 = 0
    # rho, p - initial condition
    for i in range(N):
        rho_init[i] = rho1
        u_init[i] = u1
        v_init[i] = v1
        p_init[i] = p1

    return rho_init, u_init, v_init,p_init
def BC_Cut(x):
    N =x.shape[0]
    rho_init = np.zeros((x.shape[0]))
    u_init = np.zeros((x.shape[0]))
    v_init = np.zeros((x.shape[0]))
    p_init = np.zeros((x.shape[0]))
    
    gamma = 1.4
    ms = 2.0
    rho1 = 1.0
    p1 = 1.0
    v1 = 0.0
    u1 = 0
    # rho, p - initial condition
    for i in range(N):
        rho_init[i] = 10.01
        u_init[i] =  0
        v_init[i] = 0
        p_init[i] = 10.01

    return rho_init, u_init, v_init,p_init
    
class DNN(nn.Module):

    def __init__(self):
        super(DNN, self).__init__()
        self.net = nn.Sequential()                                                  # Define neural network
        self.net.add_module('Linear_layer_1', nn.Linear(3, 90))                     # First linear layer
        self.net.add_module('Tanh_layer_1', nn.Tanh())                              # First activation Layer

        for num in range(2, 6):                                                     # Number of layers (2 through 7)
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(90, 90))       # Linear layer
            self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())                 # Activation Layer
        self.net.add_module('Linear_layer_final', nn.Linear(90, 4))                 # Output Layer

    # Forward Feed
    def forward(self, x):
        return self.net(x)

    def bd_B(self,x,sin,cos):
        yb = self.net(x)
        rhob,pb,ub,vb = yb[:, 0:1], yb[:, 1:2], yb[:, 2:3],yb[:,3:]
        drhob_g = gradients(rhob, x)[0]                                      # Gradient [u_t, u_x]
        rhob_x, rhob_y = drhob_g[:, 1:2], drhob_g[:, 2:3]                            # Partial derivatives u_t, u_x
        dub_g = gradients(ub, x)[0]                                      # Gradient [u_t, u_x]
        ub_x, ub_y = dub_g[:, 1:2], dub_g[:, 2:3]                            # Partial derivatives u_t, u_x
        dvb_g = gradients(vb, x)[0]                                      # Gradient [u_t, u_x]
        vb_x, vb_y = dvb_g[:, 1:2], dvb_g[:, 2:3]                            # Partial derivatives u_t, u_x
        dpb_g = gradients(pb, x)[0]                                      # Gradient [p_t, p_x]
        pb_x, pb_y = dpb_g[:, 1:2], dpb_g[:, 2:3]                            # Partial derivatives p_t, p_x
        
        deltau = ub_x + vb_y
        lam = 0.1*(abs(deltau) - deltau) + 1
        #lam = (deltau) - deltau) + 1
        
        fb = (((ub*cos + vb*sin)/lam)**2).mean() +\
            (((pb_x*cos + pb_y*sin)/lam)**2).mean() +\
            (((rhob_x*cos + rhob_y*sin)/lam)**2).mean()
        return fb
    def bd_OY(self,x):
        y = self.net(x)
        rho,p,u,v = y[:, 0:1], y[:, 1:2], y[:, 2:3],y[:,3:]
        
        drho_g = gradients(rho, x)[0]                                  # Gradient [rho_t, rho_x]
        rho_x,rho_y = drho_g[:, :1], drho_g[:, 1:2]                    # Partial derivatives rho_t, rho_x
        du_g = gradients(u, x)[0]                                      # Gradient [u_t, u_x]
        u_x, u_y = du_g[:, :1], du_g[:, 1:2]                            # Partial derivatives u_t, u_x
        dv_g = gradients(v, x)[0]                                      # Gradient [u_t, u_x]
        v_x, v_y = dv_g[:, :1], dv_g[:, 1:2]                            # Partial derivatives u_t, u_x
        dp_g = gradients(p, x)[0]                                      # Gradient [p_t, p_x]
        p_x, p_y = dp_g[:, :1], dp_g[:, 1:2]                            # Partial derivatives p_t, p_x
        
        deltau = u_x + v_y
        lam = 0.1*(abs(deltau) - deltau) + 1
        
        f = ((( u_y)/lam)**2).mean() +\
            ((( v_y)/lam)**2).mean() +\
            ((( p_y)/lam)**2).mean() +\
            ((( rho_y)/lam)**2).mean()
        return f
    
    def bd_OX(self,x):
        y = self.net(x)
        rho,p,u,v = y[:, 0:1], y[:, 1:2], y[:, 2:3],y[:,3:]
        
        drho_g = gradients(rho, x)[0]                                  # Gradient [rho_t, rho_x]
        rho_x,rho_y = drho_g[:, :1], drho_g[:, 1:2]                    # Partial derivatives rho_t, rho_x
        du_g = gradients(u, x)[0]                                      # Gradient [u_t, u_x]
        u_x, u_y = du_g[:, :1], du_g[:, 1:2]                            # Partial derivatives u_t, u_x
        dv_g = gradients(v, x)[0]                                      # Gradient [u_t, u_x]
        v_x, v_y = dv_g[:, :1], dv_g[:, 1:2]                            # Partial derivatives u_t, u_x
        dp_g = gradients(p, x)[0]                                      # Gradient [p_t, p_x]
        p_x, p_y = dp_g[:, :1], dp_g[:, 1:2]                            # Partial derivatives p_t, p_x
        
        deltau = u_x + v_y
        lam = 0.1*(abs(deltau) - deltau) + 1
        
        f = ((( u_x)/lam)**2).mean() +\
            ((( v_x)/lam)**2).mean() +\
            ((( p_x)/lam)**2).mean() +\
            ((( rho_x)/lam)**2).mean()
        return f
     
    # Loss function for PDE
#    def loss_pde(self, x):
#        
#       # yL = self.net(x_intL_train)
#       # yR = self.net(x_intR_train)
#       # yU = self.net(x_intU_train)
#       # yD = self.net(x_intD_train)
#       # rhoL,pL,uL,vL = yL[:, 0:1], yL[:, 1:2], yL[:, 2:3],yL[:,3:]
#       # rhoR,pR,uR,vR = yR[:, 0:1], yR[:, 1:2], yR[:, 2:3],yR[:,3:]
#       # rhoU,pU,uU,vU = yU[:, 0:1], yU[:, 1:2], yU[:, 2:3],yU[:,3:]
#       # rhoD,pD,uD,vD = yD[:, 0:1], yD[:, 1:2], yD[:, 2:3],yD[:,3:]
#        y = self.net(x)
#        gamma = 1.4                                                    # Heat Capacity Ratio
#        epsilon = 1e-5
#        rho,p,u,v = y[:, 0:1], y[:, 1:2], y[:, 2:3],y[:,3:]
#        
#        rhoE = p/(gamma - 1) +0.5*rho*(u**2+v**2)
#        
#        f1 = rho*u
#        f2 = rho*u*u+p
#        f3 = rho*u*v
#        f4 = (rhoE+p)*u
#        
#        g1 = rho*v
#        g2 = rho*v*u
#        g3 = rho*v*v + p
#        g4 = (rhoE+p)*v
#        
#        drho_g = gradients(rho,x)[0]
#        U1_t = drho_g[:, :1]
#        dU2_g = gradients(f1,x)[0]
#        U2_t = dU2_g[:, :1]
#        dU3_g = gradients(g1,x)[0]
#        U3_t = dU3_g[:, :1]
#        dU4_g = gradients(rhoE,x)[0]
#        U4_t = dU4_g[:, :1]
#        
#        df1_g = gradients(f1, x)[0]     G                             # Gradient [rho_t, rho_x]
#        f1_x = df1_g[:, 1:2]
#        df2_g = gradients(f2, x)[0]                                      # Gradient [u_t, u_x]
#        f2_x = df2_g[:, 1:2]
#        df3_g = gradients(f3, x)[0]                                      # Gradient [u_t, u_x]
#        f3_x = df3_g[:, 1:2]
#        df4_g = gradients(f4, x)[0]                                      # Gradient [u_t, u_x]
#        f4_x = df4_g[:, 1:2]
#        
#        dg1_g = gradients(g1, x)[0]                                  # Gradient [rho_t, rho_x]
#        g1_y = dg1_g[:, 2:3]
#        dg2_g = gradients(g2, x)[0]                                      # Gradient [u_t, u_x]
#        g2_y = dg2_g[:, 2:3]
#        dg3_g = gradients(g3, x)[0]                                      # Gradient [u_t, u_x]
#        g3_y = dg3_g[:, 2:3]
#        dg4_g = gradients(g4, x)[0]                                      # Gradient [u_t, u_x]
#        g4_y = dg4_g[:, 2:3]
#        
#        
#        du_g = gradients(u, x)[0]                                
#        u_x = du_g[:, 1:2]         
#        dv_g = gradients(v, x)[0]                    
#        v_y = dv_g[:, 2:3]         
#        
#      #  rho,p,u,v = y[:, 0:1], y[:, 1:2], y[:, 2:3],y[:,3:]
#      #  gamma = 1.4                                                    # Heat Capacity Ratio
#      #  epsilon = 1e-5
#      #  s = torch.log((abs(p)+epsilon)/(abs(rho)+epsilon)**1.4)
#      #  eta = -rho*s
#      #  phi1 = -rho*u*s
#      #  phi2 = -rho*v*s
#      #  
#      #  drho_g = gradients(rho, x)[0]                                  # Gradient [rho_t, rho_x]
#      #  rho_t, rho_x,rho_y = drho_g[:, :1], drho_g[:, 1:2],drho_g[:,2:]
#      #  du_g = gradients(u, x)[0]                                      # Gradient [u_t, u_x]
#      #  u_t, u_x, u_y = du_g[:, :1], du_g[:, 1:2], du_g[:,2:]                            # Partial derivatives u_t, u_x
#      #  dv_g = gradients(v, x)[0]                                      # Gradient [u_t, u_x]
#      #  v_t, v_x, v_y = dv_g[:, :1], dv_g[:, 1:2], dv_g[:,2:]                            # Partial derivatives u_t, u_x
#      #  
#      #  E = p/0.4 + 0.5*rho*(u**2+v**2)
#      #  EL = pL/0.4 + 0.5*rhoL*(uL**2+vL**2)
#      #  ER = pR/0.4 + 0.5*rhoR*(uR**2+vR**2)
#      #  EU = pU/0.4 + 0.5*rhoU*(uU**2+vU**2)
#      #  ED = pD/0.4 + 0.5*rhoD*(uD**2+vD**2)
#      #  dE_g = gradients(E, x)[0]                                      # Gradient [u_t, u_x]
#      #  E_t = dE_g[:, :1]
#      #  
#      #  
#      #  deta_g = gradients(eta, x)[0]                                      # Gradient [p_t, p_x]
#      #  eta_t, eta_x,eta_y = deta_g[:, :1], deta_g[:, 1:2],deta_g[2:3]                            # Partial derivatives p_t, p_x
#      #  dphi1_g = gradients(phi1, x)[0]                                      # Gradient [p_t, p_x]
#      #  dphi2_g = gradients(phi2, x)[0]                                      # Gradient [p_t, p_x]
#      #  phi1_t, phi1_x,phi1_y = dphi1_g[:, :1], dphi1_g[:, 1:2],dphi1_g[:,2:3]                           # Partial derivatives p_t, p_x
#      #  phi2_t, phi2_x,phi2_y = dphi2_g[:, :1], dphi2_g[:, 1:2],dphi2_g[:,2:3]                           # Partial derivatives p_t, p_x
#        
#        d = np.random.rand()
#        deltau = u_x + v_y
#        nab = abs(deltau) - deltau
#        
#        #a = np.sqrt(1.4*p/rho)
#       # q = 0.01*(rho*deltau**2)
#        
#        d = 1.0
#        lam = d*(0.1*nab) + 1
#        #lam = d + 1
#       # lam = 1/lam
#        
#        f = (((U1_t + f1_x+g1_y )/lam)**2).mean() +\
#            (((U2_t + f2_x+g2_y )/lam)**2).mean() +\
#            (((U3_t + f3_x+g3_y )/lam)**2).mean() +\
#            (((U4_t + f4_x+g4_y )/lam)**2).mean()
#
#      #  p = p+q
#        
#      #  dp_g = gradients(p, x)[0]                                      # Gradient [p_t, p_x]
#      #  p_t, p_x, p_y = dp_g[:, :1], dp_g[:, 1:2], dp_g[:,2:]                            # Partial derivatives p_t, p_x
#      #  
#      #  s1 = rho_t + (rhoR*uR - rhoL*uL)/0.02 +  (rhoU*uU - rhoD*uD)/0.02 
#      #  s2 = u*rho_t + u_t*rho + (rhoR*uR*uR +pR - rhoL*uL*uL-pL)/0.02 \
#      #       +(rhoU*uU*vU - rhoD*uD*vD)/0.02 
#      #  s3 = v*rho_t + v_t*rho + (rhoU*vU*vU +pU - rhoD*vD*vD-pD)/0.02 \
#      #       +(rhoR*uR*vR - rhoL*uL*vL)/0.02 
#      #  s4 = E_t + ((ER+pR)*uR - (EL+pL)*uL)/0.02 + ((EU+pU)*vU - (ED+pD)*vD)/0.02 
#      #  
#      #  du_gg = gradients(u_x, x)[0]                                      # Gradient [u_t, u_x]
#      #  u_xx, u_xy = du_gg[:, :1], du_gg[:, 1:2]                            # Partial derivatives u_t, u_x
#      #  
#      #  dv_gg = gradients(v_y, x)[0]                                      # Gradient [u_t, u_x]
#      #  v_yx, v_yy = dv_gg[:, :1], dv_gg[:, 1:2]                            # Partial derivatives u_t, u_x
#      #  
#      #  vis = -0.1*(u_xx + v_yy)
##
#      #  f = (((rho_t+rho*deltau+u*rho_x + v*rho_y)/lam)**2).mean() +\
#      #      (((rho*u_t+rho*u*u_x+rho*v*u_y+p_x +rho*vis)/lam)**2).mean() +\
#      #      (((rho*v_t+rho*u*v_x+rho*v*v_y+p_y +rho*vis)/lam)**2).mean() +\
#      #      (((p_t+u*p_x+v*p_y+1.4*p*deltau +rho*vis)/lam)**2).mean() + \
#      #      ((abs(s1)+s1)**2).mean() +\
#      #      ((abs(s3)+s3)**2).mean() +\
#      #      ((abs(s2)+s2)**2).mean() +\
#      #      ((abs(s4)+s4)**2).mean()
#      #      #(((abs(eta_t+phi1_x + phi2_y)+eta_t+phi1_x+ phi2_y))**2).mean()
#    #
#      #      #((abs(rho-1) - (rho-1))**2).mean()   + \
#      #      #((abs(p-0.7) - (p-0.7))**2).mean() +\
#        return f

    def loss_pde(self, x, mu=0.0):
        y = self.net(x)
        rho,p,u,v = y[:, 0:1], y[:, 1:2], y[:, 2:3],y[:,3:]

        gamma = 1.4
        # Gradients and partial derivatives
        drho_g = gradients(rho, x)[0]
        rho_t, rho_x, rho_y = drho_g[:, :1], drho_g[:, 1:2], drho_g[:, 2:]
        du_g = gradients(u, x)[0]
        u_t, u_x, u_y = du_g[:, :1], du_g[:, 1:2], du_g[:, 2:]
        dv_g = gradients(v, x)[0]
        v_t, v_x, v_y = dv_g[:, :1], dv_g[:, 1:2], dv_g[:, 2:]
        dp_g = gradients(p, x)[0]
        p_t, p_x, p_y = dp_g[:, :1], dp_g[:, 1:2], dp_g[:, 2:]

        deltau = u_x + v_y
        lam = 0.1*(abs(deltau) - deltau) + 1

        # Euler (inviscid) residuals
        R_mass = rho_t + rho*deltau + u*rho_x + v*rho_y
        R_momx = rho*u_t + rho*u*u_x + rho*v*u_y + p_x
        R_momy = rho*v_t + rho*u*v_x + rho*v*v_y + p_y
        R_ener = p_t + u*p_x + v*p_y + gamma*p*deltau

        # Viscous terms (only when mu > 0)
        if mu > 0:
            # Second derivatives for viscous stress
            u_xx = gradients(u_x, x)[0][:, 1:2]
            u_yy = gradients(u_y, x)[0][:, 2:3]
            v_xx = gradients(v_x, x)[0][:, 1:2]
            v_yy = gradients(v_y, x)[0][:, 2:3]

            div_u_x = gradients(deltau, x)[0][:, 1:2]
            div_u_y = gradients(deltau, x)[0][:, 2:3]

            # 交叉导数
            v_xy = gradients(v_x, x)[0][:, 2:3]   # ∂(v_x)/∂y
            u_xy = gradients(u_y, x)[0][:, 1:2]   # ∂(u_y)/∂x

            # ∂τ_xx/∂x + ∂τ_xy/∂y  (acting on x-momentum)
            #   τ_xx = μ(2u_x - 2/3 div_u)  →  ∂/∂x = μ(2u_xx - 2/3 div_u_x)
            #   τ_xy = μ(u_y + v_x)         →  ∂/∂y = μ(u_yy + v_xy)
            visc_x = mu * (2*u_xx - (2.0/3.0)*div_u_x) + mu * (u_yy + v_xy)
            # ∂τ_yx/∂x + ∂τ_yy/∂y  (acting on y-momentum)
            #   τ_yx = μ(u_y + v_x)         →  ∂/∂x = μ(u_xy + v_xx)
            #   τ_yy = μ(2v_y - 2/3 div_u)  →  ∂/∂y = μ(2v_yy - 2/3 div_u_y)
            visc_y = mu * (u_xy + v_xx) + mu * (2*v_yy - (2.0/3.0)*div_u_y)

            R_momx = R_momx - visc_x
            R_momy = R_momy - visc_y

            # Viscous dissipation Φ = τ_ij ∂u_i/∂x_j
            tau_xx = mu * (2*u_x - (2.0/3.0)*deltau)
            tau_yy = mu * (2*v_y - (2.0/3.0)*deltau)
            tau_xy = mu * (u_y + v_x)
            Phi = tau_xx*u_x + tau_yy*v_y + tau_xy*(u_y + v_x)

            R_ener = R_ener - (gamma - 1.0) * Phi

        f = ((R_mass/lam)**2).mean() +\
            ((R_momx/lam)**2).mean() +\
            ((R_momy/lam)**2).mean() +\
            ((R_ener/lam)**2).mean()

        return f
      
      
    # Loss function for initial condition
    def loss_ic(self, x_ic, rho_ic, u_ic, v_ic,p_ic):
        U_ic = self.net(x_ic)                                                      # Initial condition
        rho_ic_nn, p_ic_nn,u_ic_nn,v_ic_nn = U_ic[:, 0], U_ic[:, 1], U_ic[:, 2],U_ic[:,3]            # rho, u, p - initial condition

        # Loss function for the initial condition
        loss_ics = ((u_ic_nn - u_ic) ** 2).mean() + \
               ((rho_ic_nn- rho_ic) ** 2).mean()  + \
               ((p_ic_nn - p_ic) ** 2).mean() +\
               ((v_ic_nn - v_ic) ** 2).mean()

        return loss_ics

    def loss_bc(self, x_ic, rho_ic, u_ic, v_ic,p_ic):
        U_ic = self.net(x_ic)                                                      # Initial condition
        rho_ic_nn, p_ic_nn,u_ic_nn,v_ic_nn = U_ic[:, 0], U_ic[:, 1], U_ic[:, 2],U_ic[:,3]            # rho, u, p - initial condition

        # Loss function for the initial condition
        loss_ics = ((u_ic_nn - u_ic) ** 2).mean() + \
               ((rho_ic_nn- rho_ic) ** 2).mean()  + \
               ((p_ic_nn - p_ic) ** 2).mean() +\
               ((v_ic_nn - v_ic) ** 2).mean()

        return loss_ics
    def loss_bc1(self, x_ic, rho_ic, u_ic, v_ic,p_ic):
        U_ic = self.net(x_ic)                                                      # Initial condition
        rho_ic_nn, p_ic_nn,u_ic_nn,v_ic_nn = U_ic[:, 0], U_ic[:, 1], U_ic[:, 2],U_ic[:,3]            # rho, u, p - initial condition

        # Loss function for the initial condition
        loss_ics = ((rho_ic_nn- rho_ic) ** 2).mean()  + \
               ((p_ic_nn - p_ic) ** 2).mean() 

        return loss_ics

def IC_circle(t,xc,yc,r,r2,n):
    x = np.zeros((n,3)) 

    for i in range(n):
        the = 2*np.random.rand()*np.pi
        xd = np.cos(the + np.pi/2)
        yd = np.sin(the + np.pi/2)
        rr = np.random.rand()*(r2-r) + r
        x[i,0] = np.random.rand()*t
        x[i,1] = xc  + xd*rr
        x[i,2] = yc  + yd*rr
    return x

def IC_circle_init(xc,yc,r,r2,n):
    x = np.zeros((n,3)) 

    for i in range(n):
        the = 2*np.random.rand()*np.pi
        xd = np.cos(the + np.pi/2)
        yd = np.sin(the + np.pi/2)
        rr = np.random.rand()*(r2-r) + r
        x[i,0] = 0 #np.random.rand()*t
        x[i,1] = xc  + xd*rr
        x[i,2] = yc  + yd*rr
    return x


def BD_circle(t,xc,yc,r,n):
    x = np.zeros((n,3)) 
    sin = np.zeros((n,1)) 
    cos = np.zeros((n,1)) 

    for i in range(n):
        the = 2*np.random.rand()*np.pi
        xd = np.cos(the + np.pi/2)
        yd = np.sin(the + np.pi/2)
        x[i,0] = np.random.rand()*t
        x[i,1] = xc  + xd*r
        x[i,2] = yc  + yd*r
        cos[i,0] = xd 
        sin[i,0] = yd
        #cos[i,0] = 1
        #sin[i,0] = 0
    return x, sin,cos

def Pertur(x, dx):
    N =x.shape[0]
    xL = np.zeros((N,3))
    xR = np.zeros((N,3))
    xU = np.zeros((N,3))
    xD = np.zeros((N,3))
    
    for i in range(N):
        xL[i,0] = x[i,0]
        xR[i,0] = x[i,0]
        xU[i,0] = x[i,0]
        xD[i,0] = x[i,0]
        
        
        xL[i,1] = x[i,1] - dx
        xR[i,1] = x[i,1] + dx
        xU[i,1] = x[i,1]
        xD[i,1] = x[i,1]
        
        xL[i,2] = x[i,2] 
        xR[i,2] = x[i,2]
        xU[i,2] = x[i,2] + dx
        xD[i,2] = x[i,2] - dx
        
    return xL,xR,xU,xD
    
    
def Naca0012data(x):
    a = 0.594689181
    b = 0.298222773  
    c = 0.127125232 
    d = 0.357907906 
    e = 0.291984971 
    f = 0.105174606 
    y1 = a*(b*np.sqrt(x) - c*x-d*x**2+e*x**3 - f*x**4)
    y2 = -y1
    dy1 =  a*(0.5*b/np.sqrt(x) - c - 2*d*x +3*e*x**2 - 4*f*x**3)
    dy2 = -a*(0.5*b/np.sqrt(x) - c - 2*d*x +3*e*x**2 - 4*f*x**3)
    return y1,y2,dy1,dy2

def BD_naca0012(t,xb,yb,n):
    x = np.zeros((2*n,3)) 
    sin = np.zeros((2*n,1)) 
    cos = np.zeros((2*n,1)) 

    for i in range(n):
        xd = np.random.rand()
        yd1,yd2,dy1,dy2 = Naca0012data(xd)
        
        x[i,0] = np.random.rand()*t
        x[i,1] = xb + xd
        x[i,2] = yb  + yd1
        cos[i,0] = -dy1/np.sqrt(dy1**2 + 1)
        sin[i,0] =   1/np.sqrt(dy1**2 + 1)
    for i in range(n):
        xd = np.random.rand()
        yd1,yd2,dy1,dy2 = Naca0012data(xd)
        
        x[i+n,0] = np.random.rand()*t
        x[i+n,1] = xb + xd
        x[i+n,2] = yb  + yd2
        cos[i+n,0] = -dy2/np.sqrt(dy2**2 + 1)
        sin[i+n,0] =  1/np.sqrt(dy2**2 + 1)
    return x, sin,cos
    
def BD_BackCorner(t,n):
    
    x = np.zeros((n,3)) 
    x2 = np.zeros((n,3)) 
    sin = np.zeros((n,1)) 
    sin2 = np.zeros((n,1)) 
    cos = np.zeros((n,1)) 
    cos2 = np.zeros((n,1)) 
    
    for i in range(n):
        x[i,0] = np.random.rand()*t
        x[i,1] = np.random.rand()*0.3 + 0.2
        x[i,2] = 1.5
        sin[i] = 1
        cos[i] = 0
    for i in range(n):
        x2[i,0] = np.random.rand()*t
        x2[i,1] = np.random.rand()*0.5
        x2[i,2] = 1.5
        sin2[i] = 1
        cos2[i] = 0
    x = np.vstack((x,x2))
    sin = np.vstack((sin,sin2))
    cos = np.vstack((cos,cos2))
    
    for i in range(n):
        x2[i,0] = np.random.rand()*t
        x2[i,1] = 0.5
        x2[i,2] = np.random.rand()*1.5
        sin2[i] = 0
        cos2[i] = 1
        
    x = np.vstack((x,x2))
    sin = np.vstack((sin,sin2))
    cos = np.vstack((cos,cos2))
    
    for i in range(n):
        x2[i,0] = np.random.rand()*t
        x2[i,1] = 0.5
        x2[i,2] = np.random.rand()*0.3 + 1.2
        sin2[i] = 0
        cos2[i] = 1
        
    x = np.vstack((x,x2))
    sin = np.vstack((sin,sin2))
    cos = np.vstack((cos,cos2))
        
    return x,sin,cos

device = torch.device('cuda')                                          # placeholder, will be overridden by select_gpu
lr = 0.001                                                           # Learning rate
num_ib = 10000                                              # Random sampled points from IC0
num_int =50000                                                # Random sampled points in interior
Tend = 0.5
Lx = 10.00
Lx1 = 0.0
Ly = 10.0
Ly1 =0.0
rx = 0.5
ry = 1.0
rd = 0.25

# ========== 粘性参数 ==========
Re = 100.0       # Reynolds number
u_inf = 1.0      # 参考速度 (来流)
rho_ref = 1.0    # 参考密度
L_ref = 1.0      # 参考长度 (涡核半径 b=1)
mu_visc = rho_ref * u_inf * L_ref / Re   # 动力粘度 = 0.01
print(f'2D Isentropic Vortex — Viscous (Re={Re}, mu={mu_visc:.4f})')


xlimits = np.array([[0.,Tend],[Lx1, Lx], [Ly1,Ly]])  #interal
sampling = LHS(xlimits=xlimits)
x_int_train = sampling(num_ib)
#A = []
#for i in range(num_ib):
#    x = x_int_train[i,1]
#    y = x_int_train[i,2]
# #   if ((x - rx)>0 and (x-rx)<1):
# #       y1,y2,dy1,dy2 = Naca0012data(x-rx)
# #       if ((y-ry)>y2 and (y-ry)<y1):
# #           A.append(i)
#x_int_train = np.delete(x_int_train,A,axis=0)

#xlimits = np.array([[0.,Tend],[0, 2], [Ly1,Ly]])  #interal
#sampling = LHS(xlimits=xlimits)
#x_int_train1 = sampling(num_int)
#A = []
#for i in np.shape(num_int):
#    x = x_int_train1[i,1]
#    y = x_int_train1[i,2]
#    if ((x - rx)>0 and (x-rx)<1):
#        y1,y2,dy1,dy2 = Naca0012data(x-rx)
#        if ((y-ry)>y2 and (y-ry)<y1):
#            A.append(i)
#x_int_train1 = np.delete(x_int_train1,A,axis=0)
#
#x_int_train = np.concatenate((x_int_train,x_int_train1),axis=0)


xlimits = np.array([[0.,0.0],[Lx1,Lx], [Ly1,Ly]])  #interal
sampling = LHS(xlimits=xlimits)
x_ic_train = sampling(num_ib)
#A = []
#for i in range(num_ib):
#    x = x_ic_train[i,1]
#    y = x_ic_train[i,2]
#    if ((x - rx)>0 and (x-rx)<1):
#        y1,y2,dy1,dy2 = Naca0012data(x-rx)
#        if ((y-ry)>y2 and (y-ry)<y1):
#            A.append(i)
#x_ic_train = np.delete(x_ic_train,A,axis=0)
#
#x_ic_pre=np.copy(x_ic_train)
#x_ic_pre[:,0] = 5.0

#x_bcI_train,sin_bcI_train,cos_bcI_train = BD_naca0012(Tend,rx,ry,num_ib)
rho_ic_train, u_ic_train,v_ic_train, p_ic_train = IC_Vortex(x_ic_train)

xlimits = np.array([[0.0,Tend],[Lx1, Lx1], [Ly1,Ly]])
sampling = LHS(xlimits=xlimits)
x_bcL_train =  sampling(num_ib)

rho_bcL_train, u_bcL_train,v_bcL_train, p_bcL_train = IC_Vortex(x_bcL_train)  

x_int_train = torch.tensor(x_int_train, requires_grad=True, dtype=dtype).to(device)
#x_bcI_train = torch.tensor(x_bcI_train, requires_grad=True, dtype=dtype).to(device)
#sin_bcI_train = torch.tensor(sin_bcI_train, dtype=dtype).to(device)
#cos_bcI_train = torch.tensor(cos_bcI_train, dtype=dtype).to(device)

#rho_ic_train = torch.tensor(rho_ic_train, dtype=torch.float32).to(device)
#u_ic_train = torch.tensor(u_ic_train, dtype=torch.float32).to(device)
#v_ic_train = torch.tensor(v_ic_train, dtype=torch.float32).to(device)
#p_ic_train = torch.tensor(p_ic_train, dtype=torch.float32).to(device)
x_ic_train = torch.tensor(x_ic_train, dtype=dtype).to(device)
#x_ic_pre = torch.tensor(x_ic_pre, dtype=torch.float32).to(device)

rho_bcL_train = torch.tensor(rho_bcL_train, dtype=dtype).to(device)
u_bcL_train = torch.tensor(u_bcL_train, dtype=dtype).to(device)
v_bcL_train = torch.tensor(v_bcL_train, dtype=dtype).to(device)
p_bcL_train = torch.tensor(p_bcL_train, dtype=dtype).to(device)
x_bcL_train = torch.tensor(x_bcL_train, dtype=dtype).to(device)


#print('Start training...')
#
#model1 = torch.load(model_path, map_location=torch.device('cpu'))
#model1 = model1.to(device)

#U_ic = to_numpy(model1(x_ic_pre))
#rho_ic_train, p_ic_train,u_ic_train,v_ic_train = U_ic[:, 0], U_ic[:, 1], U_ic[:, 2],U_ic[:,3]            # rho, u, p - initial condition

rho_ic_train = torch.tensor(rho_ic_train, dtype=dtype).to(device)
u_ic_train = torch.tensor(u_ic_train, dtype=dtype).to(device)
v_ic_train = torch.tensor(v_ic_train, dtype=dtype).to(device)
p_ic_train = torch.tensor(p_ic_train, dtype=dtype).to(device)

# ========== GPU选择 (由CUDA_VISIBLE_DEVICES控制) ==========
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'使用 GPU: {torch.cuda.get_device_name(0)}')
else:
    device = torch.device('cpu')
    print('未检测到可用 GPU，使用 CPU')

# 重新将已有张量搬到选定 device，避免默认落在 cuda:0
x_int_train = x_int_train.to(device)
x_ic_train = x_ic_train.to(device)
rho_bcL_train = rho_bcL_train.to(device)
u_bcL_train = u_bcL_train.to(device)
v_bcL_train = v_bcL_train.to(device)
p_bcL_train = p_bcL_train.to(device)
x_bcL_train = x_bcL_train.to(device)
rho_ic_train = rho_ic_train.to(device)
u_ic_train = u_ic_train.to(device)
v_ic_train = v_ic_train.to(device)
p_ic_train = p_ic_train.to(device)

# ========== 模型 ==========
model = DNN().to(device)

# ========== 训练: Adam预热 + LBFGS精修 ==========
loss_history = []
interrupted = False
tic_total = time.time()

actual_model = model

# --- Adam 阶段 ---
print('\n========== Adam ==========')
epochs_adam = 10000
optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_adam, step_size=3000, gamma=0.5)

try:
    for epoch in range(1, epochs_adam + 1):
        optimizer_adam.zero_grad()
        loss_pde = actual_model.loss_pde(x_int_train, mu=mu_visc)
        loss_ic = actual_model.loss_ic(x_ic_train, rho_ic_train, u_ic_train, v_ic_train, p_ic_train)
        loss_bdI = torch.tensor(0.0, device=device)
        if x_bcI_train is not None:
            loss_bdI = actual_model.bd_B(x_bcI_train, sin_bcI_train, cos_bcI_train)
        loss_bdL = actual_model.loss_bc(x_bcL_train, rho_bcL_train, u_bcL_train, v_bcL_train, p_bcL_train)
        loss_ib = loss_ic + loss_bdI
        loss = loss_pde + 10 * loss_ib
        loss.backward()
        optimizer_adam.step()
        scheduler.step()
        loss_history.append(to_numpy(loss))

        if epoch % 1000 == 0 or epoch == 1:
            lr_now = optimizer_adam.param_groups[0]['lr']
            print(f'  epoch {epoch:5d}  loss={loss.item():.4e}  '
                  f'pde={loss_pde.item():.4e}  ic={loss_ic.item():.4e}  lr={lr_now:.1e}')

    adam_time = time.time() - tic_total
    print(f'Adam done: {adam_time:.1f}s')

    # --- LBFGS 阶段 ---
    print('\n========== LBFGS ==========')
    lbfgs_lr = 1.0
    lbfgs_max_iter = 30
    lbfgs_epochs = 500
    stage_boundary = len(loss_history)

    optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=lbfgs_lr,
                                         max_iter=lbfgs_max_iter,
                                         line_search_fn='strong_wolfe')
    for epoch in range(1, lbfgs_epochs + 1):
        loss = train(epoch)
        loss_val = to_numpy(loss)
        if np.isnan(loss_val):
            print(f'  LBFGS {epoch}: NaN detected, stopping LBFGS')
            break
        print(f'loss_tot:{loss_val:.8f}')
        loss_history.append(loss_val)

    lbfgs_time = time.time() - tic_total - adam_time
    print(f'LBFGS done: {lbfgs_time:.1f}s')

except KeyboardInterrupt:
    interrupted = True
    print("\n检测到中断 (Ctrl+C)，将保存当前模型与结果...")

training_time = time.time() - tic_total

# ========== 预测与结果保存 ==========
actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
actual_model.eval()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_name = f'Vortex2D_Viscous_Re{Re:.0f}_{timestamp}'
output_dir = os.path.join(current_file_dir, 'output', run_name)
os.makedirs(output_dir, exist_ok=True)

# ========== 多时刻预测 ==========
nx = ny = 200   # 高分辨率网格
x_lin = np.linspace(Lx1, Lx, nx)
y_lin = np.linspace(Ly1, Ly, ny)
Xg, Yg = np.meshgrid(x_lin, y_lin, indexing='ij')

t_snapshots = [0.0, 0.1, 0.25, 0.5]
all_preds = {}

for t_val in t_snapshots:
    Tg = np.full_like(Xg, t_val)
    x_test_np = np.stack([Tg, Xg, Yg], axis=-1).reshape(-1, 3)
    x_test = torch.tensor(x_test_np, dtype=dtype).to(device)
    with torch.no_grad():
        pred = actual_model(x_test)
    all_preds[t_val] = {
        'rho': to_numpy(pred[:, 0]).reshape(nx, ny),
        'p':   to_numpy(pred[:, 1]).reshape(nx, ny),
        'u':   to_numpy(pred[:, 2]).reshape(nx, ny),
        'v':   to_numpy(pred[:, 3]).reshape(nx, ny),
    }
print(f'多时刻预测完成: {t_snapshots}')

# ========== 精确初始条件 (用于 t=0 对比) ==========
rho_exact = np.zeros((nx, ny)); u_exact = np.zeros((nx, ny))
v_exact = np.zeros((nx, ny)); p_exact = np.zeros((nx, ny))
x0, y0_c = 5.0, 5.0; b_vort = 1.0; GAMMA = 1.4
for i in range(nx):
    for j in range(ny):
        rx = x_lin[i] - x0; ry = y_lin[j] - y0_c
        if rx < -5: rx += 10
        elif rx > 5: rx -= 10
        rsq = rx*rx + ry*ry
        rho_exact[i,j] = (1.0 - (GAMMA-1)*b_vort**2/(8*GAMMA*math.pi**2)*math.exp(1-rsq))**(1/(GAMMA-1))
        p_exact[i,j] = rho_exact[i,j]**GAMMA
        u_exact[i,j] = 1.0 - b_vort/(2*math.pi)*math.exp(0.5*(1-rsq))*ry
        v_exact[i,j] = 1.0 + b_vort/(2*math.pi)*math.exp(0.5*(1-rsq))*rx

# ========== 保存数据 ==========
torch.save(actual_model.state_dict(), os.path.join(output_dir, 'model.pt'))
np.savez(os.path.join(output_dir, 'results.npz'),
         x=Xg, y=Yg, t_snapshots=t_snapshots,
         loss_history=np.array(loss_history),
         **{f'{k}_t{t}': v for t, preds in all_preds.items() for k, v in preds.items()})

# ========== 图1: 多时刻演化 (速度大小 |V|) ==========
n_snap = len(t_snapshots)
fig1, axes1 = plt.subplots(1, n_snap, figsize=(5*n_snap, 4.5))
for idx, t_val in enumerate(t_snapshots):
    d = all_preds[t_val]
    vel = np.sqrt(d['u']**2 + d['v']**2)
    ax = axes1[idx]
    c = ax.contourf(Xg, Yg, vel, levels=30, cmap='jet')
    # 矢量场叠加
    skip = 12
    ax.quiver(Xg[::skip, ::skip], Yg[::skip, ::skip],
              d['u'][::skip, ::skip]-1.0, d['v'][::skip, ::skip]-1.0,
              color='k', alpha=0.4, scale=8)
    ax.set_title(f't = {t_val:.2f}', fontsize=13)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(c, ax=ax, label='|V|')
fig1.suptitle(f'Velocity |V| evolution — Re={Re:.0f}, μ={mu_visc:.4f}',
              fontsize=14, fontweight='bold')
fig1.tight_layout()
fig1.savefig(os.path.join(output_dir, 'velocity_evolution.png'), dpi=200, bbox_inches='tight')
plt.close(fig1)

# ========== 图2: 多时刻演化 (密度 ρ) ==========
fig2, axes2 = plt.subplots(1, n_snap, figsize=(5*n_snap, 4.5))
# 固定colorbar范围让时间演化可比
rho_min = min(all_preds[t]['rho'].min() for t in t_snapshots)
rho_max = max(all_preds[t]['rho'].max() for t in t_snapshots)
for idx, t_val in enumerate(t_snapshots):
    ax = axes2[idx]
    c = ax.contourf(Xg, Yg, all_preds[t_val]['rho'], levels=30,
                    cmap='viridis', vmin=rho_min, vmax=rho_max)
    ax.set_title(f't = {t_val:.2f}', fontsize=13)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal')
    plt.colorbar(c, ax=ax, label=r'$\rho$')
fig2.suptitle(r'Density $\rho$ evolution — viscous decay', fontsize=14, fontweight='bold')
fig2.tight_layout()
fig2.savefig(os.path.join(output_dir, 'density_evolution.png'), dpi=200, bbox_inches='tight')
plt.close(fig2)

# ========== 图3: 多时刻演化 (压力 p) ==========
fig3, axes3 = plt.subplots(1, n_snap, figsize=(5*n_snap, 4.5))
p_min = min(all_preds[t]['p'].min() for t in t_snapshots)
p_max = max(all_preds[t]['p'].max() for t in t_snapshots)
for idx, t_val in enumerate(t_snapshots):
    ax = axes3[idx]
    c = ax.contourf(Xg, Yg, all_preds[t_val]['p'], levels=30,
                    cmap='coolwarm', vmin=p_min, vmax=p_max)
    ax.set_title(f't = {t_val:.2f}', fontsize=13)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal')
    plt.colorbar(c, ax=ax, label='p')
fig3.suptitle(r'Pressure $p$ evolution', fontsize=14, fontweight='bold')
fig3.tight_layout()
fig3.savefig(os.path.join(output_dir, 'pressure_evolution.png'), dpi=200, bbox_inches='tight')
plt.close(fig3)

# ========== 图4: 涡量 ω = ∂v/∂x - ∂u/∂y ==========
fig4, axes4 = plt.subplots(1, n_snap, figsize=(5*n_snap, 4.5))
dx = x_lin[1] - x_lin[0]; dy = y_lin[1] - y_lin[0]
omega_max = 0
for t_val in t_snapshots:
    dvdx = np.gradient(all_preds[t_val]['v'], dx, axis=0)
    dudy = np.gradient(all_preds[t_val]['u'], dy, axis=1)
    omega_max = max(omega_max, abs(dvdx - dudy).max())
for idx, t_val in enumerate(t_snapshots):
    dvdx = np.gradient(all_preds[t_val]['v'], dx, axis=0)
    dudy = np.gradient(all_preds[t_val]['u'], dy, axis=1)
    omega = dvdx - dudy
    ax = axes4[idx]
    c = ax.contourf(Xg, Yg, omega, levels=30, cmap='RdBu_r',
                    vmin=-omega_max, vmax=omega_max)
    ax.set_title(f't = {t_val:.2f}', fontsize=13)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal')
    plt.colorbar(c, ax=ax, label=r'$\omega_z$')
fig4.suptitle(r'Vorticity $\omega = \partial v/\partial x - \partial u/\partial y$',
              fontsize=14, fontweight='bold')
fig4.tight_layout()
fig4.savefig(os.path.join(output_dir, 'vorticity_evolution.png'), dpi=200, bbox_inches='tight')
plt.close(fig4)

# ========== 图5: t=0 PINN vs 精确解 对比 ==========
fig5, axes5 = plt.subplots(2, 3, figsize=(18, 10))
d0 = all_preds[0.0]
vel_exact = np.sqrt(u_exact**2 + v_exact**2)
vel_pinn = np.sqrt(d0['u']**2 + d0['v']**2)

for ax, data, title, cmap in [
    (axes5[0,0], vel_exact, 'Exact |V| (t=0)', 'jet'),
    (axes5[0,1], vel_pinn,  'PINN |V| (t=0)', 'jet'),
    (axes5[0,2], abs(vel_pinn - vel_exact), 'Error |V|', 'hot'),
    (axes5[1,0], p_exact,   'Exact p (t=0)', 'coolwarm'),
    (axes5[1,1], d0['p'],   'PINN p (t=0)', 'coolwarm'),
    (axes5[1,2], abs(d0['p'] - p_exact), 'Error p', 'hot'),
]:
    c = ax.contourf(Xg, Yg, data, levels=30, cmap=cmap)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal')
    plt.colorbar(c, ax=ax)
fig5.suptitle('IC Verification: PINN vs Exact at t=0', fontsize=14, fontweight='bold')
fig5.tight_layout()
fig5.savefig(os.path.join(output_dir, 'IC_comparison.png'), dpi=200, bbox_inches='tight')
plt.close(fig5)

# ========== 图6: 中心线剖面 (量化涡旋衰减) ==========
fig6, axes6 = plt.subplots(2, 2, figsize=(12, 10))

# 沿 y=5 的 u 剖面 (不同时刻)
jc = ny // 2  # y=5 对应的 index
ax = axes6[0, 0]
ax.plot(x_lin, u_exact[:, jc], 'k--', lw=2, label='Exact IC')
for t_val in t_snapshots:
    ax.plot(x_lin, all_preds[t_val]['u'][:, jc], lw=1.5, label=f't={t_val}')
ax.set_xlabel('x'); ax.set_ylabel('u')
ax.set_title('u along y=5 (vortex centerline)'); ax.legend(); ax.grid(True, alpha=0.3)

# 沿 x=5 的 v 剖面
ic = nx // 2
ax = axes6[0, 1]
ax.plot(y_lin, v_exact[ic, :], 'k--', lw=2, label='Exact IC')
for t_val in t_snapshots:
    ax.plot(y_lin, all_preds[t_val]['v'][ic, :], lw=1.5, label=f't={t_val}')
ax.set_xlabel('y'); ax.set_ylabel('v')
ax.set_title('v along x=5 (vortex centerline)'); ax.legend(); ax.grid(True, alpha=0.3)

# 沿 y=5 的 ρ 剖面
ax = axes6[1, 0]
ax.plot(x_lin, rho_exact[:, jc], 'k--', lw=2, label='Exact IC')
for t_val in t_snapshots:
    ax.plot(x_lin, all_preds[t_val]['rho'][:, jc], lw=1.5, label=f't={t_val}')
ax.set_xlabel('x'); ax.set_ylabel(r'$\rho$')
ax.set_title(r'$\rho$ along y=5'); ax.legend(); ax.grid(True, alpha=0.3)

# Loss 曲线
ax = axes6[1, 1]
ax.semilogy(loss_history, 'b-', lw=0.5)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('Training Loss'); ax.grid(True, alpha=0.3)

fig6.suptitle(f'2D Viscous Vortex Re={Re:.0f} — Centerline Profiles & Decay',
              fontsize=14, fontweight='bold')
fig6.tight_layout()
fig6.savefig(os.path.join(output_dir, 'centerline_profiles.png'), dpi=200, bbox_inches='tight')
plt.close(fig6)

# ========== 图7: 涡旋强度随时间衰减 ==========
fig7, ax7 = plt.subplots(1, 1, figsize=(8, 5))
t_fine = np.linspace(0, Tend, 50)
vort_strength = []
for t_val in t_fine:
    Tg = np.full_like(Xg, t_val)
    x_test_np = np.stack([Tg, Xg, Yg], axis=-1).reshape(-1, 3)
    x_test = torch.tensor(x_test_np, dtype=dtype).to(device)
    with torch.no_grad():
        pred_t = actual_model(x_test)
    u_t = to_numpy(pred_t[:, 2]).reshape(nx, ny)
    v_t = to_numpy(pred_t[:, 3]).reshape(nx, ny)
    # 涡旋强度: max|V - V_inf|
    du = u_t - 1.0; dv = v_t - 1.0
    vort_strength.append(np.sqrt(du**2 + dv**2).max())

ax7.plot(t_fine, vort_strength, 'b-o', markersize=3, lw=2, label='PINN max|V-V_inf|')
ax7.set_xlabel('t', fontsize=12); ax7.set_ylabel('Vortex strength', fontsize=12)
ax7.set_title(f'Vortex Decay — Re={Re:.0f}', fontsize=14)
ax7.legend(); ax7.grid(True, alpha=0.3)
fig7.tight_layout()
fig7.savefig(os.path.join(output_dir, 'vortex_decay.png'), dpi=200, bbox_inches='tight')
plt.close(fig7)

print(f'所有图已保存到: {output_dir}')
print(f'Training time: {training_time:.1f}s, Final loss: {loss_history[-1]:.4e}')

if interrupted:
    print("\n训练被中断，已保存当前结果。")
else:
    print(f"\n训练完成，结果已保存到: {output_dir}")
