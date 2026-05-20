import torch

def get_guderley_residual(model, xi, gamma=1.4, n=2):
    """
    计算 Guderley 自相似方程的残差
    n=2 代表球对称 (Spherical), n=1 代表柱对称, n=0 代表平面
    """
    # 开启梯度追踪
    xi.requires_grad_(True)
    
    # 1. 获取网络预测
    output = model(xi)
    G = output[:, 0:1] # 无量纲密度
    V = output[:, 1:2] # 无量纲速度
    P = output[:, 2:3] # 无量纲压力
    
    alpha = model.alpha # 当前猜测的 alpha 值
    
    # 2. 计算对 xi 的一阶导数 (自动微分)
    dG = torch.autograd.grad(G, xi, grad_outputs=torch.ones_like(G), create_graph=True)[0]
    dV = torch.autograd.grad(V, xi, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    dP = torch.autograd.grad(P, xi, grad_outputs=torch.ones_like(P), create_graph=True)[0]
    
    # 3. 组装方程 (Guderley Self-Similar Euler Equations)
    # 这些公式推导自把 u(r,t) = R_dot * V(xi) 代入 Euler 方程
    # 【修正】lambda = 1/alpha (不是 (1-alpha)/alpha)
    # 文献中 lambda ≈ 1.394 对应 alpha ≈ 0.717
    lam = 1.0 / alpha

    # --- 方程 1: 连续性方程 (Mass) ---
    # (V - xi) * dG/dxi + G * dV/dxi + n * G * V / xi = 0
    res_mass = (V - xi) * dG + G * dV + (n * G * V) / xi

    # --- 方程 2: 动量方程 (Momentum) ---
    # (V - xi) * dV/dxi + (1/G) * dP/dxi + V * (V - 1) / xi + lam * V = 0
    # 注意：这里我们乘以 G 避免除法，变成: G(V-xi)dV + dP + ...
    res_mom = G * (V - xi) * dV + dP + G * (V * (V - 1) / xi + lam * V)

    # --- 方程 3: 能量/熵方程 (Energy/Entropy) ---
    # (V - xi) * (dP/dxi - gamma * P/G * dG/dxi) + 2 * P * (V - 1) / xi + 2 * lam * P = 0
    # 同样乘以 G 避免除法
    res_eng = (V - xi) * (dP * G - gamma * P * dG) + G * (2 * P * (V - 1) / xi + 2 * lam * P)

    # 4. 汇总 Loss
    # 我们把三个残差的平方和作为 PDE Loss
    loss_pde = torch.mean(res_mass**2) + torch.mean(res_mom**2) + torch.mean(res_eng**2)
    
    return loss_pde

def get_boundary_loss(model, gamma=1.4):
    """
    强激波边界条件 (Rankine-Hugoniot) 在 xi = 1 处
    """
    # 构造边界点 xi = 1
    xi_bc = torch.tensor([[1.0]], requires_grad=True, device=next(model.parameters()).device)
    
    output = model(xi_bc)
    G_pred, V_pred, P_pred = output[:, 0], output[:, 1], output[:, 2]
    
    # 强激波极限下的理论值 (常数)
    G_true = (gamma + 1) / (gamma - 1)
    V_true = 2 / (gamma + 1)
    P_true = 2 / (gamma + 1)
    
    loss_bc = (G_pred - G_true)**2 + (V_pred - V_true)**2 + (P_pred - P_true)**2
    return loss_bc