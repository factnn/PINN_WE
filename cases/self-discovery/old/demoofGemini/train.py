import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from model import DiscoveryPINN
from physics import get_guderley_residual, get_boundary_loss

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS_ADAM = 5000
EPOCHS_LBFGS = 2000
LR = 1e-3

def train():
    # 1. 初始化模型
    model = DiscoveryPINN().to(device)
    print(f"Initial alpha guess: {model.alpha.item():.5f}")
    
    # 2. 准备训练数据
    # Guderley 问题通常求解范围是激波后方 [1.0] 到 声速点附近 [0.5]
    # 我们避开 0 (圆心奇点)，采样范围 [0.4, 1.0]
    xi_collocation = torch.linspace(0.4, 1.0, 1000).view(-1, 1).to(device)
    
    # 3. 优化器 1: Adam (快速粗调)
    optimizer_adam = optim.Adam(model.parameters(), lr=LR)
    
    print("\n--- Phase 1: Adam Optimization ---")
    alpha_history = []
    
    for epoch in range(EPOCHS_ADAM):
        optimizer_adam.zero_grad()
        
        # 随机采样一些点以防过拟合 (可选)
        idx = torch.randperm(xi_collocation.size(0))[:500]
        xi_batch = xi_collocation[idx]
        
        loss_pde = get_guderley_residual(model, xi_batch)
        loss_bc = get_boundary_loss(model)
        
        # 总损失：物理残差 + 边界条件
        loss = loss_pde + 10.0 * loss_bc # 加权 BC 保证边界对齐
        
        loss.backward()
        optimizer_adam.step()
        
        curr_alpha = model.alpha.item()
        alpha_history.append(curr_alpha)
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss.item():.6e} | Alpha: {curr_alpha:.5f}")

    # 4. 优化器 2: L-BFGS (高精度精调 - 关键步骤)
    # 很多物理常数发现必须靠二阶优化器才能“落坑”
    print("\n--- Phase 2: L-BFGS Optimization ---")
    optimizer_lbfgs = optim.LBFGS(model.parameters(), 
                                  lr=1.0, 
                                  max_iter=20, 
                                  history_size=100)
    
    def closure():
        optimizer_lbfgs.zero_grad()
        loss_pde = get_guderley_residual(model, xi_collocation) # LBFGS 建议用全量数据
        loss_bc = get_boundary_loss(model)
        loss = loss_pde + 10.0 * loss_bc
        loss.backward()
        return loss

    for epoch in range(EPOCHS_LBFGS):
        loss = optimizer_lbfgs.step(closure)
        curr_alpha = model.alpha.item()
        alpha_history.append(curr_alpha)
        
        if epoch % 100 == 0:
             print(f"Step {epoch:5d} | Loss: {loss.item():.6e} | Alpha: {curr_alpha:.5f}")

    print(f"\nFinal Discovered Alpha: {model.alpha.item():.6f}")
    print(f"Theoretical Alpha (gamma=1.4, spherical): ~0.717")

    # 5. 画图：Alpha 的收敛过程
    plt.figure()
    plt.plot(alpha_history)
    plt.axhline(y=0.717, color='r', linestyle='--', label='Theory (0.717)')
    plt.xlabel('Iterations')
    plt.ylabel('Discovered Alpha')
    plt.title('Self-Discovery of Guderley Exponent')
    plt.legend()
    plt.savefig('alpha_convergence.png')
    plt.show()

if __name__ == "__main__":
    train()