import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscoveryPINN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. 骨干网络
        # 输入: xi (自相似坐标)
        # 输出: [G, V, P] (无量纲的密度、速度、压力)
        # 这里的 V, P, G 都是无量纲化后的变量
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 3) 
        )
        
        # 2. 待发现的物理参数: alpha
        # 我们初始化为 1.0 (或 0.8)，真实值约为 0.717
        # 使用 Parameter 包装，使其可被 optimizer 更新
        self._alpha_param = nn.Parameter(torch.tensor([0.85])) 

    @property
    def alpha(self):
        # 物理约束：指数 alpha 必须 > 0.5 且 < 1.0 (通常情况)
        # 使用 sigmoid 把它映射到 (0.5, 1.0) 之间，防止训练初期跑飞到负数
        return torch.sigmoid(self._alpha_param) * 0.5 + 0.5

    def forward(self, xi):
        return self.net(xi)