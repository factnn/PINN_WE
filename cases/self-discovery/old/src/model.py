"""
Guderley PINN Model
定义神经网络结构，包含可训练的物理参数 alpha
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GuderleyPINN(nn.Module):
    """
    Guderley 激波内爆问题的物理信息神经网络

    输入: xi (自相似坐标, 标量)
    输出: V (无量纲速度), C (无量纲声速)
    可训练参数: alpha (自相似指数)
    """

    def __init__(
        self,
        hidden_layers: int = 4,
        hidden_neurons: int = 64,
        alpha_init: float = 0.8,
        alpha_min: float = 0.5,
        alpha_max: float = 1.0
    ):
        """
        初始化网络

        Args:
            hidden_layers: 隐藏层数量
            hidden_neurons: 每层神经元数量
            alpha_init: alpha 的初始猜测值 (真实值约 0.717)
            alpha_min: alpha 的最小值 (用于约束)
            alpha_max: alpha 的最大值 (用于约束)
        """
        super(GuderleyPINN, self).__init__()

        # 构建神经网络 (输入 1D -> 输出 2D: V, C)
        layers = []
        layers.append(nn.Linear(1, hidden_neurons))
        layers.append(nn.Tanh())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_neurons, 2))  # 输出 V, C（移除冗余的G）

        self.net = nn.Sequential(*layers)

        # 【核心】可训练的物理参数 alpha
        # 使用 logit 空间存储，通过 sigmoid 映射到 [alpha_min, alpha_max]
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # 计算初始值对应的 logit
        # sigmoid(x) = (alpha_init - alpha_min) / (alpha_max - alpha_min)
        # x = logit(sigmoid_value)
        sigmoid_value = (alpha_init - alpha_min) / (alpha_max - alpha_min)
        sigmoid_value = torch.clamp(torch.tensor(sigmoid_value), 0.01, 0.99)  # 避免极值
        logit_value = torch.log(sigmoid_value / (1 - sigmoid_value))

        self.raw_alpha = nn.Parameter(logit_value)

        # 初始化网络权重
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def alpha(self) -> torch.Tensor:
        """
        返回约束后的 alpha 值
        使用 sigmoid 映射到 [alpha_min, alpha_max] 区间
        """
        sigmoid_val = torch.sigmoid(self.raw_alpha)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * sigmoid_val

    def forward(self, xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播 - 软约束版本，仅输出V和C

        Args:
            xi: 自相似坐标 [batch_size, 1]

        Returns:
            V: 无量纲速度 [batch_size, 1]
            C: 无量纲声速 [batch_size, 1]
        """
        output = self.net(xi)
        V = output[:, 0:1]
        C = output[:, 1:2]

        return V, C

    def get_alpha_value(self) -> float:
        """返回当前 alpha 的数值（用于监控）"""
        return self.alpha.item()


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("测试 GuderleyPINN 模型")
    print("=" * 60)

    model = GuderleyPINN(hidden_layers=3, hidden_neurons=64, alpha_init=0.8)

    print(f"\n初始 alpha 值: {model.get_alpha_value():.6f}")
    print(f"网络结构:\n{model.net}")

    # 测试前向传播
    xi_test = torch.linspace(0.1, 1.0, 10).reshape(-1, 1)
    V, C = model(xi_test)

    print(f"\n输入 xi 形状: {xi_test.shape}")
    print(f"输出 V 形状: {V.shape}")
    print(f"输出 C 形状: {C.shape}")

    print(f"\nV 的范围: [{V.min().item():.4f}, {V.max().item():.4f}]")
    print(f"C 的范围: [{C.min().item():.4f}, {C.max().item():.4f}]")

    print("\n✓ 模型测试通过（已移除冗余G变量）")
