"""
傅里叶特征映射模块 (Fourier Feature Mapping)

用于帮助神经网络捕捉高频信息（如激波、间断等）

参考文献：
Tancik et al. "Fourier Features Let Networks Learn High Frequency Functions
in Low Dimensional Domains" (NeurIPS 2020)

作者：PINN_WE 项目组
日期：2026-01-19
"""

import torch
import torch.nn as nn
import numpy as np


class FourierFeatureLayer(nn.Module):
    """傅里叶特征映射层

    将低维输入 (t, x) 映射到高维频域特征，帮助神经网络捕捉高频信息（如激波）

    核心思想：
        标准 MLP 存在频谱偏差（spectral bias），倾向于学习低频函数。
        通过在输入层后添加傅里叶变换，显式引入高频基函数 sin/cos，
        使网络能够更好地表示激波等高频特征。

    数学原理：
        输入: x ∈ R^d
        映射: γ(x) = [sin(2πB^T x), cos(2πB^T x)]
        其中 B ∈ R^(d×m) 是随机频率矩阵，B_ij ~ N(0, σ²)
        输出维度: 2m

    参数：
        input_dim (int): 输入维度
            - 1D 问题: 2 (t, x)
            - 2D 问题: 3 (t, x, y)

        mapping_size (int): 频域特征维度，输出维度 = 2 * mapping_size
            - 推荐值: 64-256
            - 越大表达能力越强，但计算量也越大
            - 默认: 128 (输出 256 维)

        scale (float): 频率尺度 σ，控制捕捉的频率范围
            - scale=1: 低频（光滑函数）
            - scale=10: 中频（一般激波）
            - scale=30: 高频（强激波、间断）
            - 默认: 10.0
            - 建议根据问题调参：先从 10 开始，如果激波捕捉不好就增大

        learnable (bool): 是否让频率矩阵 B 可学习
            - False (推荐): 固定随机频率，计算快，泛化好
            - True: 可学习频率，表达能力更强，但可能过拟合
            - 默认: False

    输入输出：
        输入: (batch_size, input_dim)
            例如: (10000, 2) 表示 10000 个 (t, x) 点

        输出: (batch_size, 2*mapping_size)
            例如: (10000, 256) 表示 256 维频域特征

    使用示例：
        >>> # 创建傅里叶层
        >>> fourier_layer = FourierFeatureLayer(input_dim=2, mapping_size=128, scale=10.0)
        >>>
        >>> # 前向传播
        >>> x = torch.randn(1000, 2)  # 1000 个 (t, x) 点
        >>> features = fourier_layer(x)  # (1000, 256)
        >>>
        >>> # 在网络中使用
        >>> net = nn.Sequential(
        >>>     fourier_layer,
        >>>     nn.Linear(256, 60),
        >>>     nn.Tanh(),
        >>>     nn.Linear(60, 3)
        >>> )
    """

    def __init__(self, input_dim=2, mapping_size=128, scale=10.0, learnable=False):
        super(FourierFeatureLayer, self).__init__()

        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = scale
        self.learnable = learnable

        # 初始化随机频率矩阵 B ~ N(0, scale²)
        # 形状: (input_dim, mapping_size)
        B = torch.randn(input_dim, mapping_size) * scale

        if learnable:
            # 可学习参数
            self.B = nn.Parameter(B)
        else:
            # 固定参数（推荐）
            self.register_buffer('B', B)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状 (batch_size, input_dim)

        Returns:
            features: 傅里叶特征，形状 (batch_size, 2*mapping_size)

        数学公式：
            x_proj = 2π * x @ B
            output = [sin(x_proj), cos(x_proj)]

        这样可以表示任意频率的周期函数组合
        """
        # x: (batch_size, input_dim)
        # B: (input_dim, mapping_size)
        # x_proj: (batch_size, mapping_size)
        x_proj = 2.0 * np.pi * x @ self.B

        # 返回 [sin, cos] 拼接
        # sin: (batch_size, mapping_size)
        # cos: (batch_size, mapping_size)
        # output: (batch_size, 2*mapping_size)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def extra_repr(self):
        """打印层信息（用于 print(model)）"""
        return (f'input_dim={self.input_dim}, '
                f'mapping_size={self.mapping_size}, '
                f'scale={self.scale}, '
                f'learnable={self.learnable}, '
                f'output_dim={2*self.mapping_size}')


def create_fourier_feature_layer(input_dim=2, mapping_size=128, scale=10.0, learnable=False):
    """
    工厂函数：创建傅里叶特征层

    这是一个便捷函数，用于快速创建 FourierFeatureLayer 实例

    Args:
        input_dim (int): 输入维度，默认 2 (t, x)
        mapping_size (int): 频域特征维度，默认 128
        scale (float): 频率尺度，默认 10.0
        learnable (bool): 是否可学习，默认 False

    Returns:
        FourierFeatureLayer: 傅里叶特征层实例

    使用示例：
        >>> # 方式1: 直接创建
        >>> fourier_layer = create_fourier_feature_layer(
        >>>     input_dim=2,
        >>>     mapping_size=128,
        >>>     scale=10.0
        >>> )
        >>>
        >>> # 方式2: 使用默认参数
        >>> fourier_layer = create_fourier_feature_layer()
    """
    return FourierFeatureLayer(
        input_dim=input_dim,
        mapping_size=mapping_size,
        scale=scale,
        learnable=learnable
    )


# ========== 调参建议 ==========
"""
根据不同问题类型，推荐的参数配置：

1. 光滑问题（无激波）：
   mapping_size=64, scale=1.0

2. 弱激波问题（如 Sod 激波管）：
   mapping_size=128, scale=10.0  ← 默认配置

3. 强激波问题（如 Blast Wave, Shu-Osher）：
   mapping_size=256, scale=20.0

4. 极强间断问题（如接触间断）：
   mapping_size=256, scale=30.0

调参策略：
- 如果训练损失下降慢 → 增大 mapping_size
- 如果激波捕捉不清晰 → 增大 scale
- 如果过拟合 → 减小 mapping_size 或 scale
- 如果计算太慢 → 减小 mapping_size

注意事项：
- scale 太大（>50）可能导致梯度爆炸
- mapping_size 太大（>512）计算开销显著增加
- 建议先用默认参数，再根据结果调整
"""
