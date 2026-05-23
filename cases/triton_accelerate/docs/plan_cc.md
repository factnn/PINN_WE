# Triton-Accelerated PINN: 执行计划

## 核心思路

用 Triton fused stencil kernel 替换 CAN-PINN 中的有限差分 PDE 残差计算，
覆盖 1D/2D/3D，对比 PyTorch 原生实现，展示加速比和显存节省。

**不碰 autograd**。网络输出物理场张量，差分残差全部在 Triton kernel 里算。

---

## 算例矩阵

| 维度 | 方程 | 网络 | 说明 |
|------|------|------|------|
| 1D | Burgers 方程 | CAN-PINN (MLP) | 最简单，验证 kernel 正确性 |
| 1D | Burgers 方程 | Phy-CNN (Conv1D) | 对比 MLP vs CNN 加速差异 |
| 2D | Allen-Cahn 方程 | CAN-PINN (MLP) | 高阶导数（四阶），加速更明显 |
| 2D | Allen-Cahn 方程 | Phy-CNN (Conv2D) | 证明算子与网络解耦 |
| 3D | 3D Taylor-Green Vortex | CAN-PINN (MLP) | 显存瓶颈最明显 |
| 3D | 3D Taylor-Green Vortex | Phy-CNN (Conv3D) | 打破 conv3d 显存瓶颈 |

**两个网络**：
- **CAN-PINN**：MLP 输出网格点上的物理场，差分算残差
- **Phy-CNN**：卷积网络输出物理场，差分算残差（与 conv3d 对比更有说服力）

**两个算例**：
- **Burgers 方程**：经典，二阶差分，baseline 简单
- **Allen-Cahn 方程**：四阶导数，原生 PyTorch 显存爆炸，Triton 优势最大

---

## 对比指标

每个算例记录：
- **单次 PDE 残差计算时间**（ms）：PyTorch 切片 vs Triton kernel
- **显存峰值**（GB）：PyTorch vs Triton
- **最大可训分辨率**：40GB 显存下各自能跑多大网格
- **端到端训练时间**：达到相同精度（L2 < 1e-3）的 wall-clock time

为了让你的论文在盲审专家眼中达到顶会（SysML/MLSys）级别的严谨度，你必须建立一套“多维立体”的评估指标体系。做底层系统和算子加速，最忌讳只给出一个“我比原来快了 X 倍”的干瘪数据。你需要用不同的指标去回答评委脑子里的不同疑问。针对你手里的 4 张 A100 (40GB) 以及 1D/2D/3D 的算例矩阵，你到底需要记录哪些具体指标？我为你分成了三大核心模块：一、 算子级微观指标 (Micro-benchmark)这一组指标完全脱离网络训练，只拿一个固定尺寸的张量（比如 $128 \times 128 \times 128$），跑一次你写的 Triton 残差函数。核心目的是“秀底层肌肉”。单次前向延迟 (Kernel Latency)：单位：毫秒 (ms) 或 微秒 ($\mu$s)。测什么：纯算子在 GPU 上的执行时间。避坑指南：绝对不能简单用 Python 的 time.time() 包裹，必须使用 CUDA Event 计时（如 triton.testing.do_bench），以剔除 CPU 异步调度的开销。显存峰值占用 (Peak Memory Allocated)：单位：兆字节 (MB) 或 吉字节 (GB)。测什么：算子在计算过程中瞬间吃掉的最大显存。叙事价值：这是你的核心杀手锏！你要证明 PyTorch 切片会产生海量中间张量导致 40GB 显存溢出，而你的 Triton 算子通过 SRAM 缓存复用，将峰值压到了原来的十分之一。显存带宽利用率 (Memory Bandwidth Utilization)：单位：GB/s 或占理论峰值的百分比 (%)。测什么：A100 (40GB PCIe) 的理论极限带宽大概是 1555 GB/s。你要记录你的算子跑到了多少。叙事价值：如果你的算子跑到了 1200 GB/s (达到峰值的 80% 左右)，这就证明你的算子已经达到了光速极限 (Speed of Light)，任何后续的优化都无法再显著提升性能，因为这在物理上已经是一个完美的 Memory Bound 算子。数值对齐误差 (Numerical Error)：单位：相对误差范围（如 $< 10^{-5}$）。测什么：你的算子输出与 PyTorch 原生 FP64 计算结果的差值（L2 范数或最大绝对误差）。证明你不仅快，而且绝对正确。二、 端到端宏观指标 (Macro-benchmark)这一组指标是将你的算子塞进 CAN-PINN、PhyCNN 或 PINO 之后，跑真实的物理训练。核心目的是证明“对最终用户的实际业务收益”。单轮迭代耗时 (Time per Epoch / Step Time)：单位：秒 (s) 或 毫秒 (ms)。测什么：包含网络前向推理、你的算子计算 PDE Loss、以及自动微分反向更新参数的整体时间。单卡极限网格分辨率 (Max Grid Resolution per GPU)：单位：网格点数量（如 $64^3$, $128^3$）。叙事价值：用极其直观的数字告诉评委：“以前一张 40G 的卡只能训 $64^3$，再大就 OOM；用了我的系统，单卡直接训 $256^3$”。这叫打破维度壁垒。达标收敛时间 (Time-to-Solution)：单位：挂钟时间 (Wall-clock time，如“3.5 小时”)。测什么：让原版和 Triton 版都跑，看谁先让 PDE Loss 降到某一个绝对阈值（如 $10^{-4}$）。这能证明你的算子不仅单步快，而且能切实缩短科研人员等待结果的时间。

---

## 代码结构

```
cases/triton_accelerate/
├── baseline/
│   ├── burgers_1d.py       # 1D Burgers PyTorch baseline
│   ├── burgers_2d.py       # 2D Burgers PyTorch baseline
│   └── tgv_3d.py           # 3D TGV PyTorch baseline
├── kernels/
│   ├── stencil_1d.py       # 1D Triton stencil kernel
│   ├── stencil_2d.py       # 2D Triton stencil kernel (halo exchange)
│   └── stencil_3d.py       # 3D Triton stencil kernel (halo exchange)
├── pinn/
│   ├── can_pinn_1d.py      # 1D CAN-PINN with Triton kernel
│   ├── can_pinn_2d.py      # 2D CAN-PINN with Triton kernel
│   └── can_pinn_3d.py      # 3D CAN-PINN with Triton kernel
└── benchmark.py            # 统一 benchmark，输出对比表格
```

---

## 执行顺序

抓取火焰图：使用 PyTorch Profiler 或 Nsight Systems 跑一次 3D 原生差分，截取内存读写（Memory Bound）占据绝大部分时间的性能分析图，这是你论文的立论基础。

### Phase 1：1D 打通（1周）
目标：跑通完整流程，验证 Triton kernel 数值正确性。

1. 写 `baseline/burgers_1d.py`：PyTorch 切片差分 + MLP，记录时间和显存
2. 写 `kernels/stencil_1d.py`：1D Triton kernel，输入场张量，输出残差
3. 数值对齐：Triton 输出 vs PyTorch 输出，误差 < 1e-5
4. 写 `pinn/can_pinn_1d.py`：把 Triton kernel 插入训练循环
5. 记录加速比

### Phase 2：2D 核心（1-2周）
目标：实现 halo exchange，这是 Triton 加速的核心难点。

1. 写 `baseline/burgers_2d.py`
2. 写 `kernels/stencil_2d.py`：Block 划分 + shared memory + halo exchange
3. 数值对齐 + 性能测试
4. 写 `pinn/can_pinn_2d.py`

### Phase 3：3D 扩展（1-2周）
目标：3D kernel，展示显存节省最明显的场景。

1. 写 `baseline/tgv_3d.py`
2. 写 `kernels/stencil_3d.py`：3D halo exchange
3. 测试最大可训分辨率（原生 OOM 的临界点 vs Triton 能跑多大）
4. 写 `pinn/can_pinn_3d.py`

### Phase 4：收网（1周）
1. 写 `benchmark.py`：统一跑所有算例，输出对比表格和图
2. 用 PyTorch Profiler 截图证明原生实现是 memory bound
3. 整理论文图表

---

## 关键技术点

**Halo Exchange**：2D/3D kernel 中，每个 Block 需要加载相邻 Block 的边界数据到 shared memory。
这是最容易出 bug 的地方，1D 先打通逻辑，再升维。

**数值精度**：差分格式用二阶中心差分，与 PyTorch baseline 保持一致，确保对比公平。

**接口设计**：
```python
# 统一接口：接收网络输出张量，返回 PDE 残差标量 loss
loss = stencil_kernel(u_pred, dx, equation="burgers")
```

---

## 预期结果

| 维度 | 原生时间 | Triton 时间 | 加速比 | 显存节省 |
|------|---------|------------|--------|---------|
| 1D | ~5ms | ~1ms | ~5x | 小 |
| 2D | ~50ms | ~8ms | ~6x | 2-3x |
| 3D | ~500ms | ~60ms | ~8x | 3-5x |

（以上为预估，实测后更新）
