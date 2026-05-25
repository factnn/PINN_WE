# 技术详解：我们怎么做的加速？

## 一、我们加速了哪个环节？

回顾一次训练步：

```
┌─────────────────────────────────────────────────────┐
│ ① 网络 Forward        │  PyTorch 原生（我们没动）    │
├─────────────────────────────────────────────────────┤
│ ② PDE 残差计算 (Forward) │  ← 我们的 Forward Kernel  │ ★
├─────────────────────────────────────────────────────┤
│ ③ Loss = mean(res²)   │  PyTorch 原生（一行代码）    │
├─────────────────────────────────────────────────────┤
│ ④ PDE 梯度回传 (Backward) │ ← 我们的 Backward Kernel │ ★
├─────────────────────────────────────────────────────┤
│ ⑤ 网络参数梯度回传    │  PyTorch 原生（我们没动）    │
├─────────────────────────────────────────────────────┤
│ ⑥ Optimizer 更新参数  │  PyTorch 原生（我们没动）    │
└─────────────────────────────────────────────────────┘
```

我们只动了 **② 和 ④**——PDE 残差的前向计算和反向梯度传播。这两步在 PyTorch 原生实现中占了训练步 50-80% 的时间。

---

## 二、PyTorch 原生实现长什么样？

以 2D 稳态 NS 为例，PyTorch FD 计算 PDE 残差：

```python
# ② PDE 残差计算（PyTorch 原生，约 20 行）
u_x  = (U[2:, 1:-1] - U[:-2, 1:-1]) / (2*dx)    # 操作 1: 减法 + 除法
u_y  = (U[1:-1, 2:] - U[1:-1, :-2]) / (2*dy)    # 操作 2
u_xx = (U[2:, 1:-1] - 2*U[1:-1,1:-1] + U[:-2, 1:-1]) / dx**2  # 操作 3
u_yy = (U[1:-1, 2:] - 2*U[1:-1,1:-1] + U[1:-1, :-2]) / dy**2  # 操作 4
v_x  = (V[2:, 1:-1] - V[:-2, 1:-1]) / (2*dx)    # 操作 5
v_y  = (V[1:-1, 2:] - V[1:-1, :-2]) / (2*dy)    # 操作 6
v_xx = ...                                         # 操作 7
v_yy = ...                                         # 操作 8
p_x  = (P[2:, 1:-1] - P[:-2, 1:-1]) / (2*dx)    # 操作 9
p_y  = (P[1:-1, 2:] - P[1:-1, :-2]) / (2*dy)    # 操作 10

uc = U[1:-1, 1:-1]; vc = V[1:-1, 1:-1]
res_u = uc*u_x + vc*u_y + p_x - nu*(u_xx + u_yy)  # 操作 11-15
res_v = uc*v_x + vc*v_y + p_y - nu*(v_xx + v_yy)  # 操作 16-20
res_div = u_x + v_y                                 # 操作 21
```

然后 `loss.backward()` 时，PyTorch autograd 对上述每个操作逆序执行反向传播——又是 20+ 个独立的 GPU kernel。

---

## 三、加速来源分析

### 来源 1：消除 Kernel Launch Overhead（启动开销）

**问题**：每执行一个 PyTorch 操作（如 `U[2:] - U[:-2]`），CPU 需要向 GPU 发送一次指令。这个"发送"本身有 ~5μs 的固定延迟。

```
PyTorch:  [launch 5μs][compute 1μs][launch 5μs][compute 1μs] × 20 = 120μs
Triton:   [launch 5μs][compute 20μs]                          =  25μs
```

20 次启动 → 1 次启动。**省了 19 次 × 5μs = 95μs 的纯开销**。

在小网格上（compute 时间很短），这个开销占主导。这就是为什么 sod_2d (200×50) 几乎没有加速（1.04x）——计算本身太快，launch overhead 反而是 Triton 的劣势（Triton kernel 比 PyTorch built-in kernel 启动稍慢）。

在大网格上，compute 时间远大于 launch 时间，一次 launch 的开销可以忽略 → 加速明显。

---

### 来源 2：减少显存带宽消耗（Memory Bandwidth）

**问题**：GPU 的计算速度远快于内存读取速度。A100 有 19.5 TFLOPS 算力，但只有 1.5 TB/s 内存带宽。一个 float32 乘法需要读 8 bytes（两个操作数），算 1 FLOP → **带宽是瓶颈，不是算力**。

PyTorch 的实现方式：

```
操作 1: 从显存读 U 的全部值 → 算 u_x → 写结果到显存
操作 2: 从显存读 U 的全部值 → 算 u_y → 写结果到显存（U 被重复读了！）
操作 3: 从显存读 U 的全部值 → 算 u_xx → 写结果到显存（U 又被读了！）
```

U 被读了至少 4 次（u_x, u_y, u_xx, u_yy 各读一次），V 读 4 次，P 读 2 次。

Triton 的实现方式：

```
对每个点 (i,j):
  一次性读取 U[i,j] 和 4 个邻居 → 在寄存器里算完 u_x, u_y, u_xx, u_yy
  一次性读取 V[i,j] 和 4 个邻居 → 算完 v_x, v_y, v_xx, v_yy
  一次性读取 P 的 2 个邻居 → 算完 p_x, p_y
  组装 res_u, res_v, res_div → 写回
```

**每个数据只从显存读一次**。对于 256³ = 1600 万点的 3D 网格：

| | PyTorch FD | Triton | 节省 |
|---|---|---|---|
| U 从显存读取次数 | ~6 次 | **1 次** | 6x |
| 写回中间结果次数 | ~20 次 | **1 次** | 20x |
| 总显存流量 | ~2.5 GB | ~0.4 GB | 6x |

这就是为什么**大网格加速更明显**——大网格下显存带宽成为瓶颈，而 Triton 减少了 6x 的带宽消耗。

---

### 来源 3：消除中间 Tensor（零额外显存）

**问题**：PyTorch autograd 的工作方式是"前向时记录，反向时回放"。每个中间结果都要保存：

```python
u_x = (U[2:] - U[:-2]) / (2*dx)   # PyTorch 保存了 U[2:], U[:-2], u_x 三个 tensor
u_xx = (U[2:] - 2*U[1:-1] + U[:-2]) / dx**2   # 又保存了 3 个 tensor
# ... 20 个操作 × 每个保存 2-3 个中间 tensor
```

以 ldc_3d (48³) 为例：
- 每个 tensor: 48³ × 4 bytes = 442 KB
- 20 个操作 × 2-3 个中间 tensor ≈ 50 个 tensor × 442 KB = **22 MB 额外开销**

对于 vanilla autograd（还要存二阶导的图）→ **17.6 GB**！

Triton 的 autograd.Function 方式：
- Forward：计算残差，只保存 `U, V`（和残差本身）用于 backward
- Backward：手写 adjoint，不需要任何中间 tensor

**显存对比**：

| ldc_3d (48³) | 显存 |
|---|---|
| Vanilla autograd | 17.6 GB |
| PyTorch FD + autograd | ~0.5 GB |
| **Triton (手写 adjoint)** | **0.475 GB** |

---

### 来源 4：Autotune 自动选最优配置

每个 Triton kernel 都有一个 BLOCK 参数（每个线程块处理多少个点）。最优值取决于：
- 网格大小（点太少 → 小 BLOCK，点多 → 大 BLOCK）
- GPU 型号（A100 有 108 SM，RTX 4090 有 128 SM）
- 寄存器压力（backward kernel 变量多 → 小 BLOCK 以提高 occupancy）

我们用 `@triton.autotune` 自动测试多种配置：

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 64}),
        triton.Config({'BLOCK': 128}),
        triton.Config({'BLOCK': 256}),
        triton.Config({'BLOCK': 512}),
    ],
    key=['Nx', 'Ny'],  # 根据网格大小选
)
@triton.jit
def ns_fwd_kernel(..., BLOCK: tl.constexpr):
    ...
```

第一次运行时 Triton 自动 benchmark 所有配置，选最快的缓存起来。之后每次调用直接用最优配置。

---

## 四、加速来源总结

| 加速来源 | 机制 | 收益估计 | 适用场景 |
|----------|------|---------|---------|
| Kernel Launch 消除 | 20 次 → 1 次 | ~20% | 小网格 |
| 显存带宽节省 | 数据只读一次 | ~50% | **大网格（主要来源）** |
| 中间 Tensor 消除 | 手写 adjoint | 显存 6-37x | 3D / 高阶方程 |
| Autotune | 自动选 BLOCK | ~10% | 所有场景 |

**核心洞察**：在 PINN 训练中，PDE 残差计算是一个 **memory-bound** 的 stencil 操作。PyTorch 的逐操作执行模式导致同一数据被反复读写。Triton 通过算子融合，把多次内存访问压缩为一次 → 直接解锁了 GPU 内存带宽上限。

---

## 五、为什么 Forward 加速比 > Backward 加速比？

从实验数据看：

| Case | Forward 加速 | Backward 加速 |
|------|:-:|:-:|
| ldc_2d | 2.64x | 1.74x |
| ldc_3d | 3.10x | 1.79x |
| tgv_2d | 2.33x | 2.05x |

Forward 始终比 Backward 加速更多。原因：

**Forward kernel**：每个点读 5-7 个邻居（stencil），做简单算术 → 纯 memory-bound，融合收益最大。

**Backward kernel**：每个点需要读"自己的邻居的上游梯度"（即邻居的邻居）→ 访问范围更大，cache 命中率下降。而且 adjoint 中有非线性项（$u \cdot u_x$ 的 adjoint 需要加载 u 的邻居值），变量更多 → 寄存器压力高 → occupancy 下降。

简单说：**Forward 是 5 点 stencil，Backward 是"5 点 stencil 的伴随" = 需要访问邻居的邻居 → 更宽的内存访问模式 → 加速受限**。

---

## 六、什么情况下加速不明显？

| 情况 | 原因 | 例子 |
|------|------|------|
| 网格太小 (< 1 万点) | Launch overhead 占主导 | sod_2d (200×50) |
| 1D 问题 | Stencil 只有 3 个点，融合收益小 | burgers_1d_steady (256 点) |
| CNN 组 | CNN forward 本身很快，PDE kernel 不是瓶颈 | cnn_triton vs cnn_canpinn 只有 1.1-1.6x |
| 网络很大 | 网络 forward/backward 时间占主导，kernel 加速被稀释 | — |

**最佳场景**：大网格 + 小网络 + 2D/3D 高阶 PDE = 加速 10-27x。
