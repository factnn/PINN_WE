# PhysNN-Triton：从问题到解决方案的完整故事

## 第一章：我们在解决什么问题？

### 1.1 一个流体力学的例子

想象你在设计一个飞机机翼。你需要知道空气流过机翼时的速度场 $(u, v)$ 和压力场 $p$ 是怎么分布的。这些物理量满足 Navier-Stokes 方程——一组偏微分方程（PDE）。

传统做法是用 CFD（计算流体力学）软件在超细网格上做有限元/有限体积模拟，可能需要百万网格 + 超算跑几天。

**PINN（Physics-Informed Neural Network）的想法**：用一个神经网络"学会"这个流场。网络输入坐标 $(x, y, t)$，输出物理量 $(u, v, p)$。训练时不需要标注数据——只需要让网络的输出满足物理方程就行。

---

### 1.2 为什么这行得通？

一个经过训练的 PINN 输出的 $(u, v, p)$ 满足两个条件：
1. **满足 PDE**：在计算域内部，$u, v, p$ 之间的关系符合 Navier-Stokes 方程
2. **满足边界/初始条件**：在边界上速度为 0（壁面），或者初始时刻和已知场匹配

如果这两个条件都满足，那网络输出就是 PDE 的解。

---

## 第二章：PINN 训练的完整流程

以 2D 稳态腔体流（Lid-Driven Cavity）为例：一个正方形盒子，顶面以速度 1 向右滑动，其余三面静止。

### Step 1: 构建网格

在 $[0,1] \times [0,1]$ 的正方形内铺一个均匀网格，比如 $64 \times 64 = 4096$ 个点：

```
(0,1) ──────── (1,1)    ← 顶面: u=1, v=0
  |              |
  |   64×64     |
  |   网格点    |
  |              |
(0,0) ──────── (1,0)    ← 底面: u=0, v=0
```

每个网格点有坐标 $(x_i, y_j)$，间距 $dx = dy = 1/63$。

### Step 2: 前向传播——网络猜一个解

把所有 4096 个坐标点送进网络：

```
输入:  (x, y)         → 4096 个二维坐标
       ↓
     神经网络 (MLP 或 CNN)
       ↓
输出:  (u, v, p)      → 4096 个三元组
       ↓ reshape
     U[64, 64]        → 速度场 u 在网格上的值
     V[64, 64]        → 速度场 v 在网格上的值
     P[64, 64]        → 压力场 p 在网格上的值
```

**这一步在做什么？** 网络就像一个"万能拟合器"——给定坐标，猜测该点的速度和压力。刚开始训练时，这个猜测是随机的（完全不对）。训练的目标是让猜测越来越接近真实物理。

### Step 3: 计算 PDE 残差——检查物理方程满足得如何

我们有了 $U, V, P$ 在网格上的值。现在要检查：这些值满足物理方程吗？

Navier-Stokes 方程说：

$$\text{res}_u = u \cdot u_x + v \cdot u_y + p_x - \nu(u_{xx} + u_{yy}) = 0$$

如果网络输出的 $(u, v, p)$ 是正确解，那等式左边（残差 $\text{res}_u$）在每个点上应该为 0。

**怎么算偏导数 $u_x, u_{xx}$ 等？** 这里有两种方法：

#### 方法 A: Autograd（解析求导）

利用 PyTorch 的自动微分，对网络直接求导：

```python
u_x = d(网络输出u) / d(输入x)
```

精确，但代价极高——每求一次导数都要遍历整个计算图，存储大量中间值。

#### 方法 B: 有限差分（FD，本项目的基础）

利用网格上相邻点的值近似导数：

```python
u_x ≈ (U[i+1, j] - U[i-1, j]) / (2 * dx)       # 一阶导
u_xx ≈ (U[i+1, j] - 2*U[i, j] + U[i-1, j]) / dx²  # 二阶导
```

就是高中学的"差商"——用相邻两点的值之差除以距离来近似斜率。

**具体对于 2D NS，需要算这些导数**：
- $u_x, u_y, u_{xx}, u_{yy}$（8 次差分操作）
- $v_x, v_y, v_{xx}, v_{yy}$（8 次差分操作）
- $p_x, p_y$（4 次差分操作）

总共约 **20 次差分操作**。

### Step 4: 计算 Loss——一个数字衡量"有多不满足物理"

```python
pde_loss = mean(res_u² + res_v² + res_div²)   # 越小越满足 PDE
bc_loss = mean((u_top - 1)² + u_wall² + v_wall²)  # 越小越满足边界条件
loss = pde_loss + 10 * bc_loss                    # 总损失
```

**loss = 0 意味着**：网络输出完美满足 PDE + BC → 即找到了正确解。
**loss > 0 意味着**：还不对，需要继续调整网络。

### Step 5: 反向传播——告诉网络该怎么调整

```python
loss.backward()
```

PyTorch 自动计算：loss 对网络每个参数（权重、偏置）的梯度 $\frac{\partial \text{loss}}{\partial \theta}$。

**这一步在做什么？** 告诉网络"每个参数该往哪个方向调，才能让 loss 减小"。

### Step 6: 更新参数

```python
optimizer.step()   # θ ← θ - lr × gradient
```

网络参数朝着让 loss 减小的方向更新一小步。

### Step 7: 重复 Step 2-6，直到 loss 接近 0

通常需要 10 万 ~ 20 万次循环（epochs）。每次循环执行一次 Step 2-6。

---

## 第三章：瓶颈在哪里？

一次训练步的时间拆解：

```
Step 2 (前向)      ████░░░░░░  ~20%   网络推理
Step 3 (PDE残差)   ██████████  ~50%   有限差分计算  ← 主要瓶颈！
Step 5 (反向)      ████████░░  ~30%   梯度回传
```

### 3.1 为什么 Step 3（PDE 残差计算）是瓶颈？

用 PyTorch 做有限差分：

```python
u_x = (U[2:, 1:-1] - U[:-2, 1:-1]) / (2 * dx)   # 操作 1
u_y = (U[1:-1, 2:] - U[1:-1, :-2]) / (2 * dy)   # 操作 2
u_xx = (U[2:, 1:-1] - 2*U[1:-1, 1:-1] + U[:-2, 1:-1]) / dx**2  # 操作 3
# ... 继续 17 个类似操作
```

**表面上看只有 20 行代码，但背后发生了什么？**

每一行代码，PyTorch 底层都要：
1. **启动一次 GPU 计算任务**（kernel launch）—— 有 ~5 微秒的启动开销
2. **从显存读取整个 U 矩阵**（bandwidth）—— U[64,64] = 16KB
3. **计算结果写回显存**
4. **为反向传播记录一个中间结果**—— 额外占显存

20 个操作 = **20 次启动开销 + 20 次读写显存 + 20 个中间 tensor**。

在小网格（64²）上这些开销不明显。但在大网格（256³ = 1600 万点）上：
- 20 次读写 × 64MB/次 = **1.3GB 显存带宽消耗**
- 20 个中间 tensor × 64MB = **1.3GB 额外显存占用**
- PyTorch 的分发/调度开销 × 20 = **不可忽略的延迟**

### 3.2 显存爆炸问题

| 方法 | ldc_3d (48³) 显存 | 说明 |
|------|:-:|------|
| Vanilla autograd | **17.6 GB** | 存储二阶导的完整计算图 |
| PyTorch FD | ~0.5 GB | 存储 20 个中间 tensor |
| **Triton fused** | **0.475 GB** | 零中间 tensor |

### 3.3 问题总结

| 问题 | 原因 | 影响 |
|------|------|------|
| 启动开销大 | 20 次独立 kernel launch | GPU 利用率低 |
| 显存带宽浪费 | 同一数据被重复读取 20 次 | 速度受限于内存带宽 |
| 中间 tensor 多 | PyTorch autograd 需要记录 | 显存占用高，大网格 OOM |

---

## 第四章：Triton 是什么？

### 4.1 背景

GPU 编程传统上有两条路：
- **CUDA C**：极度灵活但开发难度极高（需要手动管理线程、共享内存、同步）
- **PyTorch 原生操作**：简单但不能融合多个操作

**Triton**（由 OpenAI 开发）是第三条路：用 Python 语法写 GPU kernel，编译器自动处理线程管理和内存优化。开发效率接近 PyTorch，但性能接近 CUDA C。

### 4.2 核心思想：算子融合（Kernel Fusion）

PyTorch 做有限差分：

```
操作1: 从显存读 U → 算 u_x → 写回显存 → 存中间 tensor
操作2: 从显存读 U → 算 u_y → 写回显存 → 存中间 tensor
操作3: 从显存读 U → 算 u_xx → 写回显存 → 存中间 tensor
... (×20)
```

Triton 融合后：

```
一次读取: 从显存读 U（每个点读一次邻居）
         → 在寄存器里算完 u_x, u_y, u_xx, u_yy, v_x, ...
         → 组装残差 res_u, res_v, res_div
         → 一次写回结果
```

**20 次启动 → 1 次启动。20 次读写 → 1 次读写。20 个中间 tensor → 0 个。**

---

## 第五章：我们怎么做的？两个 Kernel 拆解

每个 PDE 算子由两个 Triton kernel 组成：**前向 kernel（Forward）** 和 **反向 kernel（Backward/Adjoint）**。

### 5.1 前向 Kernel（@triton.jit: Forward）

**输入**：物理场 $U, V, P$（网格上的值）
**输出**：PDE 残差 $\text{res}_u, \text{res}_v, \text{res}_{div}$

**做了什么？**

对于网格上的每个内部点 $(i, j)$：

```
1. 加载邻居值（5 点星型 stencil）:
   u_c = U[i,j]          ← 中心
   u_xp = U[i+1,j]      ← 右邻居
   u_xm = U[i-1,j]      ← 左邻居
   u_yp = U[i,j+1]      ← 上邻居
   u_ym = U[i,j-1]      ← 下邻居
   (V, P 同理)

2. 在寄存器里计算所有导数:
   u_x = (u_xp - u_xm) / (2*dx)
   u_xx = (u_xp - 2*u_c + u_xm) / dx²
   ...

3. 组装残差:
   res_u = u_c*u_x + v_c*u_y + p_x - nu*(u_xx + u_yy)

4. 写回结果:
   output[i,j] = res_u
```

**一个 kernel，一次启动，所有点并行计算。** 每个线程块处理 BLOCK 个点（autotune 自动选最优 BLOCK）。

### 5.2 反向 Kernel（@triton.jit: Backward/Adjoint）

**输入**：前向输出的残差的梯度（上游梯度 $G_u, G_v, G_{div}$）
**输出**：物理场的梯度 $\frac{\partial \text{loss}}{\partial U}, \frac{\partial \text{loss}}{\partial V}, \frac{\partial \text{loss}}{\partial P}$

**做了什么？**

这是前向 kernel 的"伴随算子"（adjoint）。数学上，如果前向是 $r = f(U)$，反向就是 $\frac{\partial L}{\partial U} = \left(\frac{\partial f}{\partial U}\right)^T G$。

对每个点，反向 kernel 做的是"把上游梯度散播给邻居"——和前向的"从邻居收集信息算残差"正好相反：

```
1. 加载上游梯度（当前点 + 邻居点的梯度）:
   gu_c = G_u[i,j], gu_xp = G_u[i+1,j], ...

2. 加载物理场邻居（用于非线性项的 adjoint）:
   u_c, u_xp, v_c, ...

3. 计算 grad_U[i,j]:
   - 来自 u*u_x 项: gu_c * u_x + "从邻居散播的贡献"
   - 来自 nu*u_xx 项: -nu * (gu_xp + gu_xm) / dx²
   - 来自 div 项: (gd_xm - gd_xp) / (2*dx)

4. 写回:
   grad_U[i,j] = 上述所有贡献之和
```

**为什么不用 PyTorch 自动算？** 因为 PyTorch 的 autograd 会把前向的 20 个操作各自反传一遍（20 个 backward kernel），还要存 20 个中间 tensor。手写 adjoint 一个 kernel 搞定，零额外内存。

### 5.3 两个 Kernel 分别加速了多少？

以 2D LDC (128×128) 为例（Track 0 数据）：

| | PyTorch FD | Triton | 加速 |
|---|:-:|:-:|:-:|
| Forward kernel | 1.03 ms | 0.39 ms | **2.64x** |
| Backward kernel | 3.73 ms | 2.14 ms | **1.74x** |
| **总计** | **4.76 ms** | **2.53 ms** | **1.88x** |

以 3D LDC (48³) 为例（Track 0 数据）：

| | PyTorch FD | Triton | 加速 |
|---|:-:|:-:|:-:|
| Forward kernel | 2.41 ms | 0.78 ms | **3.10x** |
| Backward kernel | 8.17 ms | 4.57 ms | **1.79x** |
| **总计** | **10.58 ms** | **5.34 ms** | **1.98x** |

以 3D TGV (20×96³, 1770万点) 为例（Scaling 实验）：

| | PyTorch FD | Triton | 加速 |
|---|:-:|:-:|:-:|
| Forward + Backward (total) | 67.32 ms | 3.81 ms | **17.67x** |

**规律**：
- 网格越大，加速越明显（因为 PyTorch 的 kernel launch 开销被放大，而 Triton 只有一次）
- Forward 加速比 > Backward 加速比（backward 涉及更多邻居访问，内存带宽是瓶颈）
- 3D 问题加速最猛（stencil 点数多：7 点 vs 2D 的 5 点，融合收益更大）

---

## 第六章：实际收益总结

### 整步训练加速（Track 1，包含网络推理）

| Case | 维度 | mlp_triton vs vanilla | cnn_triton vs canpinn | 显存节省 |
|------|:---:|:-:|:-:|:-:|
| burgers_1d_unsteady | 1D+T | **4.16x** | 1.15x | 6x |
| ldc_2d | 2D | **4.38x** | **1.47x** | 15x |
| ldc_3d | 3D | **27.45x** | 1.11x | **37x** |
| tgv_2d | 2D+T | **14.81x** | **1.64x** | 20x |
| transport_2d | 2D+T | **10.38x** | 1.11x | 9x |

### 纯 Kernel 加速（Track 0，不含网络推理）

随网格增大，加速比从 2x 增长到 **19x**（tgv_3d, 4200 万点）。

### 精度

Triton kernel 和 PyTorch FD 计算的是**数学上完全相同的有限差分公式**。梯度误差 < $10^{-11}$（float64 验证）。加速是**零精度损失**的。
