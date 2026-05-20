# 2D 实验计划

## 算例

### 算例1：2D Taylor-Green Vortex (TGV)
$$u_t + u \cdot \nabla u = -\nabla p + \nu \Delta u, \quad \nabla \cdot u = 0$$

- 域：$(x,y) \in [0, 2\pi]^2$，$t \in [0, 1]$，$\nu = 0.01$
- IC：$u = \sin x \cos y$，$v = -\cos x \sin y$
- 有精确解：$u(x,y,t) = \sin x \cos y \cdot e^{-2\nu t}$
- 为什么选：经典 CFD 验证算例，含对流项（非线性）+ 扩散项，stencil 加速最典型

### 算例2：2D Lid-Driven Cavity (LDC)
$$u_t + u \cdot \nabla u = -\nabla p + \nu \Delta u, \quad \nabla \cdot u = 0$$

- 域：$(x,y) \in [0,1]^2$，稳态，$\nu = 0.01$（Re=100）
- BC：顶盖 $u=1$，其余壁面 $u=v=0$
- 参考解：Ghia et al. 1982 数值解
- 为什么选：工业界最常用的 CFD 基准，无精确解但有权威参考数据

---

## 网络

- **MLP**：输入 $(x,y,t)$，4层×64宽，tanh
- **Phy-CNN**：Conv2D，输出网格场，差分算残差

---

## 四种实现 × 两种网络 = 7个实验

| 网络 | PyTorch FD | torch.compile | Triton |
|------|-----------|--------------|--------|
| MLP (Vanilla autograd) | ✓ | — | — |
| MLP (CAN-PINN) | ✓ | ✓ | ✓ |
| Phy-CNN | ✓ | ✓ | ✓ |

**Phy-CNN**：Conv2D 网络，输出网格场，差分算残差。与 Triton 对比特别有说服力——"Triton fused kernel 比 PyTorch 自己的 conv2d 还快"。

---

## 关键技术：2D Halo Exchange

每个 Block 负责 [BLOCK_X, BLOCK_Y] 的网格，需要加载相邻 Block 的边界数据到 shared memory：

```
[halo_top              ]
[halo_left | interior | halo_right]
[halo_bottom           ]
```

---

## 指标（同 1D）

- Kernel 级别：`do_bench` + 显存
- 端到端：warmup 50步，2950步，5次中位数
- 精度：L2 误差 vs 精确解（TGV）/ Ghia 参考解（LDC）

---

## 文件结构

```
baseline/tgv_2d_compare.py      # TGV 三种方法 kernel 对比
baseline/tgv_2d_train.py        # TGV 端到端训练
baseline/ldc_2d_compare.py      # LDC kernel 对比
baseline/ldc_2d_train.py        # LDC 端到端训练
kernels/stencil_2d.py           # 2D Triton kernel（TGV + LDC 共用）
```
