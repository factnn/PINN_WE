# Kernel Launch 量化分析：8 个 Case 的详细拆解

## 测量方法

使用 `torch.profiler` 精确计数单次训练步中的 CUDA kernel launch 数量。
包含：模型 forward + PDE 残差 + loss + backward + BC loss。
不含：optimizer.step()。

---

## 汇总表

| Case | 网格 | PyTorch FD (canpinn) | Triton | 减少比 | 说明 |
|------|------|:---:|:---:|:---:|------|
| burgers_1d_steady | 256 | 94 | 70 | 1.3x | 1D 太小，kernel 少 |
| burgers_1d_unsteady | 1024×100 | 144 | 98 | 1.5x | |
| ldc_2d | 128×128 | 227 | 147 | 1.5x | 3 fields (U,V,P) |
| ldc_3d | 48³ | 572 | 291 | 2.0x | 4 fields (U,V,W,P) |
| tgv_2d | 20×64×64 | 317 | 199 | 1.6x | 3 fields + time |
| tgv_3d | 10×32³ | 750 | 355 | 2.1x | 4 fields + time |
| transport_2d | 20×64×64 | 215 | 111 | 1.9x | 1 field |
| sod_2d | 1000×200 | 335 | 309 | 1.1x | kernel 效率低 |

**规律**：
- 物理场越多（3D NS 有 4 个场）→ PyTorch FD 的 kernel 数暴增 → Triton 减少越多
- 3D 问题（ldc_3d, tgv_3d）减少最明显（2.0-2.1x）
- 1D 问题（burgers_1d_steady）本来 kernel 就少，减少空间小

---

## 详细拆解：ldc_2d (128×128, 2D 稳态 NS)

### PyTorch FD：227 次 CUDA kernel

```
操作分类                    次数    说明
─────────────────────────────────────────────────────
fill (zeros初始化)           53    每个中间tensor初始化为0
add (加法)                   38    u_x+v_y, res_u+res_v, grad累加等
Memcpy DtoD (数据拷贝)       29    切片U[2:,1:-1]时的内存拷贝
copy (切片view)              26    创建tensor view/contiguous
mul_scalar (标量乘)          20    /(2*dx) 编译为 *(1/2dx)
neg (取负)                   12    -nu*(u_xx+u_yy) 中的取负
mul_tensor (张量乘)          11    uc*u_x, vc*u_y 等非线性项
mul_tensor (backward)         8    反向传播中的乘法
pow (幂运算)                  3    res² (平方)
reduce_mean                   1    最终mean()求均值
其他                         26    各种反向传播辅助kernel
─────────────────────────────────────────────────────
总计                        227
```

**逐步对应到物理**：
- `u_x = (U[2:,1:-1] - U[:-2,1:-1]) / (2*dx)` → 需要 slice(DtoD) + sub(add) + mul_scalar = 3 个 kernel
- 类似操作重复 10 次（u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, p_x, p_y）= 30 个 kernel
- 组装残差 `res_u = uc*u_x + vc*u_y + p_x - nu*(u_xx+u_yy)` → mul + add + add + mul_scalar + add + neg = 6 个 kernel × 3 (res_u, res_v, res_div) = 18 个 kernel
- `(res²).mean()` → pow + add + reduce = 4 个 kernel
- backward 对上述所有操作逆序执行 → ~150 个 kernel
- 初始化 grad tensors → 53 个 fill kernel

### Triton：147 次 CUDA kernel (72 次纯 PDE 部分)

```
操作分类                    次数    说明
─────────────────────────────────────────────────────
★ ldc_fwd_kernel             1    前向stencil（替代了~50个PyTorch ops）
★ ldc_bwd_kernel             1    反向adjoint（替代了~150个PyTorch ops）
add (boundary fix)          24    _add_boundary_gradients 边界修正
mul_scalar                  20    boundary 梯度的系数乘法
fill (grad初始化)            4    grad_U/V/P 初始化为0
neg                          4    boundary 修正取负
pow                          3    res²
mul_tensor                   5    上游梯度 scale (2/N * grad_out)
reduce_mean                  1    最终 loss reduction
网络 forward/backward       ~75    MLP 的矩阵乘法（我们没动）
其他                         9    杂项
─────────────────────────────────────────────────────
总计                        147
```

**核心 2 个 Triton kernel 替代了 PyTorch 的 ~200 个操作**。

剩余 145 个是：
- 网络自身的 forward/backward（~75 个，我们没动）
- 边界梯度修正（~50 个，用 PyTorch 原生切片实现）
- Loss reduction 和上游梯度 scale（~20 个）

---

## 详细拆解：tgv_3d (10×32³, 3D 非稳态 NS)

### PyTorch FD：750 次 CUDA kernel

3D NS 有 4 个物理场 (U, V, W, P)，每个场需要：
- 6 个空间导数（x±, y±, z±）+ 1 个时间导数 = 7 个差分
- 每个差分 ~3 个 kernel (slice + sub + div)
- 4 个场 × 7 个导数 × 3 = 84 个前向 kernel（仅导数部分）
- 加上非线性项组装、残差平方、mean、backward 等 → 750 个

### Triton：355 次 CUDA kernel

```
★ ns3d_fwd_kernel            1    一个kernel算完4场的全部导数+残差
★ ns3d_bwd_kernel            1    一个kernel算完全部adjoint
网络 forward/backward      ~200    SIREN MLP (256 width, 深)
boundary fix               ~100    6个面的边界梯度修正
其他                        ~53    reduction, scale, fill
─────────────────────────────────────────────────────
总计                        355
```

---

## 关键洞察

### 1. Triton 实际只贡献了 2 个 kernel，但替代了几百个

| Case | PyTorch 纯 PDE ops | Triton 替代为 | 等效替代比 |
|------|:---:|:---:|:---:|
| burgers_1d_steady | ~30 | 2 (fwd+bwd) | **15:1** |
| burgers_1d_unsteady | ~60 | 2 | **30:1** |
| ldc_2d | ~200 | 2 | **100:1** |
| ldc_3d | ~450 | 2 | **225:1** |
| tgv_2d | ~250 | 2 | **125:1** |
| tgv_3d | ~550 | 2 | **275:1** |
| transport_2d | ~120 | 2 | **60:1** |
| sod_2d | ~200 | 2 | **100:1** |

### 2. 为什么整体 reduction 只有 1.5-2.1x 而不是 100x？

因为还有其他 kernel 我们没动：
- **网络 forward/backward**：MLP 的 Linear + Tanh 层各自是独立 kernel（~50-200 个），这些不是我们的优化对象
- **边界梯度修正**：`_add_boundary_gradients()` 用 PyTorch 切片实现（~20-100 个 kernel），可以进一步 fuse 但性价比低
- **Loss reduction**：`mean(res²)` 等（~10 个 kernel）

### 3. 进一步优化的空间

如果把边界修正也 fuse 进 Triton kernel（技术上可行但代码复杂度高）：
- ldc_2d: 147 → ~80（网络+reduction 不可避免）
- tgv_3d: 355 → ~210

极限是**网络自身的 kernel 数量**——这是我们碰不到的下限。


---

# 8 个 Kernel 的进一步优化空间分析

## 总结表

| Case | 已 Fuse 进 Triton | 仍用 PyTorch 的操作 | 剩余 ops 数 | 可 fuse？ |
|------|:-:|------|:---:|:---:|
| burgers_1d_steady | fwd + bwd | reduction | ~4 | ✅ 简单 |
| burgers_1d_unsteady | fwd + bwd | reduction + grad_scale | ~8 | ✅ 简单 |
| ldc_2d | fwd + bwd | reduction + grad_scale + **boundary(16)** | ~24 | ⚠️ 中等 |
| ldc_3d | fwd + bwd | reduction + grad_scale + **boundary(30)** | ~38 | ⚠️ 中等 |
| tgv_2d | fwd + bwd | reduction + grad_scale + **boundary(20)** | ~28 | ⚠️ 中等 |
| tgv_3d | fwd + bwd | reduction + grad_scale + **boundary(36)** | ~44 | ⚠️ 中等 |
| transport_2d | fwd + bwd | reduction + grad_scale + **boundary(6)** | ~14 | ✅ 简单 |
| sod_2d | fwd + bwd | reduction + grad_scale + **boundary(20)** | ~28 | ⚠️ 中等 |

---

## 各 Case 详细分析

### 1. burgers_1d_steady（最简单，几乎无优化空间）

```
已 Fuse:    fwd_kernel (stencil) + bwd_kernel (adjoint)
未 Fuse:    (res²).mean() → 4 个 kernel (pow + reduce)
Boundary:   无（1D kernel 内部直接跳过边界点）
```

**优化空间**：极小。reduction 可以 fuse 进 fwd_kernel（用 `tl.sum`），但收益 < 1%（4 个 kernel vs 总共 70 个）。**不值得做。**

---

### 2. burgers_1d_unsteady（小量优化空间）

```
已 Fuse:    fwd_kernel + bwd_kernel
未 Fuse:    (res²).mean()        → 4 个 kernel
            2/N * grad_out scale → 4 个 kernel (乘法)
Boundary:   无（kernel 内部处理了边界跳过）
```

**优化空间**：可以把 `scale = 2/N * grad_out` 合并到 bwd_kernel 的开头（每个线程自己乘 scale）。收益 ~5%。**性价比一般。**

---

### 3. ldc_2d（中等优化空间）

```
已 Fuse:    ldc_fwd_kernel + ldc_bwd_kernel
未 Fuse:    (res²).mean()              → 4 个 kernel
            2/N * grad_out scale       → 4 个 kernel
            _add_boundary_gradients    → 16 个 kernel (4条边 × 4个slice+add操作)
```

**边界修正具体在做什么？**

```python
# x=0 边 (左边界)
grad_u[0, 1:-1] += -Gu[0] * (U[1, 1:-1] * inv_2dx + nu * inv_dx2)
grad_v[0, 1:-1] += -Gv[0] * (U[1, 1:-1] * inv_2dx + nu * inv_dx2)
grad_u[0, 1:-1] += -Gdiv[0] * inv_2dx
# x=Nx-1 边 (右边界) ... 类似
# y=0, y=Ny-1 边 ... 类似
# P 边界 ...
```

每条边只有 `Nx` 或 `Ny` 个点（一维切片），计算量很小。但每个 `+=` 都是一次 kernel launch。

**可否 fuse？** 可以写一个 `boundary_kernel`，一次处理所有 4 条边。但边界逻辑因 case 而异（LDC 的 BC vs TGV 的周期BC），复用性低。

**收益估算**：147 → 127（减 20，约 14%），实际时间收益 < 5%（边界点计算量极小）。

---

### 4. ldc_3d（最大优化空间）

```
已 Fuse:    ns3d_fwd_kernel + ns3d_bwd_kernel
未 Fuse:    (res²).mean()              → 4 个 kernel
            2/N * grad_out scale       → 4 个 kernel
            _add_boundary_gradients    → 30 个 kernel (6个面 × 5个slice操作)
```

**6 个面 × 4 个场 (U,V,W,P) × 多条 slice 操作 = 30 个 kernel launch。**

这是所有 case 中边界开销最大的（因为 3D 有 6 个面）。

**可否 fuse？** 可以写一个 3D boundary kernel，遍历 6 个面各自处理。每个面是一个 2D slice（48×48 个点），可以并行。

**收益估算**：291 → 257（减 34，约 12%），时间收益中等（3D 面积大，边界计算不再可忽略）。

---

### 5. tgv_2d（中等优化空间）

```
已 Fuse:    ns2d_fwd_kernel + ns2d_bwd_kernel
未 Fuse:    (res²).mean()              → 4 个 kernel
            2/N * grad_out scale       → 4 个 kernel
            _add_boundary_gradients    → 20 个 kernel (4边 + 2时间面)
```

TGV 有**时间维度**的边界（t=0 和 t=Nt-1），比纯空间多了 2 个面的处理。

**收益估算**：199 → 175（减 24，约 12%）。

---

### 6. tgv_3d（优化空间最大）

```
已 Fuse:    ns3d_fwd_kernel + ns3d_bwd_kernel
未 Fuse:    (res²).mean()              → 4 个 kernel
            2/N * grad_out scale       → 4 个 kernel
            _add_boundary_gradients    → 36 个 kernel (6空间面 + 2时间面)
```

3D + 时间 = 8 个边界面，每个面多次 slice 操作 → 36 个 kernel。

**这是优化回报最高的 case**：355 → 315（减 40，约 11%）。如果边界 fuse + reduction fuse，可达 355 → 280。

---

### 7. transport_2d（简单，边界少）

```
已 Fuse:    advdiff_fwd_kernel + advdiff_bwd_kernel
未 Fuse:    (res²).mean()              → 4 个 kernel
            2/N * grad_out scale       → 4 个 kernel
            _add_boundary_gradients    → 6 个 kernel (只有时间面t=0,t=Nt-1)
```

只有 1 个标量场，边界修正很少。**几乎不需要进一步优化。**

---

### 8. sod_2d（kernel 本身效率需改善）

```
已 Fuse:    euler_fwd_kernel + euler_bwd_kernel
未 Fuse:    (res²).mean()              → 4 个 kernel
            2/N * grad_out scale       → 4 个 kernel
            _add_boundary_gradients    → 20 个 kernel
```

**但 sod_2d 的核心问题不是边界未 fuse，而是 kernel 本身效率低**（Track 0 只有 1.04x 加速）。需要先解决：
1. 2D dispatch → 1D flatten（GPU 利用率低）
2. backward kernel 寄存器压力过高（Euler adjoint 变量太多）

这是 8 个 kernel 中唯一需要**重写**（而非追加优化）的。

---

## 优化优先级建议

| 优先级 | 操作 | 影响的 case | 工作量 | 预计收益 |
|:---:|------|------|:---:|------|
| 1 | **sod_2d kernel 重写**（1D flatten + 优化寄存器） | sod_2d | 大 | 从 1.04x → 3-5x |
| 2 | reduction fuse（`tl.sum` 进 fwd_kernel） | 全部 8 个 | 小 | 每个减 4 kernels |
| 3 | grad_scale fuse（乘法进 bwd_kernel 开头） | 7 个（除 steady） | 小 | 每个减 4 kernels |
| 4 | boundary fuse（3D cases 优先） | ldc_3d, tgv_3d | 中 | 减 30-36 kernels |
| 5 | boundary fuse（2D cases） | ldc_2d, tgv_2d, sod_2d | 中 | 减 16-20 kernels |

**结论**：除了 sod_2d 需要重写外，其余 7 个 kernel 已经很优秀了。进一步优化（boundary fuse）的收益是 10-14% kernel 数减少，但实际时间收益很小（<5%），因为边界计算量本身微不足道。优先级不高，留作论文的"future work"即可。
