# sod_2d Kernel 深度分析与优化方案（基于实测 Profiling）

## 一、当前问题

Track 0 结果（网格 Nt=200, Nx=1000, 20 万点）：

| | PyTorch FD | Triton | 加速比 |
|---|:-:|:-:|:-:|
| Forward + Backward | 1.987 ms | 1.911 ms | **1.04x（几乎没加速）** |

对比其他 case（ldc_2d 1.88x, tgv_3d 2.51x），sod_2d 是唯一没有显著加速的。

---

## 二、实测时间拆解（triton.testing.do_bench）

### Triton 前向

| 部分 | 时间 | 说明 |
|------|:---:|------|
| euler_fwd_kernel | 0.045 ms | Triton stencil（极快） |
| (res²).mean() reduction | 0.050 ms | PyTorch 原生 |
| **Forward 总计** | **0.078 ms** | |

### Triton 反向

| 部分 | 时间 | 占比 | 说明 |
|------|:---:|:---:|------|
| euler_bwd_kernel 本身 | **0.021 ms** | **1.1%** | Triton adjoint（极快！） |
| `_add_boundary_gradients` | **1.207 ms** | **65.8%** | ← **罪魁祸首！** PyTorch 切片 |
| autograd 调度开销 | 0.542 ms | 29.6% | PyTorch Function 调度 |
| scale (2/N * grad) | 0.012 ms | 0.7% | |
| reduction | 0.050 ms | 2.7% | |
| **Backward 总计** | **~1.833 ms** | 100% | |

### 结论

**Triton kernel 本身完全没有问题！**

- fwd_kernel: 0.045 ms（极快）
- bwd_kernel: 0.021 ms（比 fwd 还快！）
- **瓶颈是 `_add_boundary_gradients_euler` 函数（1.2ms）**——纯 PyTorch 切片操作，没有被 Triton fuse

---

## 三、为什么 boundary gradients 这么慢？

`_add_boundary_gradients_euler` 的代码有 ~20 个 PyTorch 操作。每个都是一次 kernel launch。

但更关键的是：**Euler 方程的边界 Jacobian 计算很复杂**（涉及 `u²`, `u³`, `e/ρ` 等非线性项），每个操作都需要从显存读数据。这些操作不大（只有 Nt-2 或 Nx-2 个点），但 **launch overhead × 20 + 中间 tensor 分配 × 20 = 1.2ms**。

对比其他 case 的 boundary：LDC/TGV 的边界 Jacobian 是线性的（只有 `±Gu * inv_2dx` 之类），耗时 ~0.15ms。sod_2d 慢 8 倍是因为 Euler flux Jacobian 的非线性复杂度。

---

## 四、之前的错误推理 vs 实测真相

| 之前推理 | 实测真相 |
|---------|---------|
| ❌ 2D dispatch 导致 GPU 没吃饱 | ✅ 1584 blocks / 14.7 waves，GPU 很饱 |
| ❌ bwd kernel 寄存器压力高 | ✅ bwd_kernel 只花 0.021ms（极快） |
| ❌ Euler adjoint 算术密度高 | ✅ kernel 内部不是瓶颈 |
| ❌ 需要 1D flatten | ✅ dispatch 方式不是问题 |
| — | ✅ **boundary 占了 65.8% 的时间！** |

---

## 五、正确的优化方案

### 方案：将 boundary gradients fuse 进一个 Triton kernel

写一个 `euler_boundary_kernel`，一次 launch 处理所有 4 条边的梯度修正：

```python
@triton.jit
def euler_boundary_kernel(
    U_ptr, M_ptr, E_ptr, Gr_ptr, Gm_ptr, Ge_ptr,
    grad_r_ptr, grad_m_ptr, grad_e_ptr,
    Nt, Nx, dx, dt, gamma, BLOCK: tl.constexpr
):
    # 处理 t=0, t=Nt-1, x=0, x=Nx-1 四条边
    # 每条边是一个 1D 数组，可以并行处理
    ...
```

**预计收益**：1.207ms → ~0.05ms（一次 launch + 显存只读一次）

**优化后总时间**：
- Forward: 0.078ms（不变）
- Backward: 0.021 + 0.05 + 0.542 + 0.062 = ~0.675ms
- **Total: ~0.753ms**
- **加速比: 1.987 / 0.753 = 2.64x**

---

## 六、优化后的预期对比

| | 当前 | 优化后（boundary fuse） |
|---|:-:|:-:|
| Forward | 0.078 ms | 0.078 ms |
| Backward | 1.833 ms | ~0.675 ms |
| Total | 1.911 ms | **~0.753 ms** |
| vs PyTorch FD | 1.04x | **2.64x** |

---

## 七、对其他 case 的启示

sod_2d 的实测揭示了一个重要规律：**kernel 本身不是瓶颈时，boundary 处理可能成为新的瓶颈**。

| Case | boundary 耗时 | 占 backward 比例 | 值得 fuse？ |
|------|:-:|:-:|:-:|
| burgers_1d_steady | ~0 ms | 0% | 无边界修正 |
| burgers_1d_unsteady | ~0 ms | 0% | 无边界修正 |
| ldc_2d | ~0.15 ms | ~7% | 不值得 |
| ldc_3d | ~0.3 ms | ~7% | 边缘 |
| tgv_2d | ~0.2 ms | ~7% | 不值得 |
| tgv_3d | ~0.4 ms | ~8% | 边缘 |
| transport_2d | ~0.05 ms | ~4% | 不值得 |
| **sod_2d** | **1.2 ms** | **66%** | **必须做！** |

sod_2d 是唯一一个 boundary 开销占主导的 case，因为 Euler 方程的边界 Jacobian 涉及复杂非线性计算。

---

## 八、实现计划

```
Phase 1: 写 euler_boundary_kernel（1.5小时）
Phase 2: 验证正确性（30分钟）
Phase 3: 重跑 Track 0 + Scaling（15分钟）
总计：~2 小时
```
