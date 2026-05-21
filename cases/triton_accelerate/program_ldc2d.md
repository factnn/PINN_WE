# autoresearch-LDC-2D

2D Lid-Driven Cavity 端到端实验：先跑 baseline，再实现 Triton kernel 并验证收敛。

## Setup

```bash
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
cd /share/project/zpy/PINN_WE/cases/triton_accelerate
```

**GPU 分配**：使用 GPU 0 和 1。

## Phase 1: 跑 Baseline（vanilla / canpinn / compile）

先跑 5 个非 Triton 脚本，建立 baseline 数据：

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python cases/ldc_2d/mlp_vanilla.py --max-epochs 10000 --threshold 1e-3
CUDA_VISIBLE_DEVICES=0 python cases/ldc_2d/mlp_canpinn.py --max-epochs 10000 --threshold 1e-3
CUDA_VISIBLE_DEVICES=0 python cases/ldc_2d/mlp_compile.py --max-epochs 10000 --threshold 1e-3

# GPU 1
CUDA_VISIBLE_DEVICES=1 python cases/ldc_2d/cnn_canpinn.py --max-epochs 10000 --threshold 1e-3
CUDA_VISIBLE_DEVICES=1 python cases/ldc_2d/cnn_compile.py --max-epochs 10000 --threshold 1e-3
```

记录每个的 Summary（T2S, Avg_Step_ms, Peak_Mem_GB, L2_Ghia）。

## Phase 2: 实现 Steady-State NS Triton Kernel

LDC 是 **steady-state**（无时间导数，无压力），PDE 残差：
```
res_u = u*u_x + v*u_y - nu*(u_xx + u_yy)
res_v = u*v_x + v*v_y - nu*(v_xx + v_yy)
```

数据维度：`U, V: [Nx, Ny]`（无 Nt 维度，无 P）。

### 需要做的事：

1. 在 `kernels/` 下创建 `stencil_ldc.py`，包含：
   - `ldc_fwd_kernel`：计算 res_u, res_v
   - `ldc_bwd_kernel`：计算 grad_u, grad_v
   - `_LDCTriton(torch.autograd.Function)` 封装
   - `_add_boundary_gradients`：边界梯度补齐函数（**重要！** kernel 只覆盖 interior，边界需要 Python 补）

2. 验证 forward 正确性：
   ```python
   from cases.ldc_2d.common import pde_residual_pytorch, make_grid
   # 比较 Triton forward 和 PyTorch FD 的 loss 值是否一致
   ```

3. 验证 backward 正确性：
   ```python
   # 用 torch.autograd.gradcheck 或对比 PyTorch autograd 的梯度
   ```

4. 更新 `cases/ldc_2d/mlp_triton.py` 和 `cases/ldc_2d/cnn_triton.py` 使用新 kernel。

## Phase 3: 端到端验证

```bash
CUDA_VISIBLE_DEVICES=0 python cases/ldc_2d/mlp_triton.py --max-epochs 10000 --threshold 1e-3
CUDA_VISIBLE_DEVICES=1 python cases/ldc_2d/cnn_triton.py --max-epochs 10000 --threshold 1e-3
```

**成功标准**：
- T2S 有值（loss < 1e-3）
- L2_Ghia 与 canpinn baseline 相当（差别 < 2x）
- Avg_Step_ms < canpinn 的 Avg_Step_ms

## 来自 2D TGV 的 Lessons Learned（必读！）

### 坑 1：Scale 缩放系数
`Loss = mean(res_u² + res_v²)` 中，`+` 是 element-wise add，分母是 `N_total`，**不是 `2 * N_total`**。
Backward 中 `scale = 2.0 / N_total * grad_out`，不要多除。

### 坑 2：边界梯度必须补齐
Triton kernel 只处理 interior 点，x=0, x=Nx-1, y=0, y=Ny-1 四条边的梯度在内核中为 0。
必须写 `_add_boundary_gradients` 函数，用 PyTorch tensor ops 补齐。

LDC 的边界贡献来自：
- **x=0**：U[0,j] 作为 u_l in res_u[0,j-1] 和 v_l in res_v[0,j-1]
  `dRes_u/du_l = -u_c/(2dx) - nu/dx²`，`dRes_v/dv_l = -u_c/(2dx) - nu/dx²`
  **注意**：res_v 中 v_l 的 u_c 来自 U 场（`u_c*v_x` 项），不是 V 场！
- **x=Nx-1**：类似，u_r 贡献，符号相反
- **y=0 / y=Ny-1**：类似

### 坑 3：用多种测试场验证梯度
只测 sin/cos 不够，必须用 MLP 输出场（随机初始化模型）再测一次。
sin/cos 场太光滑，散度接近 0，会掩盖 Gdiv 相关的符号错误。

### 坑 4：验证时检查 ALL 点（含边界）
`(gU_pt - gU_tr).abs().max()` 而不是 `(gU_pt[1:-1,1:-1] - gU_tr[1:-1,1:-1]).abs().max()`。
只查 interior 会漏掉边界问题。

### 坑 5：Triton JIT 缓存
修改 kernel 后必须 `rm -rf ~/.triton/cache && find . -name "__pycache__" -type d -exec rm -rf {} +`。
最好每次测试前设置 `export TRITON_CACHE_DIR=/tmp/triton_fresh_$$`。

### 验证检查清单（修完 kernel 后逐条过）
- [ ] Forward loss 与 PyTorch FD 完全一致（rel diff < 1e-12）
- [ ] Interior 梯度误差 < 1e-5
- [ ] ALL 点梯度误差（含边界）< 1e-5
- [ ] 用 MLP 输出场（非光滑 sin/cos）再测一遍
- [ ] 500 步小规模训练不发散
- [ ] 全量训练收敛且 L2 合理

## Experiment Loop

LOOP:
1. 跑 baseline → 记录结果
2. 写 Triton kernel（forward + backward + boundary fix）
3. 用验证检查清单逐条验证
4. `git commit`
5. 端到端训练
6. 如果 T2S 存在且 L2 合理：**DONE**，更新 `results.md`
7. 如果不收敛：诊断问题（对比 loss 值、梯度），修 kernel，回到 3

**Context management**: `/compact` when near limit, re-read this file and `kernels/stencil_2d.py`（作为 2D 参考）。

**NEVER STOP** until triton 版本端到端收敛且 L2 合理。
