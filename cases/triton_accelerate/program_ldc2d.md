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

LDC 是 **steady-state**（无时间导数），PDE 残差：
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

## Experiment Loop

LOOP:
1. 跑 baseline → 记录结果
2. 写 Triton kernel
3. 验证 forward 一致性
4. 验证 backward 梯度精度
5. 端到端训练
6. 如果 T2S 存在且 L2 合理：**DONE**
7. 如果不收敛：诊断问题（对比 loss 值、梯度），修 kernel，回到 3

**Context management**: `/compact` when near limit, re-read this file.

**NEVER STOP** until triton 版本端到端收敛且 L2 合理。
