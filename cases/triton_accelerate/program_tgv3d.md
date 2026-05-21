# autoresearch-TGV-3D

3D Taylor-Green Vortex 端到端实验：先跑 baseline，再实现 Triton kernel 并验证收敛。

## Setup

```bash
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
cd /share/project/zpy/PINN_WE/cases/triton_accelerate
```

**GPU 分配**：使用 GPU 2 和 3。

## Phase 1: 跑 Baseline（vanilla / canpinn / compile）

先跑 5 个非 Triton 脚本，建立 baseline：

```bash
# GPU 2
CUDA_VISIBLE_DEVICES=2 python cases/tgv_3d/mlp_vanilla.py --max-epochs 50000 --threshold 1e-3
CUDA_VISIBLE_DEVICES=2 python cases/tgv_3d/mlp_canpinn.py --max-epochs 50000 --threshold 1e-3
CUDA_VISIBLE_DEVICES=2 python cases/tgv_3d/mlp_compile.py --max-epochs 50000 --threshold 1e-3

# GPU 3
CUDA_VISIBLE_DEVICES=3 python cases/tgv_3d/cnn_canpinn.py --max-epochs 50000 --threshold 1e-3
CUDA_VISIBLE_DEVICES=3 python cases/tgv_3d/cnn_compile.py --max-epochs 50000 --threshold 1e-3
```

记录每个的 Summary（T2S, Avg_Step_ms, Peak_Mem_GB, L2_err）。

**注意**：3D 网格 32x32x32x10 = 327680 点，MLP 输入很大。如果 OOM：
- 减小网格到 24x24x24x8
- 或减小 MLP width 到 64

## Phase 2: 实现 3D NS Triton Kernel

3D incompressible NS：
```
res_u = u_t + u*u_x + v*u_y + w*u_z + p_x - nu*(u_xx + u_yy + u_zz)
res_v = v_t + u*v_x + v*v_y + w*v_z + p_y - nu*(v_xx + v_yy + v_zz)
res_w = w_t + u*w_x + v*w_y + w*w_z + p_z - nu*(w_xx + w_yy + w_zz)
res_div = u_x + v_y + w_z
```

数据维度：`U, V, W, P: [Nt, Nx, Ny, Nz]`。

### 需要做的事：

1. 在 `kernels/` 下创建 `stencil_3d.py`，包含：
   - `ns3d_fwd_kernel`：计算 res_u, res_v, res_w, res_div
   - `ns3d_bwd_kernel`：计算 grad_u, grad_v, grad_w, grad_p
   - 参考 `stencil_2d.py` 的结构，扩展到 3D

2. 验证 forward：
   ```python
   from cases.tgv_3d.common import pde_residual_pytorch, make_grid, exact_uvwp
   # Triton forward loss == PyTorch FD loss
   ```

3. 验证 backward：对比 PyTorch autograd 梯度，误差 < 1e-5。

4. 更新 `cases/tgv_3d/mlp_triton.py` 和 `cases/tgv_3d/cnn_triton.py`。

### Kernel 设计参考（2D → 3D 扩展）

2D kernel grid: `(Nt-2, ceil(Nx-2/BX), ceil(Ny-2/BY))`
3D kernel grid: `(Nt-2, ceil(Nx-2/BX), ceil((Ny-2)*(Nz-2)/(BY*BZ)))`

或者用 flatten 的方式：每个 thread 处理一个 (t, x, y, z) 内部点。

## Phase 3: 端到端验证

```bash
CUDA_VISIBLE_DEVICES=2 python cases/tgv_3d/mlp_triton.py --max-epochs 50000 --threshold 1e-3
CUDA_VISIBLE_DEVICES=3 python cases/tgv_3d/cnn_triton.py --max-epochs 50000 --threshold 1e-3
```

**成功标准**：
- T2S 有值（loss < 1e-3）
- L2_err 与 canpinn baseline 相当
- Avg_Step_ms < canpinn 的 Avg_Step_ms（期望 1.5x-2x 加速）

## Experiment Loop

LOOP:
1. Phase 1 baseline → 记录结果
2. 写 3D Triton kernel（forward first）
3. 验证 forward 一致性（loss 值相同）
4. 写 backward kernel
5. 验证 backward 梯度精度
6. 端到端训练
7. 如果 T2S 存在且 L2 合理：**DONE**
8. 如果不收敛：
   - loss 值不同 → 修 forward kernel
   - 梯度错 → 修 backward kernel
   - loss 相同但不收敛 → 训练超参数问题
   回到对应步骤

**Context management**: `/compact` when near limit, re-read this file and `kernels/stencil_2d.py`（作为参考）。

**NEVER STOP** until triton 版本端到端收敛且 L2 合理。
