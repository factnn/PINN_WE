# autoresearch-TGV-3D

3D Taylor-Green Vortex 端到端实验：先跑 baseline，再实现 3D Triton kernel 并验证收敛。

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
   - `_NS3DTriton(torch.autograd.Function)` 封装
   - `_add_boundary_gradients`：边界梯度补齐函数（**重要！** 2D 有 6 个边界面，3D 有 8 个）

2. 验证 forward：
   ```python
   from cases.tgv_3d.common import pde_residual_pytorch, make_grid, exact_uvwp
   # Triton forward loss == PyTorch FD loss
   ```

3. 验证 backward：对比 PyTorch autograd 梯度，误差 < 1e-5。

4. 更新 `cases/tgv_3d/mlp_triton.py` 和 `cases/tgv_3d/cnn_triton.py`。

### Kernel 设计参考（参考 2D `stencil_2d.py` 结构）

2D kernel grid: `(Nt-2, ceil((Nx-2)/BX), ceil((Ny-2)/BY))`
3D kernel grid: `(Nt-2, ceil((Nx-2)/BX), ceil((Ny-2)/BY), ceil((Nz-2)/BZ))`

或者用 flatten 的方式：每个 thread 处理一个 (t, x, y, z) 内部点，BLOCK 在 1D 上划分。

## Phase 3: 端到端验证

```bash
CUDA_VISIBLE_DEVICES=2 python cases/tgv_3d/mlp_triton.py --max-epochs 50000 --threshold 1e-3
CUDA_VISIBLE_DEVICES=3 python cases/tgv_3d/cnn_triton.py --max-epochs 50000 --threshold 1e-3
```

**成功标准**：
- T2S 有值（loss < 1e-3）
- L2_err 与 canpinn baseline 相当
- Avg_Step_ms < canpinn 的 Avg_Step_ms（期望 1.5x-2x 加速）

## 来自 2D TGV 的 Lessons Learned（必读！）

### 坑 1：Scale 缩放系数
`Loss = mean(res_u² + res_v² + res_w² + res_div²)` 中，`+` 是 element-wise add，分母是 `N_total`，**不是 `4 * N_total`**。
Backward 中 `scale = 2.0 / N_total * grad_out`。

### 坑 2：3D 边界梯度面更多，必须补齐
2D 有 6 个边界面（t=0/Nt-1, x=0/Nx-1, y=0/Ny-1），3D 多一个 z 维度，有 8 个边界面。
每个面都需要在 `_add_boundary_gradients` 中补齐。写的时候对齐 Python tensor slice 和 Triton kernel 的索引逻辑。

**注意交叉项**：grad_V 的 x=0 边界中，`dRes_v/dv_l` 里 `u_c` 来自 U 场（不是 V 场！），因为 `res_v = ... + u_c*v_x` 中 `u_c` 是 U 值。

### 坑 3：用多种测试场验证梯度
至少测 3 种场：
1. 光滑 sin/cos 场（散度 ≈ 0）
2. **MLP 随机初始化输出场**（这是真实训练场景，最关键）
3. TGV exact 带衰减场

### 坑 4：验证时检查 ALL 点（含边界）
`(gU_pt - gU_tr).abs().max()` 而不是 `(gU_pt[1:-1,1:-1,1:-1,1:-1] - gU_tr[...]).abs().max()`。

### 坑 5：Triton JIT 缓存
修改 kernel 后必须清理缓存：
```bash
rm -rf ~/.triton/cache && find . -name "__pycache__" -type d -exec rm -rf {} +
export TRITON_CACHE_DIR=/tmp/triton_fresh_$$
```

### 坑 6：Gdiv 边界符号
散度伴随的边界符号很容易写反。推导规则：
- `res_div = u_x + v_y + w_z`
- `d(res_div)/d(u_l) = -1/(2dx)`（u 是左邻居时负贡献）
- `d(res_div)/d(u_r) = +1/(2dx)`（u 是右邻居时正贡献）

### 验证检查清单（修完 kernel 后逐条过）
- [ ] Forward loss 与 PyTorch FD 完全一致（rel diff < 1e-12）
- [ ] Interior 梯度误差 < 1e-5（只查 [1:-1,1:-1,1:-1,1:-1]）
- [ ] ALL 点梯度误差（含边界）< 1e-5
- [ ] 用 MLP 输出场再测一遍
- [ ] 用 TGV exact 场再测一遍
- [ ] 500 步小规模训练不发散
- [ ] 全量训练收敛且 L2 合理

## Experiment Loop

LOOP:
1. Phase 1 baseline → 记录结果
2. 写 3D Triton kernel（forward + backward + boundary fix）
3. 用验证检查清单逐条验证
4. `git commit`
5. 端到端训练
6. 如果 T2S 存在且 L2 合理：**DONE**，更新 `results.md`
7. 如果不收敛：诊断问题，修 kernel，回到 3

**Context management**: `/compact` when near limit, re-read this file 和 `kernels/stencil_2d.py`（2D 参考实现）。

**NEVER STOP** until triton 版本端到端收敛且 L2 合理。
