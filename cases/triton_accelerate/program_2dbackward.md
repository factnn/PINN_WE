# autoresearch-TritonBackward-2D

自动实现 2D NS 方程的 Triton adjoint backward kernel。

## Setup (do once)

```bash
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
cd /share/project/zpy/PINN_WE/cases/triton_accelerate
```

Read `kernels/stencil_2d.py` once at the start.

## Task

**Goal**: 让 `tgv_mlp_triton.py` 和 `tgv_cnn_triton.py` 用完全 Triton forward+backward 跑通训练。

**Target scripts**（实际的训练脚本，修 kernel 的最终目的就是它们能跑通）:
- `baseline/tgv_mlp_triton.py` — MLP + Triton fwd+bwd
- `baseline/tgv_cnn_triton.py` — PhyCNN + Triton fwd+bwd

**Current state**:
- Forward: `ns2d_fwd_kernel` ✓
- Backward: `ns2d_bwd_kernel` 已修复 interior 伴随符号错误，但缺少边界梯度
- `_NS2DTriton.backward()` 和 `_NSTriton.backward()` 已切换为 Triton kernel
- 但训练结果比 PyTorch fallback 差 → 边界梯度缺失是根因

**Success criterion**:
- `tgv_mlp_triton.py` 和 `tgv_cnn_triton.py` 训练正常收敛
- 结果不比 PyTorch fallback 差（可对比 `results.md` 中已有数据）

## Physics: 2D NS Adjoint

Residuals:
```
res_u   = u_t + u*u_x + v*u_y + p_x - ν*(u_xx+u_yy)
res_v   = v_t + u*v_x + v*v_y + p_y - ν*(v_xx+v_yy)
res_div = u_x + v_y
Loss = mean(res_u² + res_v² + res_div²)
```

## Experiment loop

LOOP FOREVER:

1. 修改 `kernels/stencil_2d.py` 中的 `ns2d_bwd_kernel` 或 `_NS2DTriton.backward()`
2. 梯度验证：`CUDA_VISIBLE_DEVICES=<gpu> python kernels/stencil_2d.py`（确保含边界点）
3. `git commit -m "bwd: <description>"`
4. **直接跑目标训练脚本验证**：
   ```bash
   # GPU 0: MLP
   CUDA_VISIBLE_DEVICES=0 python baseline/tgv_mlp_triton.py --max-epochs 50000 --threshold 1e-4 --gpu 0 --runs 1
   # GPU 1: CNN
   CUDA_VISIBLE_DEVICES=1 python baseline/tgv_cnn_triton.py --max-epochs 50000 --threshold 1e-4 --gpu 1 --runs 1
   ```
5. 提取结果：`grep -E "Summary|T2S|Avg_Step|L2" run.log`
6. 如果训练正常收敛且结果不差于 PyTorch fallback：**DONE**，更新 `results.md`
7. 如果没收敛/结果差：分析原因，回到 Step 1

**Context management**: `/compact` when near limit, re-read `kernels/stencil_2d.py`.

**NEVER STOP** until both scripts converge normally.
