# autoresearch-TritonBackward-2D

自动实现 2D NS 方程的 Triton adjoint backward kernel。

## Setup (do once)

```bash
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
cd /share/project/zpy/PINN_WE/cases/triton_accelerate
```

Read `kernels/stencil_2d.py` once at the start.

## Task

**Goal**: 让 `mlp_triton.py` 和 `cnn_triton.py` 用完全 Triton forward+backward 跑通训练。

**Target scripts**:
- `cases/tgv_2d/mlp_triton.py` — MLP + Triton fwd+bwd
- `cases/tgv_2d/cnn_triton.py` — PhyCNN + Triton fwd+bwd

**Success criterion**:
- 训练正常收敛，T2S 有值
- 结果不比 PyTorch fallback 差（可对比 `results.md`）

## Physics: 2D NS Adjoint

```
res_u   = u_t + u*u_x + v*u_y + p_x - ν*(u_xx+u_yy)
res_v   = v_t + u*v_x + v*v_y + p_y - ν*(v_xx+v_yy)
res_div = u_x + v_y
Loss = mean(res_u² + res_v² + res_div²)
```

## 来自实战的 Lessons Learned（必读！）

### 坑 1：Scale 缩放系数
`Loss = mean(res_u² + res_v² + res_div²)` 中，`+` 是 element-wise add，分母是 `N_total`，**不是 `3 * N_total`**。
Backward 中 `scale = 2.0 / N_total * grad_out`。

### 坑 2：边界梯度必须补齐
`ns2d_bwd_kernel` 只处理 interior `[1:Nt-2, 1:Nx-2, 1:Ny-2]`，6 个边界面（t=0/Nt-1, x=0/Nx-1, y=0/Ny-1）的梯度在内核中为 0。
必须写 `_add_boundary_gradients` 函数，用 PyTorch tensor ops 补齐。

### 坑 3：Gdiv 边界符号
散度伴随的 4 个边界符号：
- x=0：`-Gdiv / (2dx)`（u_l 贡献为负）
- x=Nx-1：`+Gdiv / (2dx)`（u_r 贡献为正）
- y=0：`-Gdiv / (2dy)`
- y=Ny-1：`+Gdiv / (2dy)`

### 坑 4：交叉项中 U/V 别搞混
x=0 边界的 grad_V 中，`dRes_v/dv_l` 里 `u_c` 来自 U 场（`res_v = ... + u_c*v_x`），不是 V 场。

### 坑 5：用 MLP 输出场验证梯度
sin/cos 场太光滑，散度接近 0，Gdiv 相关错误会被掩盖。必须用 MLP 随机初始化输出场再测一次。

### 坑 6：验证 ALL 点（含边界）
`(gU_pt - gU_tr).abs().max()` 而不是 `(gU_pt[1:-1,1:-1,1:-1] - gU_tr[...]).abs().max()`。

### 坑 7：Triton JIT 缓存
修改 kernel 后必须清理：
```bash
rm -rf ~/.triton/cache && find . -name "__pycache__" -type d -exec rm -rf {} +
```

### 验证检查清单（修完 kernel 后逐条过）
- [ ] Forward loss 与 PyTorch FD 完全一致（rel diff < 1e-12）
- [ ] Interior 梯度误差 < 1e-5
- [ ] ALL 点梯度误差（含边界）< 1e-5
- [ ] 用 MLP 输出场（非光滑 sin/cos）再测一遍
- [ ] 用 TGV exact 带衰减场再测一遍
- [ ] 500 步小规模训练不发散
- [ ] 全量训练收敛且 L2 合理

## Experiment loop

LOOP FOREVER:

1. 修改 `kernels/stencil_2d.py` 中的 `ns2d_bwd_kernel` 或 `_add_boundary_gradients`
2. 用验证检查清单逐条验证
3. `git commit -m "bwd: <description>"`
4. **直接跑目标训练脚本验证**：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python cases/tgv_2d/mlp_triton.py --max-epochs 50000 --threshold 1e-4 --gpu 0 --runs 1
   CUDA_VISIBLE_DEVICES=1 python cases/tgv_2d/cnn_triton.py --max-epochs 50000 --threshold 1e-4 --gpu 1 --runs 1
   ```
5. 提取结果：`grep -E "Summary|T2S|Avg_Step|L2" run.log`
6. 如果训练正常收敛且结果不差于 PyTorch fallback：**DONE**，更新 `results.md`
7. 如果没收敛/结果差：分析原因，回到 Step 1

**Context management**: `/compact` when near limit, re-read `kernels/stencil_2d.py`.

**NEVER STOP** until both scripts converge normally.

---

## ✅ DONE

- `ns2d_bwd_kernel` 所有伴随符号错误已修复
- Scale bug 修正（2/(3N) → 2/N）
- 4 个 Gdiv 边界符号修正
- `_add_boundary_gradients` 补齐 6 个边界面
- `_NSTriton.backward()` 全部切换为 Triton kernel（无 PyTorch fallback）
- 梯度精度 < 1e-12（sin/cos, MLP, TGV exact 三种场全部通过）
- Track 1 吞吐量：mlp 7.58ms（1.55x），cnn 6.95ms（1.53x）
- Track 2 收敛：mlp 264.9s（1.29x），cnn 278.5s（1.32x）
- 结果已写入 `results.md`
