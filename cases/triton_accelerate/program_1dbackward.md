# autoresearch-TritonBackward-1D

用 Triton backward kernel 跑 1D Burgers 端到端训练，修到收敛正确为止。

## Setup (do once)

```bash
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
cd /share/project/zpy/PINN_WE/cases/triton_accelerate
```

Read these files once:
- `kernels/stencil_1d.py` — Triton forward + backward kernel
- `baseline/burgers_1d_triton.py` — 训练脚本
- `baseline/common_1d.py` — 共用训练逻辑

## Current state

- Triton backward kernel 通过了梯度精度测试（err 2.63e-15）
- **边界梯度 bug 已修复**（burgers_2d_bwd_kernel 只算 interior，边界贡献为 0）
- **T2S=84.2s, L2=0.23%**（cosine LR, lr=2e-3）
- 对比 canpinn 基线：T2S=201.8s, L2=0.18%
- Triton 比 canpinn **快 2.4x**，L2 略高但 < 1%

## Root cause to investigate

1. `burgers_2d_loss_triton_autograd` 和 `loss_canpinn` 计算的 loss 值是否完全一致？
2. Triton kernel 的 forward 是否和 PyTorch 差分算的残差完全一样？（时间差分 + 空间差分 + 非线性项 + BC/IC 的处理方式）
3. 训练脚本里 `loss_fn` 的写法是否正确？目前是：
   ```python
   burgers_2d_loss_triton_autograd(U, dx, dt, nu_val) + 10*ic_loss_from_U(U,X) + 10*bc_loss_from_U(U)
   ```
   而 canpinn 是：
   ```python
   loss_canpinn(model, X, T, dx, dt) + 10*ic_loss_from_U(U,X) + 10*bc_loss_from_U(U)
   ```
   注意 `loss_canpinn` 内部会重新跑 model forward，而 Triton 版直接用 U。

## What to do

### Step 1: 诊断 loss 差异

写一段诊断代码，用同一个 model 和同一个 U，比较：
```python
loss_canpinn_val = loss_canpinn(model, X, T, dx, dt).item()
loss_triton_val = burgers_2d_loss_triton_autograd(U, dx, dt, nu).item()
print(f"canpinn: {loss_canpinn_val}  triton: {loss_triton_val}  diff: {abs(loss_canpinn_val-loss_triton_val)}")
```

如果两者不同，说明 forward 计算方式不一致（比如边界处理、u_t 的差分方式等），需要对齐。

### Step 2: 对齐后重新训练

修改 `kernels/stencil_1d.py` 中的 kernel 或 `baseline/burgers_1d_triton.py` 中的 `loss_fn`，确保 loss 值和 canpinn 完全一致。

### Step 3: 跑端到端训练验证

```bash
rm -f output/burgers_1d/model_triton.pt output/burgers_1d/meta_triton.npy
CUDA_VISIBLE_DEVICES=0 python baseline/burgers_1d_triton.py --max-epochs 200000 --threshold 1e-4 --gpu 0
```

### Step 4: 对比结果

**必须达到**：
- T2S 有值（loss < 1e-4）
- L2 < 1%（和 canpinn 的 0.18% 相当）
- Avg_Step_ms < canpinn 的 6.09ms

**参考基线**（canpinn 已有结果）：
```
T2S=201.8s  Epochs=32917  Avg_Step=6.09ms  Mem=0.230GB  L2=0.18%
```

## Experiment loop

LOOP FOREVER:
1. 诊断 loss 差异（Step 1）
2. 修 kernel 或 loss_fn 对齐
3. `git commit -m "bwd: <description>"`
4. 跑训练验证（Step 3），用 `sleep 600` 等待，不要频繁轮询
5. 提取结果：`grep -E "Summary|T2S|Avg_Step|L2" run.log`
6. 如果 T2S 有值且 L2 < 1%：**DONE**
7. 如果没收敛：分析原因，回到 Step 1

**Context management**: `/compact` when near limit, re-read this file and `kernels/stencil_1d.py`.

**NEVER STOP** until T2S exists and L2 < 1%.
