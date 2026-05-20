# autoresearch-TritonBackward-2D

自动实现 2D NS 方程的 Triton adjoint backward kernel。先完成 1D（program_1dbackward.md），再做这个。

## Setup (do once)

```bash
source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
cd /share/project/zpy/PINN_WE/cases/triton_accelerate
git checkout -b autoresearch/triton-bwd-2d-$(date +%Y%m%d) 2>/dev/null || true
```

Read `kernels/stencil_2d.py` once at the start.

## Task

**Goal**: Replace PyTorch fallback in `_NS2DTriton.backward()` with correct Triton kernel `ns2d_bwd_kernel`.

**Current state**:
- Forward: `ns2d_fwd_kernel` ✓ (correct, 30x speedup)
- Backward: PyTorch fallback (correct but slow)
- `ns2d_bwd_kernel` exists but has grad errors ~2.7e-3

**Success criterion**: `Grad U/V/P err < 1e-5`

Test:
```bash
CUDA_VISIBLE_DEVICES=0 python kernels/stencil_2d.py
```

## Physics: 2D NS Adjoint

Residuals:
```
res_u   = u_t + u*u_x + v*u_y + p_x - ν*(u_xx+u_yy)
res_v   = v_t + u*v_x + v*v_y + p_y - ν*(v_xx+v_yy)
res_div = u_x + v_y
Loss = mean(res_u² + res_v² + res_div²)
```

Upstream gradients: `Gu=2*res_u/N`, `Gv=2*res_v/N`, `Gdiv=2*res_div/N`

**grad_U[t,i,j]**:
```
# u_t adjoint (central diff in t):
+= Gu[t-1,i,j]/(2dt) - Gu[t+1,i,j]/(2dt)

# u_c * u_x (u_c part, at point [t,i,j]):
+= Gu[t,i,j] * (u[t,i+1,j]-u[t,i-1,j])/(2dx)

# u_x used by neighbor's u_c*u_x (u[i] is u_xp for i-1, u_xm for i+1):
+= Gu[t,i-1,j] * u[t,i-1,j]/(2dx)
-= Gu[t,i+1,j] * u[t,i+1,j]/(2dx)

# v*u_y (u_y part, u[j] is u_yp for j-1, u_ym for j+1):
+= Gu[t,i,j-1] * v[t,i,j-1]/(2dy)
-= Gu[t,i,j+1] * v[t,i,j+1]/(2dy)

# u_xx adjoint: -ν*(Gu[i+1]+Gu[i-1]-2*Gu[i])/dx²
+= -ν*(Gu[t,i+1,j]+Gu[t,i-1,j]-2*Gu[t,i,j])/dx²

# u_yy adjoint:
+= -ν*(Gu[t,i,j+1]+Gu[t,i,j-1]-2*Gu[t,i,j])/dy²

# res_div u_x adjoint:
+= Gdiv[t,i-1,j]/(2dx) - Gdiv[t,i+1,j]/(2dx)
```

**grad_V[t,i,j]** (symmetric, swap u↔v, x↔y roles):
```
+= Gv[t-1,i,j]/(2dt) - Gv[t+1,i,j]/(2dt)
+= Gv[t,i,j] * (v[t,i+1,j]-v[t,i-1,j])/(2dx)
+= Gv[t,i-1,j]*v[t,i-1,j]/(2dx) - Gv[t,i+1,j]*v[t,i+1,j]/(2dx)
+= Gv[t,i,j-1]*u[t,i,j-1]/(2dy) - Gv[t,i,j+1]*u[t,i,j+1]/(2dy)
+= Gu[t,i,j] * (u[t,i,j+1]-u[t,i,j-1])/(2dy)   # v_c in v*u_y of res_u
+= -ν*(Gv[t,i+1,j]+Gv[t,i-1,j]-2*Gv[t,i,j])/dx²
+= -ν*(Gv[t,i,j+1]+Gv[t,i,j-1]-2*Gv[t,i,j])/dy²
+= Gdiv[t,i,j-1]/(2dy) - Gdiv[t,i,j+1]/(2dy)
```

**grad_P[t,i,j]**:
```
= (Gu[t,i-1,j]-Gu[t,i+1,j])/(2dx) + (Gv[t,i,j-1]-Gv[t,i,j+1])/(2dy)
```

## What to modify

Only `ns2d_bwd_kernel` in `kernels/stencil_2d.py`. Fix term by term using physics above.

## Experiment loop

LOOP FOREVER:
1. Run test, check which grad (U/V/P) has largest error
2. Compare current kernel with physics formulas above, find wrong term
3. Fix one term, `git commit -m "bwd: fix <term>"`
4. Run test again
5. If all < 1e-5: switch `_NS2DTriton.backward()` to use Triton kernel, verify training

**Context management**: `/compact` when near limit, re-read `kernels/stencil_2d.py`.

**NEVER STOP** until done.

---

## ✅ DONE

- `ns2d_bwd_kernel` 所有伴随符号错误已修复
- `_NS2DTriton.backward()` 已用 Triton kernel 替换 PyTorch fallback
- 梯度精度：gradU 1.8e-11, gradV 1.8e-11, gradP 3.3e-11（目标 < 1e-5）
- `_NSTriton` (TGV 无 P) 也已切换为 Triton backward
- TGV 训练验证：Triton 137s loss=0.218 vs canpinn 275s loss=0.270（2x 加速 + 更好精度）
- 多分辨率测试全部 PASS（6x16x16 ~ 20x64x64）
