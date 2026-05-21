# Benchmark Results

## 1. 1D Steady Burgers (`burgers_1d_steady`, Nx=256)

### Track 1: Throughput (warmup=50, measure=2950 steps, 5 runs)

| method | median(s) | avg_ms | tput(steps/s) | mem_GB | speedup |
|--------|----------|--------|--------------|--------|---------|
| mlp_vanilla | - | - | - | - | 1.00x |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

### Track 2: Convergence (threshold=1e-4, max=50000 epochs)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | L2_err |
|--------|--------|-------------|------------|------------|--------|
| mlp_vanilla | - | - | - | - | - |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

---

## 2. 1D Unsteady Burgers (`burgers_1d_unsteady`, Nx=1024, Nt=100)

### Track 1: Throughput (warmup=50, measure=2950 steps, 5 runs)

| method | median(s) | avg_ms | tput(steps/s) | mem_GB | speedup |
|--------|----------|--------|--------------|--------|---------|
| mlp_vanilla | 34.27 | 11.62 | 86.1 | 0.904 | 1.00x |
| mlp_canpinn | 15.89 | 5.38 | 185.7 | 0.231 | 2.16x |
| mlp_compile | 19.00 | 6.44 | 155.2 | 0.210 | 1.80x |
| **mlp_triton** | **12.70** | **4.31** | **232.2** | **0.145** | **2.70x** |
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

### Track 2: Convergence (threshold=1e-4, max=200000 epochs)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | L2_err |
|--------|--------|-------------|------------|------------|--------|
| mlp_vanilla | 1082.6 | 87482 | 12.32 | 0.966 | 0.31% |
| mlp_canpinn | 639.2 | 103359 | 6.14 | 0.230 | 0.14% |
| mlp_compile | N/A | 200000 | 6.33 | 0.208 | 0.35% |
| **mlp_triton** | **396.3** | 98670 | **3.98** | **0.145** | 0.22% |
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

**Key findings**:
- Triton T2S 396.3s vs canpinn 639.2s (**1.61x faster**)
- vanilla mem 0.966GB vs triton 0.145GB (**6.7x less**)
- compile never converged to 1e-4 in 200k epochs

---

## 3. 2D LDC (`ldc_2d`, Nx=Ny=64, Re=100, steady-state, NS with P+div)

### Track 1: Throughput (warmup=50, measure=2950 steps, 5 runs)

| method | median(s) | avg_ms | tput(steps/s) | mem_GB | speedup |
|--------|----------|--------|--------------|--------|---------|
| mlp_vanilla | - | - | - | - | 1.00x |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

### Track 2: Convergence (threshold=1e-4, max=50000 epochs)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | L2_Ghia |
|--------|--------|-------------|------------|------------|---------|
| mlp_vanilla | N/A | 10000 | 22.40 | 0.171 | - |
| mlp_canpinn | N/A | 10000 | 7.40 | 0.026 | - |
| mlp_compile | N/A | 10000 | 43.95 | 0.025 | - |
| **mlp_triton** | N/A | 50000 | **4.83** | 0.026 | - |
| cnn_canpinn | N/A | 10000 | 8.17 | 0.048 | - |
| cnn_compile | N/A | 10000 | 45.49 | 0.048 | - |
| cnn_triton | N/A | 10000 | 8.98 | 0.048 | - |

**Note**: None reached threshold 1e-4 within max_epochs (all stuck ~loss 0.16). Model capacity or optimization bottleneck — kernel correctness verified (forward rel diff 0, grad errors < 1e-9). mlp_triton: best avg_step (4.83ms), mlp_compile slowest (43.95ms).
All 7 scripts pass 500-step sanity with P+div (steady NS with pressure and divergence constraint).

---

## 4. 2D Scalar Transport (`transport_2d`, Nx=Ny=64, Nt=20)

### Track 1: Throughput (warmup=50, measure=2950 steps, 5 runs)

| method | median(s) | avg_ms | tput(steps/s) | mem_GB | speedup |
|--------|----------|--------|--------------|--------|---------|
| mlp_vanilla | - | - | - | - | 1.00x |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

### Track 2: Convergence (threshold=1e-4, max=50000 epochs)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | L2_err |
|--------|--------|-------------|------------|------------|--------|
| mlp_vanilla | - | - | - | - | - |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

---

## 5. 2D TGV (`tgv_2d`, Nx=Ny=64, Nt=20)

### Track 1: Throughput (warmup=50, measure=2950 steps, 5 runs)

| method | median(s) | avg_ms | tput(steps/s) | mem_GB | speedup |
|--------|----------|--------|--------------|--------|---------|
| mlp_vanilla | 35.50 | 12.03 | 83.1 | 0.399 | 1.00x |
| mlp_canpinn | 34.76 | 11.78 | 84.9 | 0.400 | 1.02x |
| mlp_compile | 35.97 | 12.19 | 82.0 | 0.401 | 0.99x |
| **mlp_triton** | **22.37** | **7.58** | **131.9** | 0.399 | **1.59x** |
| cnn_canpinn | 31.34 | 10.62 | 94.1 | **0.227** | 1.00x |
| cnn_compile | 34.08 | 11.55 | 86.6 | 0.228 | 0.92x |
| **cnn_triton** | **20.51** | **6.95** | **143.8** | **0.227** | **1.53x** |

### Track 2: Convergence (threshold=1e-4, max=50000 epochs)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | L2_err |
|--------|--------|-------------|------------|------------|--------|
| mlp_vanilla | - | - | - | 7.830 | 2.09% |
| mlp_canpinn | 340.7 | 27071 | 12.55 | 0.399 | 0.45% |
| mlp_compile | - | - | - | - | - |
| **mlp_triton** | **264.9** | 30379 | **8.68** | 0.399 | 0.57% |
| cnn_canpinn | 366.6 | 30036 | 12.16 | 0.227 | 1.13% |
| cnn_compile | - | - | - | - | - |
| **cnn_triton** | **278.5** | 30951 | **8.95** | 0.227 | 1.46% |

**Key findings**:
- Triton backward fully fixed: scale bug + Gdiv boundary sign bugs
- mlp_triton 264.9s vs canpinn 340.7s (**1.29x faster**)
- cnn_triton 278.5s vs canpinn 366.6s (**1.32x faster**)
- mlp_vanilla mem 7.83GB vs triton 0.399GB (**19.6x less**)

---

## 6. 2D Sod Shock Tube (`sod_2d`, Nx=200, Nt=50, compressible Euler)

### Track 1: Throughput (warmup=50, measure=2950 steps, 5 runs)

| method | median(s) | avg_ms | tput(steps/s) | mem_GB | speedup |
|--------|----------|--------|--------------|--------|---------|
| mlp_vanilla | - | - | - | - | 1.00x |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

### Track 2: Convergence (threshold=1e-4, max=50000 epochs)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | L2_err |
|--------|--------|-------------|------------|------------|--------|
| mlp_vanilla | - | - | - | - | - |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

---

## 7. 3D LDC (`ldc_3d`, Nx=Ny=Nz=32, Re=100, steady-state, NS with P+div)

### Track 1: Throughput (warmup=50, measure=2950 steps, 5 runs)

| method | median(s) | avg_ms | tput(steps/s) | mem_GB | speedup |
|--------|----------|--------|--------------|--------|---------|
| mlp_vanilla | - | - | - | - | 1.00x |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

### Track 2: Convergence (threshold=1e-4, max=50000 epochs)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | L2_err |
|--------|--------|-------------|------------|------------|--------|
| mlp_vanilla | N/A | 500* | 49.82 | 2.625 | - |
| mlp_canpinn | N/A | 500* | 13.59 | 0.078 | - |
| mlp_compile | N/A | 500* | 22.88 | 0.145 | - |
| **mlp_triton** | N/A | 50000 | **7.13** | **0.078** | - |
| cnn_canpinn | N/A | 500* | 17.27 | 0.037 | - |
| cnn_compile | N/A | 500* | 24.55 | 0.058 | - |
| cnn_triton | N/A | 500* | **12.52** | **0.037** | - |

**Note**: `*` = 500-step sanity only (not full run). mlp_vanilla OOM risk at 2.6 GB vs triton 0.078 GB (**33.6x less**). mlp_triton ran full 50k epochs (loss stuck at ~0.61).
Triton kernel verified: forward rel diff 0, grad errors < 7e-11 (sin/cos float64), < 2e-11 (MLP float32).
All 7 scripts pass 500-step sanity with P+div (steady NS with pressure and divergence constraint).

---

## 8. 3D TGV (`tgv_3d`, Nx=Ny=Nz=32, Nt=10)

### Track 1: Throughput (warmup=50, measure=2950 steps, 5 runs)

| method | median(s) | avg_ms | tput(steps/s) | mem_GB | speedup |
|--------|----------|--------|--------------|--------|---------|
| mlp_vanilla | - | - | - | - | 1.00x |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

### Track 2: Convergence (threshold=1e-4, max=50000 epochs)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | L2_err |
|--------|--------|-------------|------------|------------|--------|
| mlp_vanilla | - | - | - | - | - |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |
