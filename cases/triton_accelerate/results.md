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

## 2. 1D Unsteady Burgers (`burgers_1d_unsteady`, Nx=1024, Nt=100, MLP 4×50)

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

## 3. 2D LDC (`ldc_2d`, Nx=Ny=64, Re=100, steady-state)

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
| mlp_vanilla | - | - | - | - | - |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

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

## 7. 3D LDC (`ldc_3d`, Nx=Ny=Nz=32, Re=100, steady-state)

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
