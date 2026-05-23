# Benchmark Results

> Run with: `python run.py --case <case> --track <0|1|2> --gpu <id>`

---

## Track 0: Kernel-Level Speedup (PyTorch FD vs Triton Fused Kernel)

Pure PDE residual computation on random fields. No model forward/backward — isolates kernel performance.

| Case | Grid | PyTorch FD (ms) | Triton (ms) | Fwd Speedup | Bwd Speedup | **Total Speedup** |
|------|------|:-:|:-:|:-:|:-:|:-:|
| burgers_1d_steady | 256 | 0.98 | 0.59 | 1.41x | 1.75x | **1.68x** |
| burgers_1d_unsteady | 1024×100 | 1.71 | 1.14 | 1.59x | 1.47x | **1.49x** |
| ldc_2d | 64×64 | 4.76 | 2.53 | 2.64x | 1.74x | **1.88x** |
| ldc_3d | 32³ | 10.58 | 5.34 | 3.10x | 1.79x | **1.98x** |
| transport_2d | 64×64×20 | 2.60 | 1.34 | 2.70x | 1.80x | **1.94x** |
| tgv_2d | 64×64×20 | 6.55 | 3.11 | 2.33x | 2.05x | **2.11x** |
| tgv_3d | 32³×10 | 13.87 | 5.52 | 2.99x | 2.41x | **2.51x** |
| sod_2d | 200×50 | 4.06 | — | — | — | N/A (no kernel) |

**Key findings**:
- Forward speedup: 1.4x (1D) → 2.7x (2D) → 3.1x (3D)
- Backward speedup: 1.5–2.4x (adjoint kernel vs autograd-traced FD)
- Largest gain on 3D TGV: **2.51x total** (most stencil ops to fuse)
- All kernels use `@triton.autotune` with BLOCK in [64, 128, 256, 512]

---

## 1. 1D Steady Burgers (`burgers_1d_steady`, Nx=256)

### Track 1: Throughput (warmup=100, measure=2900 steps, 5 runs)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| mlp_vanilla | - | - | - | - | 1.00x |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

### Track 2: Convergence (threshold=1e-5, max=200000 epochs)

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

### Track 1: Throughput (warmup=100, measure=2900 steps, 5 runs)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| mlp_vanilla | 11.62 | - | 86.1 | 0.904 | 1.00x |
| mlp_canpinn | 5.38 | - | 185.7 | 0.231 | 2.16x |
| mlp_compile | 6.44 | - | 155.2 | 0.210 | 1.80x |
| **mlp_triton** | **4.31** | - | **232.2** | **0.145** | **2.70x** |
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

### Track 2: Convergence (threshold=1e-5, max=200000 epochs)

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
- compile never converged to 1e-5 in 200k epochs

---

## 3. 2D LDC (`ldc_2d`, Nx=Ny=64, Re=100, steady NS with P+div)

### Track 1: Throughput (warmup=100, measure=2900 steps, 5 runs)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| mlp_vanilla | - | - | - | - | 1.00x |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

### Track 2: Convergence (threshold=1e-5, max=200000 epochs)

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

### Track 1: Throughput (warmup=100, measure=2900 steps, 5 runs)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| mlp_vanilla | - | - | - | - | 1.00x |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

### Track 2: Convergence (threshold=1e-5, max=200000 epochs)

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

## 5. 2D TGV (`tgv_2d`, Nx=Ny=64, Nt=20, unsteady NS)

### Track 1: Throughput (warmup=100, measure=2900 steps, 5 runs)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| mlp_vanilla | 12.03 | - | 83.1 | 0.399 | 1.00x |
| mlp_canpinn | 11.78 | - | 84.9 | 0.400 | 1.02x |
| mlp_compile | 12.19 | - | 82.0 | 0.401 | 0.99x |
| **mlp_triton** | **7.58** | - | **131.9** | 0.399 | **1.59x** |
| cnn_canpinn | 10.62 | - | 94.1 | 0.227 | 1.00x |
| cnn_compile | 11.55 | - | 86.6 | 0.228 | 0.92x |
| **cnn_triton** | **6.95** | - | **143.8** | 0.227 | **1.53x** |

### Track 2: Convergence (threshold=1e-5, max=200000 epochs)

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
- mlp_triton 264.9s vs canpinn 340.7s (**1.29x faster**)
- cnn_triton 278.5s vs canpinn 366.6s (**1.32x faster**)
- mlp_vanilla mem 7.83GB vs triton 0.399GB (**19.6x less**)

---

## 6. 2D Sod Shock Tube (`sod_2d`, Nx=200, Nt=50, compressible Euler)

### Track 1: Throughput (warmup=100, measure=2900 steps, 5 runs)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| mlp_vanilla | - | - | - | - | 1.00x |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | N/A (no kernel) |
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | N/A (no kernel) |

### Track 2: Convergence (threshold=1e-5, max=200000 epochs)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | L2_err |
|--------|--------|-------------|------------|------------|--------|
| mlp_vanilla | - | - | - | - | - |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |

---

## 7. 3D LDC (`ldc_3d`, Nx=Ny=Nz=32, Re=100, steady NS with P+div)

### Track 1: Throughput (warmup=100, measure=2900 steps, 5 runs)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| mlp_vanilla | - | - | - | - | 1.00x |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

### Track 2: Convergence (threshold=1e-5, max=200000 epochs)

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

## 8. 3D TGV (`tgv_3d`, Nx=Ny=Nz=32, Nt=10, unsteady NS)

### Track 1: Throughput (warmup=100, measure=2900 steps, 5 runs)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| mlp_vanilla | - | - | - | - | 1.00x |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

### Track 2: Convergence (threshold=1e-5, max=200000 epochs)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | L2_err |
|--------|--------|-------------|------------|------------|--------|
| mlp_vanilla | - | - | - | - | - |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |
