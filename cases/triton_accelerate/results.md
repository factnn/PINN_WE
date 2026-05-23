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

### Scaling Analysis (Kernel Speedup vs Grid Size)

> `python scaling.py --case <case> --gpu <id>`

As grid size increases, PyTorch FD becomes memory-bandwidth-bound while Triton's fused kernel stays compute-efficient. Speedup grows from ~2-3x at small grids to **10-19x at large grids**.

#### burgers_1d_steady (1D, single field)

| Grid | Points | PT (ms) | TR (ms) | Speedup |
|------|--------|---------|---------|---------|
| 256 | 256 | 0.65 | 0.40 | 1.60x |
| 4,096 | 4K | 0.66 | 0.39 | 1.68x |
| 65,536 | 65K | 0.58 | 0.38 | 1.51x |
| 1,048,576 | 1M | 0.60 | 0.39 | 1.53x |
| 4,194,304 | 4M | 1.00 | 0.39 | **2.59x** |

#### burgers_1d_unsteady (2D grid [Nt, Nx], single field)

| Grid | Points | PT (ms) | TR (ms) | Speedup |
|------|--------|---------|---------|---------|
| 50×512 | 25K | 1.03 | 0.09 | 10.98x |
| 200×2048 | 410K | 0.94 | 0.39 | 2.41x |
| 1000×2048 | 2M | 2.21 | 0.40 | **5.59x** |

#### ldc_2d (2D steady, 3 fields U/V/P)

| Grid | Points | PT (ms) | TR (ms) | Speedup |
|------|--------|---------|---------|---------|
| 32² | 1K | 3.34 | 1.33 | 2.52x |
| 128² | 16K | 3.33 | 1.37 | 2.43x |
| 512² | 262K | 3.43 | 1.37 | 2.51x |
| 2048² | 4.2M | 12.45 | 1.35 | **9.22x** |
| 4096² | 16.8M | 54.92 | 5.23 | **10.50x** |

#### ldc_3d (3D steady, 4 fields U/V/W/P)

| Grid | Points | PT (ms) | TR (ms) | Speedup |
|------|--------|---------|---------|---------|
| 32³ | 33K | 8.17 | 2.43 | 3.37x |
| 64³ | 262K | 7.98 | 2.22 | 3.59x |
| 128³ | 2.1M | 13.22 | 2.37 | **5.59x** |
| 192³ | 7.1M | 25.07 | 2.37 | **10.57x** |
| 256³ | 16.8M | 56.00 | 3.10 | **18.08x** |

#### tgv_2d (2D unsteady NS, 3 fields)

| Grid | Points | PT (ms) | TR (ms) | Speedup |
|------|--------|---------|---------|---------|
| 20×64² | 82K | 4.40 | 1.62 | 2.72x |
| 50×256² | 3.3M | 4.99 | 1.55 | 3.22x |
| 100×256² | 6.6M | 12.57 | 1.60 | **7.88x** |
| 200×256² | 13.1M | 24.23 | 2.43 | **9.96x** |
| 100×512² | 26.2M | 46.52 | 4.54 | **10.25x** |

#### tgv_3d (3D unsteady NS, 4 fields)

| Grid | Points | PT (ms) | TR (ms) | Speedup |
|------|--------|---------|---------|---------|
| 10×32³ | 328K | 9.76 | 2.74 | 3.56x |
| 10×64³ | 2.6M | 9.75 | 2.73 | 3.58x |
| 10×96³ | 8.8M | 32.49 | 2.76 | **11.79x** |
| 20×96³ | 17.7M | 67.32 | 3.81 | **17.67x** |
| 20×128³ | 41.9M | 155.18 | 8.10 | **19.16x** |

#### transport_2d (2D advection-diffusion, 1 field)

| Grid | Points | PT (ms) | TR (ms) | Speedup |
|------|--------|---------|---------|---------|
| 20×64² | 82K | 1.68 | 0.51 | 3.32x |
| 50×256² | 3.3M | 1.86 | 0.53 | 3.52x |
| 100×256² | 6.6M | 4.86 | 0.50 | **9.83x** |
| 200×256² | 13.1M | 9.41 | 0.61 | **15.55x** |
| 100×512² | 26.2M | 17.98 | 1.12 | **16.08x** |

#### sod_2d (compressible Euler, 3 fields)

| Grid | Points | PT (ms) | TR (ms) | Speedup |
|------|--------|---------|---------|---------|
| 50×200 | 10K | 1.89 | 1.81 | 1.04x |
| 200×2000 | 400K | 2.16 | 1.69 | 1.28x |
| 1000×4000 | 4M | 3.22 | 1.69 | 1.91x |
| 2000×4000 | 8M | 7.33 | 1.70 | **4.31x** |
| 4000×4000 | 16M | 13.95 | 2.20 | **6.34x** |

#### Summary: Peak Speedup by Case

| Case | Dimensionality | Max Grid Tested | Peak Speedup |
|------|:---:|------|:---:|
| burgers_1d_steady | 1D | 4M pts | 2.59x |
| burgers_1d_unsteady | 1D+T | 2M pts | 5.59x |
| ldc_2d | 2D | 16.8M pts | **10.50x** |
| ldc_3d | 3D | 16.8M pts | **18.08x** |
| tgv_2d | 2D+T | 26.2M pts | **10.25x** |
| tgv_3d | 3D+T | 41.9M pts | **19.16x** |
| transport_2d | 2D+T | 26.2M pts | **16.08x** |
| sod_2d | 1D+T | 16M pts | 6.37x |

---

## 1. 1D Steady Burgers (`burgers_1d_steady`, Nx=1024)

### Track 1: Throughput (warmup=100, measure=2900 steps, 5 runs)

**MLP** (baseline: mlp_vanilla)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| mlp_vanilla | - | - | - | - | 1.00x |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |

**CNN** (baseline: cnn_canpinn)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| cnn_canpinn | - | - | - | - | 1.00x |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

### Track 2: Convergence (threshold=1e-5, max=200000 epochs)

**MLP** (baseline: mlp_vanilla)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| mlp_vanilla | - | - | - | - | - |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |

**CNN** (baseline: cnn_canpinn)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

---

## 2. 1D Unsteady Burgers (`burgers_1d_unsteady`, Nx=1024, Nt=100)

### Track 1: Throughput (warmup=100, measure=2900 steps, 5 runs)

**MLP** (baseline: mlp_vanilla)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| mlp_vanilla | 11.62 | - | 86.1 | 0.904 | 1.00x |
| mlp_canpinn | 5.38 | - | 185.7 | 0.231 | 2.16x |
| mlp_compile | 6.44 | - | 155.2 | 0.210 | 1.80x |
| **mlp_triton** | **4.31** | - | **232.2** | **0.145** | **2.70x** |

**CNN** (baseline: cnn_canpinn)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| cnn_canpinn | - | - | - | - | 1.00x |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

### Track 2: Convergence (threshold=1e-5, max=200000 epochs)

**MLP** (baseline: mlp_vanilla)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| mlp_vanilla | 1082.6 | 87482 | 12.32 | 0.966 | 0.31% |
| mlp_canpinn | 639.2 | 103359 | 6.14 | 0.230 | 0.14% |
| mlp_compile | N/A | 200000 | 6.33 | 0.208 | 0.35% |
| **mlp_triton** | **396.3** | 98670 | **3.98** | **0.145** | 0.22% |

**CNN** (baseline: cnn_canpinn)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

**Key findings**:
- Triton T2S 396.3s vs canpinn 639.2s (**1.61x faster**)
- vanilla mem 0.966GB vs triton 0.145GB (**6.7x less**)
- compile never converged to 1e-5 in 200k epochs

---

## 3. 2D LDC (`ldc_2d`, Nx=Ny=128, Re=100, steady NS with P+div)

### Track 1: Throughput (warmup=100, measure=2900 steps, 5 runs)

**MLP** (baseline: mlp_vanilla)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| mlp_vanilla | - | - | - | - | 1.00x |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |

**CNN** (baseline: cnn_canpinn)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| cnn_canpinn | - | - | - | - | 1.00x |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

### Track 2: Convergence (threshold=1e-5, max=200000 epochs)

**MLP** (baseline: mlp_vanilla)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| mlp_vanilla | - | - | - | - | - |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |

**CNN** (baseline: cnn_canpinn)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |
---

## 4. 2D Scalar Transport (`transport_2d`, Nx=Ny=64, Nt=20)

### Track 1: Throughput (warmup=100, measure=2900 steps, 5 runs)

**MLP** (baseline: mlp_vanilla)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| mlp_vanilla | - | - | - | - | 1.00x |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |

**CNN** (baseline: cnn_canpinn)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| cnn_canpinn | - | - | - | - | 1.00x |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

### Track 2: Convergence (threshold=1e-5, max=200000 epochs)

**MLP** (baseline: mlp_vanilla)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| mlp_vanilla | - | - | - | - | - |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |

**CNN** (baseline: cnn_canpinn)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |
---

## 5. 2D TGV (`tgv_2d`, Nx=Ny=64, Nt=20, unsteady NS)

### Track 1: Throughput (warmup=100, measure=2900 steps, 5 runs)

**MLP** (baseline: mlp_vanilla)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| mlp_vanilla | 12.03 | - | 83.1 | 0.399 | 1.00x |
| mlp_canpinn | 11.78 | - | 84.9 | 0.400 | 1.02x |
| mlp_compile | 12.19 | - | 82.0 | 0.401 | 0.99x |
| **mlp_triton** | **7.58** | - | **131.9** | 0.399 | **1.59x** |

**CNN** (baseline: cnn_canpinn)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| cnn_canpinn | 10.62 | - | 94.1 | 0.227 | 1.00x |
| cnn_compile | 11.55 | - | 86.6 | 0.228 | 0.92x |
| **cnn_triton** | **6.95** | - | **143.8** | 0.227 | **1.53x** |

### Track 2: Convergence (threshold=1e-5, max=200000 epochs)

**MLP** (baseline: mlp_vanilla)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| mlp_vanilla | - | - | - | 7.830 | 2.09% |
| mlp_canpinn | 340.7 | 27071 | 12.55 | 0.399 | 0.45% |
| mlp_compile | - | - | - | - | - |
| **mlp_triton** | **264.9** | 30379 | **8.68** | 0.399 | 0.57% |

**CNN** (baseline: cnn_canpinn)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| cnn_canpinn | 366.6 | 30036 | 12.16 | 0.227 | 1.13% |
| cnn_compile | - | - | - | - | - |
| **cnn_triton** | **278.5** | 30951 | **8.95** | 0.227 | 1.46% |

**Key findings**:
- mlp_triton 264.9s vs canpinn 340.7s (**1.29x faster**)
- cnn_triton 278.5s vs canpinn 366.6s (**1.32x faster**)
- mlp_vanilla mem 7.83GB vs triton 0.399GB (**19.6x less**)
---

## 6. 2D Sod Shock Tube (`sod_2d`, Nx=1000, Nt=200, compressible Euler)

### Track 1: Throughput (warmup=100, measure=2900 steps, 5 runs)

**MLP** (baseline: mlp_vanilla)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| mlp_vanilla | - | - | - | - | 1.00x |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |

**CNN** (baseline: cnn_canpinn)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| cnn_canpinn | - | - | - | - | 1.00x |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

### Track 2: Convergence (threshold=1e-5, max=200000 epochs)

**MLP** (baseline: mlp_vanilla)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| mlp_vanilla | - | - | - | - | - |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |

**CNN** (baseline: cnn_canpinn)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |
---

## 7. 3D LDC (`ldc_3d`, Nx=Ny=Nz=64, Re=100, steady NS with P+div)

### Track 1: Throughput (warmup=100, measure=2900 steps, 5 runs)

**MLP** (baseline: mlp_vanilla)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| mlp_vanilla | - | - | - | - | 1.00x |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |

**CNN** (baseline: cnn_canpinn)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| cnn_canpinn | - | - | - | - | 1.00x |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |

### Track 2: Convergence (threshold=1e-5, max=200000 epochs)

**MLP** (baseline: mlp_vanilla)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| mlp_vanilla | - | - | - | - | - |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |

**CNN** (baseline: cnn_canpinn)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
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

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| mlp_vanilla | - | - | - | - | - |
| mlp_canpinn | - | - | - | - | - |
| mlp_compile | - | - | - | - | - |
| mlp_triton | - | - | - | - | - |
| cnn_canpinn | - | - | - | - | - |
| cnn_compile | - | - | - | - | - |
| cnn_triton | - | - | - | - | - |
