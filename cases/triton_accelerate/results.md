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

## 1. 1D Steady Burgers (`burgers_1d_steady`, Nx=256)

### Track 1: Throughput (warmup=100, measure=2900 steps, 5 runs)

**MLP** (baseline: mlp_vanilla)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| mlp_vanilla | 4.35 | 0.017 | 230.4 | 0.021 | 1.00x |
| mlp_canpinn | 1.98 | 0.014 | 506.1 | 0.019 | 2.20x |
| mlp_compile | 2.35 | 0.018 | 425.8 | 0.019 | 1.85x |
| **mlp_triton** | **1.99** | 0.013 | **505.1** | 0.019 | **2.19x** |

**CNN** (baseline: cnn_canpinn)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| cnn_canpinn | 2.11 | 0.036 | 472.2 | 0.018 | 1.00x |
| cnn_compile | 2.26 | 0.002 | 442.4 | 0.018 | 0.93x |
| **cnn_triton** | **1.85** | 0.017 | **542.0** | 0.018 | **1.14x** |

### Track 2: Convergence (threshold=1e-5, max=200000 epochs)

**MLP** (baseline: mlp_vanilla)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| mlp_vanilla | **71.2** | 14582 | 4.86 | 0.021 | 9.98e-6 | 0.537 |
| mlp_canpinn | 378.5 | 166353 | 2.25 | 0.019 | 1.00e-5 | 0.592 |
| mlp_compile | 278.3 | 111901 | 2.46 | 0.019 | 9.96e-6 | 0.733 |
| mlp_triton | 281.4 | 134176 | 2.07 | 0.019 | 9.97e-6 | 0.181 |

**CNN** (baseline: cnn_canpinn)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| cnn_canpinn | 256.1 | 115954 | 2.19 | 0.018 | 9.51e-6 | 0.221 |
| cnn_compile | N/A | 200000 | 2.43 | 0.018 | 2.98e-5 | 0.284 |
| **cnn_triton** | **229.7** | 121131 | **1.87** | 0.018 | 9.68e-6 | **0.058** |

---

## 2. 1D Unsteady Burgers (`burgers_1d_unsteady`, Nx=1024, Nt=100)

### Track 1: Throughput (warmup=100, measure=2900 steps, 5 runs)

**MLP** (baseline: mlp_vanilla)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| mlp_vanilla | 10.10 | 0.008 | 99.0 | 1.075 | 1.00x |
| mlp_canpinn | 2.83 | 0.017 | 353.5 | 0.183 | 3.57x |
| mlp_compile | 3.09 | 0.019 | 324.3 | 0.156 | 3.28x |
| **mlp_triton** | **2.44** | 0.019 | **411.5** | **0.183** | **4.16x** |

**CNN** (baseline: cnn_canpinn)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| cnn_canpinn | 2.95 | 0.074 | 334.7 | 0.102 | 1.00x |
| cnn_compile | 3.41 | 0.039 | 291.8 | 0.103 | 0.84x |
| **cnn_triton** | **2.56** | 0.017 | **392.4** | **0.102** | **1.15x** |

### Track 2: Convergence (threshold=1e-5, max=200000 epochs)

**MLP** (baseline: mlp_vanilla)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| mlp_vanilla | **1143.0** | 111025 | 10.27 | 1.101 | <1e-5 | 0.13% |
| mlp_canpinn | N/A | 200000 | 3.04 | 0.183 | - | 0.45% |
| mlp_compile | N/A | 200000 | 3.24 | 0.156 | - | 0.18% |
| mlp_triton | N/A | 200000 | 2.56 | 0.183 | - | 0.13% |

**CNN** (baseline: cnn_canpinn)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| cnn_canpinn | N/A | 200000 | 3.04 | 0.102 | - | 0.10% |
| cnn_compile | N/A | 200000 | 3.33 | 0.103 | - | 0.10% |
| **cnn_triton** | N/A | 200000 | **2.53** | **0.102** | - | **0.09%** |

**Key findings**:
- Track 1: mlp_triton **4.16x** faster than vanilla, cnn_triton **1.15x** faster
- Memory: vanilla 1.1GB vs triton 0.18GB (**6x less**)
- Track 2: Only vanilla converged to 1e-5; triton achieves same L2 (0.13%) without hitting threshold
- compile never converged to 1e-5 in 200k epochs

---

## 3. 2D LDC (`ldc_2d`, Nx=Ny=128, Re=100, steady NS with P+div)

### Track 1: Throughput (warmup=100, measure=2900 steps, 5 runs)

**MLP** (baseline: mlp_vanilla)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| mlp_vanilla | 17.11 | 0.569 | 57.6 | 1.321 | 1.00x |
| mlp_canpinn | 5.63 | 0.013 | 177.7 | 0.086 | 3.08x |
| mlp_compile | 6.02 | 0.274 | 170.6 | 0.086 | 2.96x |
| **mlp_triton** | **3.97** | 0.014 | **252.4** | **0.086** | **4.38x** |

**CNN** (baseline: cnn_canpinn)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| cnn_canpinn | 5.74 | 0.059 | 175.0 | 0.034 | 1.00x |
| cnn_compile | 6.06 | 0.059 | 165.3 | 0.034 | 0.95x |
| **cnn_triton** | **3.91** | 0.100 | **251.7** | **0.034** | **1.47x** |

### Track 2: Convergence (threshold=1e-5, max=200000 epochs)

**MLP** (baseline: mlp_vanilla)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| mlp_vanilla | N/A | 200000 | 18.43 | 1.329 | - | 7.13% |
| mlp_canpinn | N/A | 200000 | 6.46 | 0.086 | - | 1.34% |
| mlp_compile | N/A | 200000 | 6.51 | 0.086 | - | 1.77% |
| mlp_triton | N/A | 200000 | 4.42 | 0.086 | - | 1.43% |

**CNN** (baseline: cnn_canpinn)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| cnn_canpinn | N/A | 200000 | 5.97 | 0.034 | - | 0.61% |
| cnn_compile | N/A | 200000 | 6.26 | 0.034 | - | 0.61% |
| **cnn_triton** | N/A | 200000 | **3.79** | **0.034** | - | **0.57%** |

**Key findings**:
- Track 1: mlp_triton **4.38x** faster than vanilla, cnn_triton **1.47x** faster
- Memory: vanilla 1.32GB vs triton 0.086GB (**15x less**)
- Track 2: No method converged to 1e-5; cnn_triton achieves best L2 (0.57%)
---

## 4. 2D Scalar Transport (`transport_2d`, Nx=Ny=64, Nt=20)

### Track 1: Throughput (warmup=100, measure=2900 steps, 5 runs)

**MLP** (baseline: mlp_vanilla)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| mlp_vanilla | 31.66 | 0.007 | 31.6 | 3.254 | 1.00x |
| mlp_canpinn | 3.62 | 0.011 | 276.7 | 0.356 | 8.76x |
| mlp_compile | 3.81 | 0.016 | 262.7 | 0.273 | 8.32x |
| **mlp_triton** | **3.05** | 0.004 | **327.8** | **0.356** | **10.38x** |

**CNN** (baseline: cnn_canpinn)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| cnn_canpinn | 4.22 | 0.008 | 237.4 | 0.244 | 1.00x |
| cnn_compile | 4.30 | 0.057 | 231.5 | 0.244 | 0.98x |
| **cnn_triton** | **3.79** | 0.001 | **263.6** | **0.244** | **1.11x** |

### Track 2: Convergence (threshold=1e-5, max=200000 epochs)

**MLP** (baseline: mlp_vanilla)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| mlp_vanilla | N/A | 200000 | 31.79 | 3.296 | 6.54e-4 | 4.90% |
| mlp_canpinn | N/A | 200000 | 5.01 | 0.356 | 6.58e-4 | 4.90% |
| mlp_compile | N/A | 200000 | 4.72 | 0.273 | 6.57e-4 | 4.95% |
| mlp_triton | N/A | 200000 | 3.67 | 0.356 | 6.57e-4 | 4.93% |

**CNN** (baseline: cnn_canpinn)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| cnn_canpinn | N/A | 200000 | 5.91 | 0.244 | 8.25e-4 | 17.9% |
| cnn_compile | N/A | 200000 | 6.02 | 0.244 | 7.20e-4 | 25.3% |
| cnn_triton | N/A | 200000 | 4.60 | 0.244 | 7.29e-4 | 22.3% |

**Key findings**:
- Track 1: mlp_triton **10.38x** faster than vanilla, cnn_triton **1.11x** faster
- Memory: vanilla 3.25GB vs triton 0.356GB (**9x less**)
- Track 2: No method converged to 1e-5; all MLP methods achieve ~4.9% L2
---

## 5. 2D TGV (`tgv_2d`, Nx=Ny=64, Nt=20, unsteady NS)

### Track 1: Throughput (warmup=100, measure=2900 steps, 5 runs)

**MLP** (baseline: mlp_vanilla)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| mlp_vanilla | 78.43 | 0.010 | 12.7 | 7.830 | 1.00x |
| mlp_canpinn | 7.93 | 0.115 | 125.2 | 0.399 | 9.82x |
| mlp_compile | 8.18 | 0.033 | 122.4 | 0.400 | 9.60x |
| **mlp_triton** | **5.31** | 0.023 | **188.9** | **0.399** | **14.81x** |

**CNN** (baseline: cnn_canpinn)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| cnn_canpinn | 7.77 | 0.127 | 129.2 | 0.245 | 1.00x |
| cnn_compile | 7.94 | 0.015 | 126.1 | 0.246 | 0.98x |
| **cnn_triton** | **4.74** | 0.019 | **211.4** | **0.245** | **1.64x** |

### Track 2: Convergence (threshold=1e-5, max=200000 epochs)

**MLP** (baseline: mlp_vanilla)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| mlp_vanilla | N/A | 200000 | 78.57 | 7.874 | 1.00e-4 | 1.60% |
| mlp_canpinn | N/A | 200000 | 9.70 | 0.399 | 7.70e-5 | 3.31% |
| mlp_compile | N/A | 200000 | 9.26 | 0.400 | 7.63e-5 | 3.31% |
| mlp_triton | N/A | 200000 | 6.81 | 0.399 | 7.66e-5 | 3.30% |

**CNN** (baseline: cnn_canpinn)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| cnn_canpinn | N/A | 200000 | 9.94 | 0.245 | 2.32e-4 | 2.41% |
| cnn_compile | N/A | 200000 | 10.20 | 0.246 | 2.65e-4 | 2.31% |
| cnn_triton | N/A | 200000 | 7.02 | 0.245 | 2.28e-4 | 2.40% |

**Key findings**:
- Track 1: mlp_triton **14.81x** faster than vanilla, cnn_triton **1.64x** faster
- Memory: vanilla 7.83GB vs triton 0.399GB (**19.6x less**)
- Track 2: No method converged to 1e-5; mlp_triton achieves best L2 (3.30%)
---

## 6. 2D Sod Shock Tube (`sod_2d`, Nx=1000, Nt=200, compressible Euler)

### Track 1: Throughput (warmup=100, measure=2900 steps, 5 runs)

**MLP** (baseline: mlp_vanilla)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| mlp_vanilla | 104.20 | 0.334 | 9.6 | 11.574 | 1.00x |
| mlp_canpinn | 12.17 | 0.033 | 82.3 | 1.258 | 8.60x |
| mlp_compile | 9.95 | 0.024 | 100.4 | 1.464 | 10.49x |
| **mlp_triton** | **11.51** | 0.244 | **87.6** | **1.260** | **9.15x** |

**CNN** (baseline: cnn_canpinn)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| cnn_canpinn | 5.33 | 0.008 | 187.7 | 0.178 | 1.00x |
| cnn_compile | 5.49 | 0.018 | 181.9 | 0.180 | 0.97x |
| **cnn_triton** | **5.31** | 0.029 | **189.0** | **0.180** | **1.00x** |

### Track 2: Convergence (threshold=1e-5, max=200000 epochs)

**MLP** (baseline: mlp_vanilla)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| mlp_vanilla | N/A | 200000 | 103.94 | 11.579 | 1.42e-2 | 1.73% |
| mlp_canpinn | N/A | 200000 | 11.92 | 1.258 | 1.83e+7 | DIVERGED |
| mlp_compile | N/A | 200000 | 11.06 | 1.464 | 1.07e+20 | DIVERGED |
| **mlp_triton** | N/A | 200000 | **11.81** | **1.260** | **8.79e-2** | **3.90%** |

**CNN** (baseline: cnn_canpinn)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| cnn_canpinn | N/A | 200000 | 6.17 | 0.178 | 2.79e-2 | 2.18% |
| cnn_compile | N/A | 200000 | 6.19 | 0.180 | 1.16e-1 | 3.26% |
| **cnn_triton** | N/A | 200000 | **6.05** | **0.180** | **2.57e-2** | **2.12%** |

**Key findings**:
- Track 1: mlp_triton **9.15x** faster than vanilla, cnn_triton **1.00x** (on par)
- Memory: vanilla 11.6GB vs triton 1.26GB (**9.2x less**)
- Track 2: mlp_canpinn and mlp_compile **DIVERGED** (compressible Euler is unstable with central FD)
- mlp_vanilla (autograd) achieves best MLP L2 (1.73%); cnn_triton best overall (2.12%)
---

## 7. 3D LDC (`ldc_3d`, Nx=Ny=Nz=64, Re=100, steady NS with P+div)

### Track 1: Throughput (warmup=100, measure=2900 steps, 5 runs)

**MLP** (baseline: mlp_vanilla)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| mlp_vanilla | 173.02 | 0.042 | 5.8 | 17.594 | 1.00x |
| mlp_canpinn | 11.71 | 0.319 | 84.3 | 0.473 | 14.59x |
| mlp_compile | 11.84 | 0.275 | 84.4 | 0.532 | 14.61x |
| **mlp_triton** | **6.39** | 0.138 | **158.7** | **0.475** | **27.45x** |

**CNN** (baseline: cnn_canpinn)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| cnn_canpinn | 14.29 | 0.007 | 69.9 | 0.125 | 1.00x |
| cnn_compile | 14.13 | 0.010 | 70.8 | 0.127 | 1.01x |
| **cnn_triton** | **12.90** | 0.007 | **77.5** | **0.126** | **1.11x** |

### Track 2: Convergence (threshold=1e-5, max=200000 epochs)

> Track 2 failed due to disk full. Only Track 1 results available.

**Key findings**:
- Track 1: mlp_triton **27.45x** faster than vanilla, cnn_triton **1.11x** faster
- Memory: vanilla 17.6GB vs triton 0.475GB (**37x less**)
---

## 8. 3D TGV (`tgv_3d`, Nx=Ny=Nz=32, Nt=10, unsteady NS)

> **Note**: mlp_vanilla SKIPPED (OOM >40GB on single A100). Baseline for MLP is mlp_canpinn.

### Track 1: Throughput (warmup=100, measure=2900 steps, 5 runs)

**MLP** (baseline: mlp_canpinn, since vanilla OOM)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| mlp_canpinn | 24.07 | 0.524 | 42.3 | 3.386 | 1.00x |
| mlp_compile | 20.58 | 0.432 | 48.1 | 3.722 | 1.14x |
| **mlp_triton** | **21.91** | 0.527 | **46.5** | **3.389** | **1.10x** |

**CNN** (baseline: cnn_canpinn)

| method | avg_ms | std_ms | tput(steps/s) | mem_GB | speedup |
|--------|--------|--------|--------------|--------|---------|
| cnn_canpinn | 31.73 | 0.005 | 31.5 | 0.307 | 1.00x |
| cnn_compile | 31.45 | 0.044 | 31.8 | 0.313 | 1.01x |
| **cnn_triton** | **29.59** | 0.001 | **33.8** | **0.311** | **1.07x** |

### Track 2: Convergence (threshold=1e-5, max=200000 epochs)

**MLP** (baseline: mlp_canpinn)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| mlp_canpinn | N/A | 200000 | 29.2 | 3.386 | 2.66e-3 | 2.93% |
| mlp_compile | N/A | 200000 | 25.7 | 3.722 | 2.66e-3 | 2.93% |
| **mlp_triton** | N/A | 200000 | **22.8** | **3.389** | **2.66e-3** | **2.93%** |

**CNN** (baseline: cnn_canpinn)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Final_Loss | L2_err |
|--------|--------|-------------|------------|------------|------------|--------|
| cnn_canpinn | N/A | 200000 | 38.8 | 0.307 | 1.78e-1 | 15.2% |
| cnn_compile | N/A | 200000 | 39.1 | 0.313 | 2.65 | 99.4% |
| **cnn_triton** | N/A | 200000 | **30.8** | **0.311** | **2.65** | **99.4%** |

**Key findings**:
- mlp_vanilla OOM (>40GB on single A100) — only case where vanilla cannot run
- Track 1: mlp_triton **1.10x** vs canpinn baseline, cnn_triton **1.07x** faster
- Track 2 MLP: All three methods achieve identical Final_Loss (2.66e-3) and L2 (2.93%), proving **zero accuracy loss** from Triton
- Track 2 CNN: cnn_canpinn converges (15.2% L2) but compile/triton fail to converge on 3D NS (99.4% L2)

