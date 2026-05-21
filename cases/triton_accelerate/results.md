# Benchmark Results

## 1D Burgers (Nx=1024, Nt=100, MLP 4×50, A100 GPU)

### Track 1: Throughput (GPU 2, warmup=50, measure=2950 steps, 5 runs)

| method | median(s) | avg_ms | tput(steps/s) | mem_GB | bw_GBs | bw_pct | do_bench_ms | speedup |
|--------|----------|--------|--------------|--------|--------|--------|------------|---------|
| vanilla | 34.27 | 11.62 | 86.1 | 0.904 | 0.212 | 0.014% | 11.77 | 1.00x |
| canpinn | 15.89 | 5.38 | 185.7 | 0.231 | 0.456 | 0.029% | 5.22 | 2.16x |
| compile | 19.00 | 6.44 | 155.2 | 0.210 | 0.381 | 0.025% | 4.33 | 1.80x |
| **triton** | **12.70** | **4.31** | **232.2** | **0.145** | 0.571 | 0.037% | **3.42** | **2.70x** |

### Track 2: Convergence (threshold=1e-4, max=200000 epochs)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Mem_BW_GBs | bw_pct | L2_err |
|--------|--------|-------------|------------|------------|-----------|--------|--------|
| vanilla | 1082.6 | 87482 | 12.32 | 0.966 | 0.2 | 0.01% | 0.31% |
| canpinn | 639.2 | 103359 | 6.14 | 0.230 | 0.4 | 0.03% | 0.14% |
| compile | N/A | 200000 | 6.33 | 0.208 | 0.4 | 0.02% | 0.35% |
| **triton** | 396.3 | 98670 | **3.98** | **0.145** | 0.6 | 0.04% | 0.22% |

**Key findings**:
- Triton T2S 396.3s vs canpinn 639.2s（**1.61x faster**），avg_step 3.98ms vs 6.14ms（**1.54x faster per step**）
- Boundary gradient bug fixed（kernel 只算 interior → Python wrapper 补边界）
- vanilla mem 0.966GB vs triton 0.145GB（**6.7x**）
- bw_pct 0.01-0.04% → bottleneck is MatMul
- compile never converged to 1e-4 in 200k epochs
- CosineAnnealingLR + depth=6 可进一步加速（76.1s），见 experiments log
---

## 2D TGV (Nx=Ny=64, Nt=20, A100 GPU)

### Track 1: Throughput (GPU 0,1,2, warmup=50, measure=2950 steps, 5 runs)

| method | median(s) | avg_ms | tput(steps/s) | mem_GB | bw_GBs | bw_pct | do_bench_ms | speedup |
|--------|----------|--------|--------------|--------|--------|--------|------------|---------|
| mlp_vanilla | 35.50 | 12.03 | 83.1 | 0.399 | 1.2 | 0.1% | 11.86 | 1.00x |
| mlp_canpinn | 34.76 | 11.78 | 84.9 | 0.400 | 1.3 | 0.1% | 11.59 | 1.02x |
| mlp_compile | 35.97 | 12.19 | 82.0 | 0.401 | 1.2 | 0.1% | 12.15 | 0.99x |
| **mlp_triton** | **22.37** | **7.58** | **131.9** | 0.399 | 1.9 | 0.1% | **7.70** | **1.59x** |
| cnn_canpinn | 31.34 | 10.62 | 94.1 | **0.227** | 1.4 | 0.1% | 10.81 | 1.00x |
| cnn_compile | 34.08 | 11.55 | 86.6 | 0.228 | 1.3 | 0.1% | 12.09 | 0.92x |
| **cnn_triton** | **20.51** | **6.95** | **143.8** | **0.227** | 2.1 | 0.1% | **7.06** | **1.53x** |

Note: bw_pct ~0.1% → bottleneck is MatMul (network backward), not memory bandwidth.
Note: Track 1 includes full training step (forward + backward + optimizer). Triton backward now fixed (scale + Gdiv boundary signs).

### Track 2: Convergence (threshold=1e-4, max=50000 epochs)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Mem_BW_GBs | L2_err |
|--------|--------|-------------|------------|------------|-----------|--------|
| mlp_canpinn | 340.7 | 27071 | 12.55 | 0.399 | 1.1 | 0.45% |
| **mlp_triton** | **264.9** | 30379 | **8.68** | 0.399 | 1.6 | 0.57% |
| cnn_canpinn | 366.6 | 30036 | 12.16 | 0.227 | 1.1 | 1.13% |
| **cnn_triton** | **278.5** | 30951 | **8.95** | 0.227 | 1.5 | 1.46% |

Note: mlp_vanilla, mlp_compile, cnn_compile not re-run (unchanged from old data).

**Key findings (complete)**:
- **Triton backward fully fixed and validated**: scale bug (2/(3N) → 2/N) + 4 Gdiv boundary sign bugs
- Gradient errors < 1e-12 on all test fields (sin/cos, MLP, TGV exact)
- Track 2 Triton now converges normally: mlp 264.9s vs canpinn 340.7s (**1.29x faster**), cnn 278.5s vs 366.6s (**1.32x faster**)
- Track 1 Triton throughput: mlp 7.58ms vs canpinn 11.78ms (**1.55x faster**), cnn 6.95ms vs 10.62ms (**1.53x faster**)
- mlp_vanilla 显存 **7.83 GB**（autograd 爆炸），Triton 0.399GB（**19.6x less**）
- torch.compile 两个 track 都比 canpinn 慢（1D/2D 一致结论）
- bw_pct 0.1-1.9% → 瓶颈在网络 MatMul，不在 PDE 残差计算

---

## 3D TGV (pending)

| method | kernel_ms | peak_mem(GB) | median_2950steps(s) | T2S(s) | L2_err |
|--------|-----------|-------------|--------------------|----|--------|
| CAN-PINN | - | - | - | - | - |
| CAN-PINN + torch.compile | - | - | - | - | - |
| Triton | - | - | - | - | - |
