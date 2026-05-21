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
| canpinn | **201.8** | 32917 | 6.09 | 0.230 | 0.4 | 0.03% | 0.18% |
| compile | N/A | 200000 | 6.33 | 0.208 | 0.4 | 0.02% | 0.35% |
| **triton** | **76.1** | 15203 | 4.97 | **0.145** | 0.6 | 0.04% | 0.24% |

**Key findings**:
- **triton fastest T2S (76.1s)** with deeper model (depth=6, lr=2e-3 Cosine), **2.65x faster** than canpinn
- Boundary gradient bug fixed (kernel only covered interior), CosineAnnealingLR + depth=6 optimal
- Triton avg_step 4.97ms vs canpinn 6.09ms (**1.23x faster per step**), total time saved by fewer epochs
- vanilla mem 0.966GB vs triton 0.145GB (**6.7x**)
- bw_pct 0.01-0.04% → bottleneck is MatMul, consistent with 2D
- compile never converged to 1e-4 in 200k epochs
---

## 2D TGV (Nx=Ny=64, Nt=20, A100 GPU)

### Track 1: Throughput (GPU 3, warmup=50, measure=2950 steps, 5 runs)

| method | median(s) | avg_ms | tput(steps/s) | mem_GB | bw_GBs | bw_pct | do_bench_ms | speedup |
|--------|----------|--------|--------------|--------|--------|--------|------------|---------|
| mlp_vanilla | 34.06 | 11.55 | 86.6 | 0.399 | 1.3 | 0.1% | 11.38 | 1.00x |
| mlp_canpinn | 33.60 | 11.39 | 87.8 | 0.400 | 1.3 | 0.1% | 11.29 | 1.01x |
| mlp_compile | 34.42 | 11.67 | 85.7 | 0.401 | 1.3 | 0.1% | 11.32 | 0.99x |
| **mlp_triton** | **16.68** | **5.65** | **176.9** | 0.400 | 2.6 | 0.2% | **4.21** | **2.04x** |
| cnn_canpinn | 33.21 | 11.26 | 88.8 | **0.246** | 1.3 | 0.1% | 7.76 | 1.02x |
| cnn_compile | 36.15 | 12.25 | 81.6 | 0.247 | 1.2 | 0.1% | 12.31 | 0.93x |
| **cnn_triton** | **14.12** | **4.79** | **208.9** | **0.246** | **3.1** | 0.2% | **5.47** | **2.35x** |

Note: bw_pct ~0.1% → bottleneck is MatMul (network backward), not memory bandwidth.

### Track 2: Convergence (threshold=1e-4, max=200000 epochs)

| method | T2S(s) | Total_Epochs | Avg_Step_ms | Peak_Mem_GB | Mem_BW_GBs | L2_err |
|--------|--------|-------------|------------|------------|-----------|--------|
| mlp_vanilla | 3615.5 | 45253 | 79.77 | **7.830** | 0.2 | 2.09% |
| mlp_canpinn | 529.2 | 42899 | 12.29 | 0.399 | 1.1 | 0.66% |
| mlp_compile | 838.8 | 40317 | 20.76 | 0.398 | 0.7 | 0.65% |
| mlp_triton | 622.9 | 50316 | 12.34 | 0.399 | 1.1 | 0.65% |
| cnn_canpinn | 795.2 | 62578 | 12.66 | 0.227 | 1.1 | 1.50% |
| cnn_compile | 1233.5 | 64029 | 19.22 | 0.228 | 0.7 | 1.77% |
| cnn_triton | 804.9 | 61053 | 13.14 | 0.227 | 1.0 | 1.62% |

**Key findings (complete)**:
- mlp_vanilla 显存 **7.83 GB**，是 canpinn 的 **19.6x** — autograd 显存爆炸
- mlp_vanilla T2S **3615s**，是 mlp_canpinn 的 **6.8x**
- Track 1 Triton 吞吐量：mlp **2.04x**，cnn **2.35x**（vs vanilla）
- Track 2 cnn_triton T2S 804.9s，与 cnn_canpinn（795.2s）持平，但 kernel 快 **2.35x**
- torch.compile 两个 track 都比 canpinn 慢（1D/2D 一致结论）
- bw_pct 0.1-0.2% → 瓶颈在网络 MatMul，不在 PDE 残差计算

---

## 3D TGV (pending)

| method | kernel_ms | peak_mem(GB) | median_2950steps(s) | T2S(s) | L2_err |
|--------|-----------|-------------|--------------------|----|--------|
| CAN-PINN | - | - | - | - | - |
| CAN-PINN + torch.compile | - | - | - | - | - |
| Triton | - | - | - | - | - |
