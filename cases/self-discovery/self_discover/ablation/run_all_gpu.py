#!/usr/bin/env python3
"""
在4张GPU上并行运行所有ablation实验。
每张GPU分配2个实验，同时跑。
"""
import subprocess
import sys
import time
import os
from pathlib import Path

# (script, GPU_id)
# 每张GPU分2个实验
EXPERIMENTS = [
    # GPU 0
    ("exp7_success_baseline.py", 0),
    ("exp1_soft_constraint.py", 0),
    # GPU 1
    ("exp2_single_endpoint_shock.py", 1),
    ("exp2b_single_endpoint_sonic.py", 1),
    # GPU 2
    ("exp3_divided_form_only.py", 2),
    ("exp3b_multiplied_form_only.py", 2),
    # GPU 3
    ("exp4_no_warmup.py", 3),
    ("exp5_xi_formulation.py", 3),
]

ablation_dir = Path(__file__).parent
log_dir = ablation_dir / "logs"
log_dir.mkdir(exist_ok=True)

processes = []
t0 = time.time()

for script, gpu_id in EXPERIMENTS:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_file = log_dir / f"{script.replace('.py', '.log')}"
    print(f"  Launching {script} on GPU {gpu_id}")
    p = subprocess.Popen(
        [sys.executable, str(ablation_dir / script), "--device", "cuda"],
        env=env,
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        cwd=str(ablation_dir),
    )
    processes.append((script, p, log_file))

print(f"\n  {len(processes)} experiments launched on 4 GPUs")
print(f"  Logs: {log_dir}/")
print(f"  Waiting for completion...\n")

# Wait for all
for script, p, log_file in processes:
    p.wait()
    status = "OK" if p.returncode == 0 else f"FAILED(rc={p.returncode})"
    elapsed = time.time() - t0
    print(f"  [{elapsed:6.0f}s] {script:<40} {status}")

total = time.time() - t0
print(f"\n  All done in {total:.0f}s ({total/60:.1f}min)")
print(f"  Output plots: {ablation_dir / 'output'}/")

# Also run shooting baseline (CPU-only, fast)
print(f"\n  Running shooting baseline (CPU)...")
subprocess.run([sys.executable, str(ablation_dir / "exp6_shooting_baseline.py")],
               cwd=str(ablation_dir))
