#!/usr/bin/env python3
"""
运行全部ablation实验 + 汇总大表

用法:
  python run_all.py           # 运行全部
  python run_all.py --quick   # 快速模式 (只跑spherical gamma=1.4)
"""
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENTS = [
    ("exp7_success_baseline.py", "Our Method (baseline)"),
    ("exp1_soft_constraint.py", "Soft Constraint Only"),
    ("exp2_single_endpoint_shock.py", "Shock-Only Hard Constraint"),
    ("exp2b_single_endpoint_sonic.py", "Sonic-Only Hard Constraint"),
    ("exp3_divided_form_only.py", "Divided-Form Residual Only"),
    ("exp3b_multiplied_form_only.py", "Multiplied-Form Residual Only"),
    ("exp4_no_warmup.py", "No Warmup (alpha free from start)"),
    ("exp5_xi_formulation.py", "Xi-Based 2-Equation"),
    ("exp6_shooting_baseline.py", "Traditional Shooting Method"),
]


def main():
    ablation_dir = Path(__file__).parent
    total_t0 = time.time()

    for script, name in EXPERIMENTS:
        print(f"\n{'#'*70}")
        print(f"# Running: {name}")
        print(f"# Script: {script}")
        print(f"{'#'*70}\n")

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(ablation_dir / script)],
            cwd=str(ablation_dir),
        )
        elapsed = time.time() - t0

        status = "OK" if result.returncode == 0 else f"FAILED (code={result.returncode})"
        print(f"\n  [{name}] {status} in {elapsed:.1f}s")

    total = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"  All experiments done in {total:.0f}s ({total/60:.1f}min)")
    print(f"  Output: {ablation_dir / 'output'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
