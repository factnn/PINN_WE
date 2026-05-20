#!/usr/bin/env python3
"""
Exp7: 成功方案的复现 — 作为所有ablation的参照基准

直接调用 pinn_self_discover.py 的逻辑，记录详细数据。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import json
from pinn_self_discover import train, GEOMETRY_N
from common import DEFAULT_CONFIGS, get_ref_alpha, plot_ablation_result, print_summary_table, ALPHA_INIT


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    device = args.device

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Exp7: Success Baseline (dual-endpoint + mixed residual + warmup)")
    print("=" * 70)

    results = {}
    for gamma, geo in DEFAULT_CONFIGS:
        n = GEOMETRY_N[geo]
        key = f"g{gamma:.2f}_{geo}"
        print(f"\n{'='*60}")
        print(f"  gamma={gamma}, {geo}")
        print(f"{'='*60}")

        result = train(gamma, n, geo, alpha_init=ALPHA_INIT[(gamma, n)], epochs=10000, device=device)
        results[key] = {
            "alpha_fit": result["alpha_fit"],
            "ref_alpha": result["ref_alpha"],
            "rel_err_pct": result["rel_err_pct"],
            "final_loss": result["final_loss"],
            "alpha_history": result["alpha_history"],
            "loss_history": result["loss_history"],
        }

    print_summary_table(results, "Exp7: Success Baseline (our method)")
    plot_ablation_result(results, "Exp7: Our Method (dual-endpoint + mixed residual + warmup)",
                         str(out_dir / "exp7_success_baseline.png"))

    # Save alpha trajectories for comparison plot
    save_data = {}
    for key, r in results.items():
        save_data[key] = {
            "alpha_fit": r["alpha_fit"],
            "ref_alpha": r["ref_alpha"],
            "rel_err_pct": r["rel_err_pct"],
            "final_loss": r["final_loss"],
        }
    with open(out_dir / "exp7_success_baseline.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {out_dir}")
