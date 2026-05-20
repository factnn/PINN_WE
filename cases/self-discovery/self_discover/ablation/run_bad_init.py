#!/usr/bin/env python3
"""
Bad-init鲁棒性测试: alpha_init=0.75 统一初始值
每个实验4个config，分配到4张GPU并行跑。
一次跑一个实验，串行执行5个实验。
"""
import subprocess
import sys
import json
import textwrap
from pathlib import Path

BAD_INIT = 0.75
CONFIGS = [
    (1.4, "spherical", 3),
    (1.4, "cylindrical", 2),
    (5/3, "spherical", 3),
    (5/3, "cylindrical", 2),
]

EXPERIMENTS = {
    "exp1_soft": {
        "import": "from exp1_soft_constraint import train_soft as train_fn",
        "label": "Exp1 Soft Constraint",
    },
    "exp2a_shock": {
        "import": "from exp2_single_endpoint_shock import train_shock_only as train_fn",
        "label": "Exp2a Shock-Only",
    },
    "exp2b_sonic": {
        "import": "from exp2b_single_endpoint_sonic import train_sonic_only as train_fn",
        "label": "Exp2b Sonic-Only",
    },
    "exp4_no_warmup": {
        "import": "from exp4_no_warmup import train_no_warmup as train_fn",
        "label": "Exp4 No Warmup",
    },
    "exp7_ours": {
        "import": "from exp7_bad_init_helper import train_ours as train_fn",
        "label": "Exp7 Our Method",
    },
}


def make_worker_script(exp_key, gamma, n, geo, gpu_id):
    """生成单个config的worker脚本"""
    exp = EXPERIMENTS[exp_key]
    return textwrap.dedent(f"""\
        import sys, json
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        sys.path.insert(0, str(Path(__file__).parent.parent))
        {exp["import"]}
        result = train_fn({gamma}, {n}, "{geo}", alpha_init={BAD_INIT}, epochs=10000, device="cuda")
        out = {{
            "alpha_fit": result["alpha_fit"],
            "ref_alpha": result["ref_alpha"],
            "rel_err_pct": result["rel_err_pct"],
            "final_loss": result["final_loss"],
        }}
        print("RESULT_JSON:" + json.dumps(out))
    """)


def run_experiment(exp_key):
    exp = EXPERIMENTS[exp_key]
    print(f"\n{'='*70}")
    print(f"  {exp['label']} (bad init={BAD_INIT})")
    print(f"{'='*70}")

    # 4个config并行, 每个一张卡
    procs = []
    for gpu_id, (gamma, geo, n) in enumerate(CONFIGS):
        script = make_worker_script(exp_key, gamma, n, geo, gpu_id)
        script_path = Path(__file__).parent / f"_tmp_worker_{exp_key}_{gpu_id}.py"
        script_path.write_text(script)

        env_cmd = f"CUDA_VISIBLE_DEVICES={gpu_id}"
        cmd = f"{env_cmd} {sys.executable} {script_path}"
        print(f"  GPU{gpu_id}: gamma={gamma:.4f} {geo}")
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(Path(__file__).parent)
        )
        procs.append((gamma, geo, n, gpu_id, proc, script_path))

    # 收集结果
    results = {}
    for gamma, geo, n, gpu_id, proc, script_path in procs:
        stdout, stderr = proc.communicate(timeout=600)
        stdout_str = stdout.decode()
        key = f"g{gamma:.2f}_{geo}"

        # 提取结果JSON
        for line in stdout_str.split("\n"):
            if line.startswith("RESULT_JSON:"):
                results[key] = json.loads(line[len("RESULT_JSON:"):])
                break
        else:
            print(f"  WARNING: GPU{gpu_id} ({key}) no result found!")
            print(f"  stderr: {stderr.decode()[-500:]}")
            results[key] = {
                "alpha_fit": -1, "ref_alpha": -1,
                "rel_err_pct": None, "final_loss": 1e10,
            }

        # 清理临时文件
        script_path.unlink(missing_ok=True)

    # 打印结果表
    print(f"\n  {'Config':<25} {'alpha_fit':<14} {'ref':<14} {'err%':<12} {'loss':<12}")
    print(f"  {'-'*75}")
    for key, r in results.items():
        err_s = f"{r['rel_err_pct']:.4f}" if r['rel_err_pct'] is not None else "N/A"
        print(f"  {key:<25} {r['alpha_fit']:<14.6f} "
              f"{r['ref_alpha']:<14.6f} {err_s:<12} {r['final_loss']:<12.3e}")

    return results


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for exp_key in EXPERIMENTS:
        results = run_experiment(exp_key)
        all_results[exp_key] = results

    # 保存汇总
    with open(out_dir / "bad_init_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # 最终汇总表
    print(f"\n\n{'='*80}")
    print(f"  FINAL SUMMARY: Bad Init (alpha_init={BAD_INIT}) Robustness Test")
    print(f"{'='*80}")
    print(f"  {'Experiment':<25} {'g1.4 sph':<12} {'g1.4 cyl':<12} {'g5/3 sph':<12} {'g5/3 cyl':<12}")
    print(f"  {'-'*73}")
    for exp_key, results in all_results.items():
        label = EXPERIMENTS[exp_key]["label"]
        errs = []
        for key in ["g1.40_spherical", "g1.40_cylindrical", "g1.67_spherical", "g1.67_cylindrical"]:
            r = results.get(key, {})
            e = r.get("rel_err_pct")
            errs.append(f"{e:.4f}%" if e is not None else "N/A")
        print(f"  {label:<25} {errs[0]:<12} {errs[1]:<12} {errs[2]:<12} {errs[3]:<12}")
    print()
