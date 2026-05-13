# autoresearch-PINN

自动优化参数化SCOPE的超参数和代码结构。

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may13`). Branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current branch.
3. **Read in-scope files**:
   - `scope_selfdiscovery.py` — the file you modify. Contains network, loss, training loop.
   - `EXPERIMENTS.md` — experiment log (do not modify, it's for human reference).
4. **Initialize results.tsv**: Create `results.tsv` with header row only.
5. **Confirm and go**.

## Task

**Problem**: Parametric SCOPE PINN for Guderley converging shock.
- Input: (s, γ, n) — collocation coordinate, heat ratio, geometry
- Output: α — the eigenvalue (self-similar exponent)
- Training = solving the ODE eigenvalue problem (no separate inference phase)

**What the code does**:
- `alpha_net(γ, n)` outputs α
- Hard constraint constructs C(s) satisfying boundary conditions exactly
- ODE residual loss drives α to the correct eigenvalue
- Span constraint prevents trivial solution α→1

**Fixed files** (do not modify):
- `common.py`, `guderley_ode_solver.py` — physics utilities

**The file you modify**: `scope_selfdiscovery.py`

## Metric

**Goal: minimize Max α error (%) across all γ values, for both spherical and cylindrical.**

The metric is printed at the end of each run:
```
spherical: Max=X.XXXX%  Mean=X.XXXX%
cylindrical: Max=X.XXXX%  Mean=X.XXXX%
```

Extract with:
```bash
grep "Max=" run.log | tail -2
```

**Combined metric** = max(spherical_max, cylindrical_max). Lower is better.

Current baseline (Exp-01): spherical Max=5.65%, cylindrical Max=3.09%

## Running an experiment

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python scope_selfdiscovery.py --epochs 10000 --device cuda > run.log 2>&1
```

Each run takes ~10 minutes on A100.

## What you CAN modify in scope_selfdiscovery.py

- Hyperparameters: `epochs`, `lr`, alpha_net lr multiplier, span constraint weight/threshold
- Network architecture: alpha_net depth/width, C_net depth/width, activation functions
- Loss design: add/remove/modify constraint terms
- Initialization: alpha_net initial value strategy
- Training strategy: learning rate schedule, gradient clipping

## What you CANNOT modify

- Physics functions: `shock_conditions`, `critical_point`, `chisnell_rhs`
- The hard constraint structure: `C = (1-s)*Cs + s*C0 + s*(1-s)*correction`
- The test gamma values and evaluation logic

## Output format

```
spherical: Max=X.XXXX%  Mean=X.XXXX%
cylindrical: Max=X.XXXX%  Mean=X.XXXX%
```

## Logging results

Log to `results.tsv` (tab-separated):

```
commit	sph_max	cyl_max	combined	status	description
```

- `combined` = max(sph_max, cyl_max)
- `status`: `keep`, `discard`, or `crash`

Example:
```
commit	sph_max	cyl_max	combined	status	description
a1b2c3d	5.6515	3.0870	5.6515	keep	baseline
b2c3d4e	3.2100	2.1000	3.2100	keep	increase alpha_net lr to 0.5
```

## Experiment loop

Use GPU 0 and GPU 1 alternately (or run two experiments in parallel on different GPUs).

LOOP FOREVER:

1. Check current git state and results.tsv
2. Propose one change to `scope_selfdiscovery.py` based on analysis of previous results
3. `git commit`
4. Run: `CUDA_VISIBLE_DEVICES=<0 or 1> python scope_selfdiscovery.py --epochs 10000 --device cuda > run.log 2>&1`
5. Extract metric: `grep "Max=" run.log | tail -2`
6. If crash: `tail -50 run.log`, fix if trivial, else discard
7. Log to results.tsv
8. If combined metric improved → keep commit
9. If not improved → `git reset --hard HEAD~1`

**NEVER STOP** until manually interrupted.

**Ideas to try** (in rough priority order):
1. Increase alpha_net lr (currently 0.3x → try 0.5x, 1.0x)
2. Increase epochs (10000 → 20000, 30000)
3. Tune span constraint weight (10.0 → 5.0, 20.0) and threshold (0.1 → 0.05, 0.2)
4. Deeper alpha_net (2 layers → 3 layers)
5. Different activation (Tanh → SiLU, GELU)
6. Add monotonicity regularization on α(γ) (α should decrease as γ increases)
7. Curriculum: start with γ near initial value, gradually expand range
8. Different initial α per geometry (currently 0.71/0.83)
9. Add second-order ODE constraint term
10. Try different lr schedules (cosine → exponential decay)
