#!/usr/bin/env python3
"""
Symbolic regression for Guderley eigenvalue alpha(gamma).
Runs separately for n=2 (cylindrical) and n=3 (spherical).
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from guderley_ode_solver import find_eigenvalue
from pysr import PySRRegressor

out = Path(__file__).parent / "output" / "symbolic_regression"
out.mkdir(parents=True, exist_ok=True)

gammas = np.linspace(1.2, 3.0, 50)

for n, geo in [(2, "cylindrical"), (3, "spherical")]:
    print(f"\n{'='*50}")
    print(f"Symbolic regression: {geo} (n={n})")

    alphas = []
    gs = []
    for g in gammas:
        res = find_eigenvalue(g, n, geo, verbose=False)
        if res and res["alpha"]:
            gs.append(g)
            alphas.append(res["alpha"])

    X = np.array(gs).reshape(-1, 1)
    y = np.array(alphas)
    print(f"  {len(y)} data points, alpha range: {y.min():.4f} - {y.max():.4f}")

    model = PySRRegressor(
        niterations=200,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["log", "sqrt", "exp"],
        populations=20,
        population_size=50,
        maxsize=15,
        verbosity=1,
        random_state=42,
        procs=8,
    )
    model.fit(X, y, variable_names=["g"])

    print(f"\n=== Best equations ({geo}) ===")
    print(model)
    model.equations_.to_csv(out / f"equations_{geo}.csv")
    np.save(out / f"data_{geo}.npy", np.column_stack([gs, alphas]))
    print(f"Saved to {out}/equations_{geo}.csv")

