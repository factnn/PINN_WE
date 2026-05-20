#!/usr/bin/env python3
"""Legacy non-PINN quick check for the old self-discovery branch.

Prefer `cases/self-discovery/canonical/nonpinn_fit.py` for the current mainline.

Quick non-PINN validation for the Guderley self-similar ODE.

This script performs a classical shooting-style scan over alpha without any
neural network machinery. For each alpha, it integrates the first-order
self-similar ODE inward from the shock and measures how well the solution
approaches a smooth sonic crossing.

Key idea:
    A physically regular Guderley solution should satisfy both
        Delta = (V - 1)^2 - C^2 -> 0
    and the ODE numerators -> 0 at the same location.

We therefore scan alpha and score each trajectory by a "regularity defect"
near the closest approach to the sonic line.

This is intentionally dependency-free (pure Python stdlib) so it can run even
when the current virtual environment lacks numpy/scipy/torch.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


@dataclass
class ScanResult:
    alpha: float
    status: str
    xi_end: float
    steps: int
    min_abs_delta: float
    xi_at_min_delta: float
    V_at_min_delta: float
    C_at_min_delta: float
    numerator_v_at_min_delta: float
    numerator_c_at_min_delta: float
    regularity_score: float
    crossed_sonic: bool
    notes: str


@dataclass
class TrajectoryState:
    xi: float
    V: float
    C: float
    delta: float
    numerator_v: float
    numerator_c: float


def shock_bc_strong(gamma: float) -> Tuple[float, float]:
    """Strong-shock boundary values at xi=1."""
    V1 = 2.0 / (gamma + 1.0)
    C1 = math.sqrt(2.0 * gamma * (gamma - 1.0)) / (gamma + 1.0)
    return V1, C1


def shock_bc_finite_mach(gamma: float, mach_inf: float) -> Tuple[float, float]:
    """Finite-Mach Rankine-Hugoniot values used in some repository variants."""
    m2 = mach_inf * mach_inf
    V1 = (2.0 + (gamma - 1.0) * m2) / ((gamma + 1.0) * m2)
    C1 = math.sqrt(2.0 * gamma * m2 - (gamma - 1.0)) / ((gamma + 1.0) * mach_inf)
    return V1, C1


def numerators(alpha: float, gamma: float, geometry_source: int, V: float, C: float) -> Tuple[float, float]:
    """Return the two first-order ODE numerators from PRINCIPLE.md.

    The implementation follows the repository documentation, not an external CAS.
    `geometry_source` is the coefficient appearing in radial source terms, i.e.
    typically 0 for planar, 1 for cylindrical, 2 for spherical in 3D Euler form.
    """
    vm1 = V - 1.0
    numerator_v = (
        vm1 * (V * vm1 * (V - 1.0 / alpha) - 2.0 * C * C / gamma)
        - geometry_source * C * C * vm1 / (gamma * alpha)
    )
    numerator_c = (
        C * (vm1 * (V - 1.0 / alpha) - C * C / gamma)
        - geometry_source * C / (gamma * alpha)
    )
    return numerator_v, numerator_c


def ode_rhs(alpha: float, gamma: float, geometry_source: int, xi: float, V: float, C: float) -> Tuple[float, float, float, float, float]:
    """First-order ODE RHS dV/dxi and dC/dxi together with diagnostics."""
    delta = (V - 1.0) * (V - 1.0) - C * C
    nv, nc = numerators(alpha, gamma, geometry_source, V, C)

    if xi <= 0.0:
        raise ValueError("xi must stay positive")
    if abs(delta) < 1e-14:
        raise ZeroDivisionError("trajectory hit sonic singularity too sharply")

    dV = nv / (xi * delta)
    dC = nc / (xi * delta)
    return dV, dC, delta, nv, nc


def rk4_step(alpha: float, gamma: float, geometry_source: int, xi: float, V: float, C: float, h: float) -> Tuple[float, float, float]:
    """Single RK4 step for the first-order system."""
    k1V, k1C, _, _, _ = ode_rhs(alpha, gamma, geometry_source, xi, V, C)

    k2V, k2C, _, _, _ = ode_rhs(
        alpha, gamma, geometry_source,
        xi + 0.5 * h,
        V + 0.5 * h * k1V,
        C + 0.5 * h * k1C,
    )

    k3V, k3C, _, _, _ = ode_rhs(
        alpha, gamma, geometry_source,
        xi + 0.5 * h,
        V + 0.5 * h * k2V,
        C + 0.5 * h * k2C,
    )

    k4V, k4C, _, _, _ = ode_rhs(
        alpha, gamma, geometry_source,
        xi + h,
        V + h * k3V,
        C + h * k3C,
    )

    V_next = V + (h / 6.0) * (k1V + 2.0 * k2V + 2.0 * k3V + k4V)
    C_next = C + (h / 6.0) * (k1C + 2.0 * k2C + 2.0 * k3C + k4C)
    xi_next = xi + h
    return xi_next, V_next, C_next


def integrate_trajectory(
    alpha: float,
    gamma: float,
    geometry_source: int,
    xi_stop: float,
    step: float,
    use_finite_mach: bool,
    mach_inf: float,
    sonic_tol: float,
    slope_limit: float,
) -> ScanResult:
    """Integrate from xi=1 inward and score closest sonic approach."""
    if use_finite_mach:
        V1, C1 = shock_bc_finite_mach(gamma, mach_inf)
        bc_note = f"finite-Mach BC (M_inf={mach_inf:g})"
    else:
        V1, C1 = shock_bc_strong(gamma)
        bc_note = "strong-shock BC"

    start_offset = abs(step)
    xi = 1.0

    dV1, dC1, delta1, nv1, nc1 = ode_rhs(alpha, gamma, geometry_source, xi, V1, C1)
    V = V1 - start_offset * dV1
    C = C1 - start_offset * dC1
    xi = 1.0 - start_offset

    best = TrajectoryState(
        xi=xi,
        V=V,
        C=C,
        delta=(V - 1.0) * (V - 1.0) - C * C,
        numerator_v=nv1,
        numerator_c=nc1,
    )

    crossed_sonic = False
    status = "ok"
    notes = bc_note
    steps = 0

    while xi > xi_stop:
        steps += 1
        try:
            dV, dC, delta, nv, nc = ode_rhs(alpha, gamma, geometry_source, xi, V, C)
        except ZeroDivisionError:
            status = "singular"
            crossed_sonic = True
            break
        except Exception as exc:
            status = "invalid"
            notes += f"; rhs failure: {exc}"
            break

        abs_delta = abs(delta)
        current_score = abs(nv) + abs(nc)
        best_score = abs(best.numerator_v) + abs(best.numerator_c)
        if abs_delta < abs(best.delta) or (math.isclose(abs_delta, abs(best.delta)) and current_score < best_score):
            best = TrajectoryState(xi=xi, V=V, C=C, delta=delta, numerator_v=nv, numerator_c=nc)

        if abs_delta < sonic_tol:
            crossed_sonic = True
            status = "near-sonic"
            break

        if not (math.isfinite(V) and math.isfinite(C) and math.isfinite(dV) and math.isfinite(dC)):
            status = "nan"
            notes += "; non-finite state"
            break

        if abs(dV) > slope_limit or abs(dC) > slope_limit:
            status = "stiff-blowup"
            notes += "; slope limit exceeded"
            break

        try:
            xi, V, C = rk4_step(alpha, gamma, geometry_source, xi, V, C, -abs(step))
        except ZeroDivisionError:
            status = "singular-step"
            crossed_sonic = True
            break
        except Exception as exc:
            status = "step-failure"
            notes += f"; step failure: {exc}"
            break

        if C <= 0.0:
            status = "nonphysical"
            notes += "; sound speed became non-positive"
            break

    min_abs_delta = abs(best.delta)
    reg_score = abs(best.numerator_v) + abs(best.numerator_c) + 10.0 * min_abs_delta
    if not crossed_sonic:
        reg_score += 1.0

    return ScanResult(
        alpha=alpha,
        status=status,
        xi_end=xi,
        steps=steps,
        min_abs_delta=min_abs_delta,
        xi_at_min_delta=best.xi,
        V_at_min_delta=best.V,
        C_at_min_delta=best.C,
        numerator_v_at_min_delta=best.numerator_v,
        numerator_c_at_min_delta=best.numerator_c,
        regularity_score=reg_score,
        crossed_sonic=crossed_sonic,
        notes=notes,
    )


def parse_alpha_list(alpha_text: Optional[str], start: float, stop: float, num: int) -> List[float]:
    if alpha_text:
        return [float(item.strip()) for item in alpha_text.split(',') if item.strip()]
    if num < 2:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def format_result(result: ScanResult) -> str:
    return (
        f"alpha={result.alpha:.6f} | score={result.regularity_score:.6e} | "
        f"min|Delta|={result.min_abs_delta:.6e} at xi={result.xi_at_min_delta:.6f} | "
        f"|Nv|+|Nc|={(abs(result.numerator_v_at_min_delta) + abs(result.numerator_c_at_min_delta)):.6e} | "
        f"status={result.status}"
    )


def run_scan(args: argparse.Namespace) -> List[ScanResult]:
    alphas = parse_alpha_list(args.alphas, args.alpha_start, args.alpha_stop, args.num)
    results: List[ScanResult] = []

    print("=" * 78)
    print("Non-PINN quick check: Guderley sonic-regularity scan")
    print("=" * 78)
    print(
        f"gamma={args.gamma}, geometry_source={args.geometry_source}, "
        f"BC={'finite-Mach' if args.finite_mach else 'strong-shock'}, "
        f"xi_stop={args.xi_stop}, step={args.step}"
    )
    print("Interpretation: lower regularity_score is better.")
    print()

    for alpha in alphas:
        result = integrate_trajectory(
            alpha=alpha,
            gamma=args.gamma,
            geometry_source=args.geometry_source,
            xi_stop=args.xi_stop,
            step=args.step,
            use_finite_mach=args.finite_mach,
            mach_inf=args.mach_inf,
            sonic_tol=args.sonic_tol,
            slope_limit=args.slope_limit,
        )
        results.append(result)
        print(format_result(result))

    results.sort(key=lambda item: item.regularity_score)
    print("\n" + "-" * 78)
    print("Top candidates by regularity score:")
    for idx, result in enumerate(results[: min(5, len(results))], start=1):
        print(f"{idx:>2d}. {format_result(result)}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "meta": {
                        "gamma": args.gamma,
                        "geometry_source": args.geometry_source,
                        "finite_mach": args.finite_mach,
                        "mach_inf": args.mach_inf,
                        "xi_stop": args.xi_stop,
                        "step": args.step,
                        "sonic_tol": args.sonic_tol,
                        "slope_limit": args.slope_limit,
                    },
                    "results": [asdict(item) for item in results],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved JSON results to {output_path}")

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quick non-PINN scan for Guderley alpha")
    parser.add_argument("--alphas", type=str, default=None, help="Comma-separated alpha list, e.g. 0.65,0.70,0.717,0.75")
    parser.add_argument("--alpha-start", type=float, default=0.65)
    parser.add_argument("--alpha-stop", type=float, default=0.82)
    parser.add_argument("--num", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument(
        "--geometry-source",
        type=int,
        default=2,
        help="Radial source coefficient m in m/xi terms. For standard 3D spherical Euler form, use 2.",
    )
    parser.add_argument("--finite-mach", action="store_true", help="Use finite-Mach shock BC instead of strong-shock BC")
    parser.add_argument("--mach-inf", type=float, default=10.0)
    parser.add_argument("--xi-stop", type=float, default=0.05)
    parser.add_argument("--step", type=float, default=1e-4)
    parser.add_argument("--sonic-tol", type=float, default=5e-4)
    parser.add_argument("--slope-limit", type=float, default=5e3)
    parser.add_argument("--output-json", type=str, default="/share/project/zpy/PINN_WE/cases/self-discovery/output_quickcheck/results.json")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_scan(args)


if __name__ == "__main__":
    main()
