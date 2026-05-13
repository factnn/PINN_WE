from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple
import json

import numpy as np

REFERENCE_ALPHA = {
    "spherical": 0.717,
    "cylindrical": 0.800,
}


@dataclass
class FitSummary:
    method: str
    geometry: str
    alpha_true: float
    alpha_fit: float
    amplitude_true: float
    amplitude_fit: float
    tc_true: float
    tc_fit: float
    noise_level: float
    relative_alpha_error_percent: float
    residual_l2: float
    n_samples: int
    seed: int


def similarity_radius(t: np.ndarray, amplitude: float, tc: float, alpha: float) -> np.ndarray:
    gap = np.maximum(tc - t, 1e-14)
    return amplitude * gap ** alpha


def generate_dataset(
    geometry: str,
    n_samples: int = 80,
    amplitude: float = 1.0,
    tc: float = 1.0,
    noise_level: float = 1e-3,
    seed: int = 42,
) -> Dict[str, np.ndarray | float | str]:
    alpha_true = REFERENCE_ALPHA[geometry]
    rng = np.random.default_rng(seed)
    t = np.linspace(0.05, 0.95 * tc, n_samples)
    r_clean = similarity_radius(t, amplitude, tc, alpha_true)
    noise = np.zeros_like(r_clean)
    if noise_level > 0.0:
        noise = noise_level * np.max(r_clean) * rng.normal(size=r_clean.shape)
    r_obs = np.clip(r_clean + noise, 1e-12, None)
    return {
        "geometry": geometry,
        "alpha_true": alpha_true,
        "amplitude_true": amplitude,
        "tc_true": tc,
        "noise_level": noise_level,
        "seed": seed,
        "t": t,
        "r_clean": r_clean,
        "r_obs": r_obs,
    }


def save_summary(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summary_to_dict(summary: FitSummary) -> dict:
    return asdict(summary)
