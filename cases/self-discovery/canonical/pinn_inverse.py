#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common import FitSummary, generate_dataset, save_summary, similarity_radius, summary_to_dict


class SimilarityPINN(nn.Module):
    def __init__(self, alpha_init: float = 0.75, tc_init: float = 1.02, amplitude_init: float = 1.0):
        super().__init__()
        self.log_amplitude = nn.Parameter(torch.tensor(np.log(amplitude_init), dtype=torch.float32))
        self.raw_tc_gap = nn.Parameter(torch.tensor(max(tc_init - 0.95, 0.05), dtype=torch.float32))
        self.raw_alpha = nn.Parameter(torch.tensor((alpha_init - 0.5) / 0.5, dtype=torch.float32))

    @property
    def amplitude(self) -> torch.Tensor:
        return torch.exp(self.log_amplitude)

    @property
    def alpha(self) -> torch.Tensor:
        return 0.5 + 0.5 * torch.sigmoid(self.raw_alpha)

    def tc(self, t_max: torch.Tensor) -> torch.Tensor:
        return t_max + torch.nn.functional.softplus(self.raw_tc_gap)

    def forward(self, t: torch.Tensor, t_max: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tc = self.tc(t_max)
        gap = torch.clamp(tc - t, min=1e-6)
        r = self.amplitude * torch.pow(gap, self.alpha)
        return r, tc


def train_pinn(dataset: dict, epochs: int = 3000, lr: float = 2e-3, device: str = "cpu") -> tuple[FitSummary, dict]:
    geometry = str(dataset["geometry"])
    alpha_true = float(dataset["alpha_true"])
    amplitude_true = float(dataset["amplitude_true"])
    tc_true = float(dataset["tc_true"])
    noise_level = float(dataset["noise_level"])
    seed = int(dataset["seed"])
    t_np = np.asarray(dataset["t"], dtype=np.float32)
    r_obs_np = np.asarray(dataset["r_obs"], dtype=np.float32)
    r_clean_np = np.asarray(dataset["r_clean"], dtype=np.float32)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SimilarityPINN(alpha_init=0.74 if geometry == "spherical" else 0.82).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    t = torch.tensor(t_np.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
    r_obs = torch.tensor(r_obs_np.reshape(-1, 1), dtype=torch.float32, device=device)
    t_max = torch.tensor(float(np.max(t_np)), dtype=torch.float32, device=device)
    t_colloc = torch.linspace(float(np.min(t_np)), float(np.max(t_np)), 4 * len(t_np), device=device).reshape(-1, 1)
    t_colloc.requires_grad_(True)

    history_alpha = []
    history_loss = []

    def compute_loss() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r_pred, tc = model(t, t_max)
        r_colloc, tc_colloc = model(t_colloc, t_max)
        dr_dt = torch.autograd.grad(r_colloc, t_colloc, grad_outputs=torch.ones_like(r_colloc), create_graph=True)[0]
        physics_residual = dr_dt + model.alpha * r_colloc / torch.clamp(tc_colloc - t_colloc, min=1e-6)
        loss_data = torch.mean((torch.log(torch.clamp(r_pred, min=1e-8)) - torch.log(torch.clamp(r_obs, min=1e-8))) ** 2)
        loss_phys = torch.mean(physics_residual ** 2)
        loss = loss_data + 1000.0 * loss_phys
        return loss, loss_data, loss_phys, r_pred

    adam_epochs = max(epochs - 300, 200)
    for epoch in range(1, adam_epochs + 1):
        optimizer.zero_grad()
        loss, _, _, _ = compute_loss()
        loss.backward()
        optimizer.step()

        history_alpha.append(float(model.alpha.detach().cpu()))
        history_loss.append(float(loss.detach().cpu()))

    lbfgs = optim.LBFGS(model.parameters(), lr=0.8, max_iter=300, history_size=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        lbfgs.zero_grad()
        loss, _, _, _ = compute_loss()
        loss.backward()
        return loss

    lbfgs.step(closure)
    final_loss, _, _, _ = compute_loss()
    history_alpha.append(float(model.alpha.detach().cpu()))
    history_loss.append(float(final_loss.detach().cpu()))

    with torch.no_grad():
        r_pred, tc = model(t, t_max)
        alpha_fit = float(model.alpha.detach().cpu())
        amplitude_fit = float(model.amplitude.detach().cpu())
        tc_fit = float(tc.detach().cpu())
        r_fit_np = r_pred.detach().cpu().numpy().reshape(-1)

    rel_err = abs(alpha_fit - alpha_true) / alpha_true * 100.0
    residual_l2 = float(np.linalg.norm(r_fit_np - r_obs_np))
    summary = FitSummary(
        method="pinn",
        geometry=geometry,
        alpha_true=alpha_true,
        alpha_fit=alpha_fit,
        amplitude_true=amplitude_true,
        amplitude_fit=amplitude_fit,
        tc_true=tc_true,
        tc_fit=tc_fit,
        noise_level=noise_level,
        relative_alpha_error_percent=rel_err,
        residual_l2=residual_l2,
        n_samples=len(t_np),
        seed=seed,
    )
    return summary, {
        "t": t_np,
        "r_obs": r_obs_np,
        "r_clean": r_clean_np,
        "r_fit": r_fit_np,
        "alpha_history": np.array(history_alpha),
        "loss_history": np.array(history_loss),
    }


def plot_result(output_dir: Path, summary: FitSummary, data: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(data["t"], data["r_clean"], "k--", linewidth=2, label="true law")
    axes[0].scatter(data["t"], data["r_obs"], s=18, alpha=0.75, label="observations")
    axes[0].plot(data["t"], data["r_fit"], "r-", linewidth=2, label=f"PINN α={summary.alpha_fit:.6f}")
    axes[0].set_title(f"PINN inverse ({summary.geometry})")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("R(t)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(data["alpha_history"], linewidth=2)
    axes[1].axhline(summary.alpha_true, color="r", linestyle="--", linewidth=2)
    axes[1].set_title("alpha convergence")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("alpha")
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(np.maximum(data["loss_history"], 1e-16), linewidth=2)
    axes[2].set_title("training loss")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("loss")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f"pinn_{summary.geometry}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=["spherical", "cylindrical"], default="spherical")
    parser.add_argument("--noise", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="/share/project/zpy/PINN_WE/cases/self-discovery/canonical/output")
    args = parser.parse_args()

    dataset = generate_dataset(args.geometry, n_samples=args.n_samples, noise_level=args.noise, seed=args.seed)
    summary, data = train_pinn(dataset, epochs=args.epochs, lr=args.lr, device=args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_result(out, summary, data)
    save_summary(out / f"pinn_{summary.geometry}.json", {"summary": summary_to_dict(summary)})

    print(f"[{summary.geometry}] PINN")
    print(f"  true alpha   = {summary.alpha_true:.6f}")
    print(f"  fitted alpha = {summary.alpha_fit:.6f}")
    print(f"  rel err      = {summary.relative_alpha_error_percent:.6e}%")
    print(f"  residual L2  = {summary.residual_l2:.6e}")


if __name__ == "__main__":
    main()
