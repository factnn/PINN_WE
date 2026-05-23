"""Utility functions for model creation and loss dispatch."""
import torch


BACKENDS = [
    "mlp_vanilla", "mlp_canpinn", "mlp_compile", "mlp_triton",
    "cnn_canpinn", "cnn_compile", "cnn_triton",
]


def make_model(physics, backend: str, device: str):
    """Build model from backend string and physics module."""
    is_cnn = "cnn" in backend
    model = physics.make_cnn() if is_cnn else physics.make_mlp()
    model = model.to(device)
    if "compile" in backend:
        model = torch.compile(model)
    return model


def get_loss_fn(physics, backend: str):
    """Select loss function variant from backend string."""
    if "vanilla" in backend:
        return physics.loss_vanilla
    elif "triton" in backend:
        return physics.loss_triton
    else:  # canpinn or compile
        return physics.loss_canpinn
