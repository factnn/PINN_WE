"""AdvDiff 2D - Phy-CNN + Triton fused kernel."""
import sys, os; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.parent.__str__())
import torch
from cases.advdiff_2d.common import *

# TODO: implement advection-diffusion Triton kernel (stencil_advdiff.py)
# For now, falls back to PyTorch FD (same as canpinn)

if __name__ == "__main__":
    args = base_argparser("Phy-CNN + Triton").parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("cnn_triton", PhyCNN, runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
