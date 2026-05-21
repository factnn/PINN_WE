"""TGV 3D - MLP + Triton fused kernel."""
import sys, os; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.parent.__str__())
import torch
from cases.tgv_3d.common import *

# TODO: implement 3D NS Triton kernel (stencil_3d.py)
# For now, falls back to PyTorch FD (same as canpinn)

if __name__ == "__main__":
    args = base_argparser("MLP + Triton").parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("mlp_triton", MLP, runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
