"""Euler 2D (Sod) - MLP + Triton fused kernel."""
import sys, os; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.parent.__str__())
from cases.euler_2d.common import *

# TODO: implement Euler Triton kernel (stencil_euler.py)

if __name__ == "__main__":
    args = base_argparser("MLP + Triton").parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("mlp_triton", MLP, runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
