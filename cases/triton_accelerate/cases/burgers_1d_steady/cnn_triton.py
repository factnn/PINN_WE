"""Steady Burgers 1D - Phy-CNN + Triton fused kernel."""
import sys, os; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.parent.__str__())
from cases.burgers_1d_steady.common import *

# TODO: implement steady 1D Burgers Triton kernel

if __name__ == "__main__":
    args = base_argparser("Phy-CNN + Triton").parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("cnn_triton", PhyCNN, runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
