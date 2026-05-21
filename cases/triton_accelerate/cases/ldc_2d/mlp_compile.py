"""LDC 2D - MLP CAN-PINN + torch.compile."""
import sys, os; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.parent.__str__())
import torch
from cases.ldc_2d.common import *

if __name__ == "__main__":
    args = base_argparser("MLP CAN-PINN + torch.compile").parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("mlp_compile", lambda: torch.compile(MLP()), runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
