"""TGV 2D - MLP CAN-PINN (PyTorch FD)."""
import sys, os; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.parent.__str__())
from cases.tgv_2d.common import *

if __name__ == "__main__":
    args = base_argparser("MLP CAN-PINN").parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("mlp_canpinn", MLP, runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
