"""AdvDiff 2D - Phy-CNN (PyTorch FD)."""
import sys, os; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.parent.__str__())
from cases.advdiff_2d.common import *

if __name__ == "__main__":
    args = base_argparser("Phy-CNN CAN-PINN").parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("cnn_canpinn", PhyCNN, runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
