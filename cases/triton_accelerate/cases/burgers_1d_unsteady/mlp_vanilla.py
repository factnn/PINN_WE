"""1D Burgers - Vanilla PINN (autograd)."""
import sys; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.parent.__str__())
from cases.burgers_1d_unsteady.common import *
from cases.burgers_1d_unsteady.compare import loss_vanilla

def loss_fn(model, U, X, T, dx, dt):
    return loss_vanilla(model, X, T) + 10*ic_loss_from_U(U,X) + 10*bc_loss_from_U(U)

if __name__ == "__main__":
    args = base_argparser("Vanilla PINN").parse_args()
    import os; os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("vanilla", loss_fn, runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
