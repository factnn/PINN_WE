"""1D Burgers - CAN-PINN + torch.compile."""
import sys; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.parent.__str__())
from cases.burgers_1d.common import *
from cases.burgers_1d.compare import loss_canpinn

def loss_fn(model, U, X, T, dx, dt):
    return loss_canpinn(model, X, T, dx, dt) + 10*ic_loss_from_U(U,X) + 10*bc_loss_from_U(U)

if __name__ == "__main__":
    args = base_argparser("CAN-PINN + torch.compile").parse_args()
    import os; os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("compile", loss_fn,
                   model_fn=lambda: torch.compile(MLP().to(device)),
                   runs=args.runs, max_epochs=args.max_epochs,
                   lr=args.lr, loss_threshold=args.threshold)
