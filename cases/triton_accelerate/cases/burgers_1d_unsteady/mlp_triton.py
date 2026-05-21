"""1D Unsteady Burgers - MLP + Triton fused kernel."""
import sys, os; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.parent.__str__())
from cases.burgers_1d_unsteady.common import *
from kernels.stencil_1d_unsteady import burgers_2d_loss_triton_autograd

def loss_fn(model, xt, X, T, dx, dt):
    U = infer(model, xt, X, T)
    return burgers_2d_loss_triton_autograd(U, dx, dt, nu) + 10*ic_loss_from_U(U, X) + 10*bc_loss_from_U(U)

if __name__ == "__main__":
    args = base_argparser("MLP + Triton").parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("mlp_triton", MLP, loss_fn, runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
