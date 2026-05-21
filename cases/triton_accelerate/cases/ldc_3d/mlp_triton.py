"""LDC 3D - MLP + Triton fused kernel."""
import sys, os; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.parent.__str__())
from cases.ldc_3d.common import *
from kernels.stencil_3d_ns_steady import ldc_residual_triton


def loss_fn(model, xyz, X, Y, Z, dx, dy, dz):
    U, V, W, P = infer(model, xyz, X, Y, Z)
    return ldc_residual_triton(U, V, W, P, dx, dy, dz, nu) + 10 * bc_loss(U, V, W)


if __name__ == "__main__":
    args = base_argparser("MLP + Triton").parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("mlp_triton", MLP, loss_fn=loss_fn, runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
