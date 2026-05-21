"""TGV 3D - Phy-CNN + Triton fused kernel (full Triton fwd+bwd)."""
import sys, os; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.parent.__str__())
import torch
from cases.tgv_3d.common import *
from kernels.stencil_3d_ns_unsteady import ns3d_residual_triton

def loss_fn(model, xyzt, X, Y, Z, T, U_exact, V_exact, W_exact, P_exact, dx, dy, dz, dt):
    U, V, W, P = infer(model, xyzt, X, Y, Z, T)
    return ns3d_residual_triton(U, V, W, P, dx, dy, dz, dt, nu) + \
           ic_bc_loss(U, V, W, P, U_exact, V_exact, W_exact, P_exact)

if __name__ == "__main__":
    args = base_argparser("Phy-CNN + Triton").parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("cnn_triton", PhyCNN, loss_fn, runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
