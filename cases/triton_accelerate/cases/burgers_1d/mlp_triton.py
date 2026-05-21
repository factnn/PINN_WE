"""1D Burgers - Triton fused kernel (forward + backward in Triton)."""
import sys, os; sys.path.insert(0, __import__('pathlib').Path(__file__).parent.parent.parent.__str__())

# Clear Triton cache to avoid stale kernels
import shutil
triton_cache = os.path.expanduser('~/.triton/cache')
if os.path.exists(triton_cache):
    shutil.rmtree(triton_cache)

from cases.burgers_1d.common import *
from kernels.stencil_1d import burgers_2d_loss_triton_autograd

nu_val = 0.01 / np.pi

def loss_fn(model, U, X, T, dx, dt):
    return burgers_2d_loss_triton_autograd(U, dx, dt, nu_val) + 10*ic_loss_from_U(U,X) + 10*bc_loss_from_U(U)

if __name__ == "__main__":
    args = base_argparser("Triton fused kernel").parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_and_save("triton", loss_fn, runs=args.runs,
                   max_epochs=args.max_epochs, lr=args.lr, loss_threshold=args.threshold)
