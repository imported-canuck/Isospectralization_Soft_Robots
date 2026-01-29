"""Reconstruct a shape from a portion of eigenvalue sequence."""
from shape_library import load_mesh, prepare_mesh, resample
from spectrum_alignment import OptimizationParams, calc_evals, run_optimization

# ===== reproducibility block =====
import os, random, numpy as np

SEED = 14                 
os.environ["PYTHONHASHSEED"] = str(SEED)   # python deterministic hash
random.seed(SEED)                           # built-in RNG
np.random.seed(SEED)                        # NumPy

# PyTorch variant ------------------------------------------------
import torch
torch.manual_seed(SEED)                   # CPU
torch.cuda.manual_seed_all(SEED)          # GPU
torch.backends.cudnn.deterministic = True # turn off nondeterministic kernels
torch.backends.cudnn.benchmark     = False
# =======================================================================

params = OptimizationParams()
params.evals = [20]
params.min_eval_loss = 0.05
# params.decay_target = 0.15
params.steps = 5000
params.plot = False

[VERT, TRIV] = load_mesh("data/oval/")
[VERT, TRIV] = resample(VERT, TRIV, 300)

[VERT_t, TRIV_t] = load_mesh("data/bell/")
evals_t = calc_evals(VERT_t, TRIV_t)
mesh = prepare_mesh(VERT, TRIV, "float32")
run_optimization(
    mesh=mesh, target_evals=evals_t, out_path="results/bell", params=params
)
