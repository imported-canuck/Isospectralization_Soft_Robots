"""Reconstruct a shape from a portion of eigenvalue sequence."""
from shape_library import load_mesh, prepare_mesh
from spectrum_alignment import OptimizationParams, calc_evals, run_optimization

params = OptimizationParams()
params.checkpoint_steps = 100
params.eval_steps = 100
params.min_eval_loss = 0.0001
params.evals = [200]
params.steps = 3000

VERT, TRIV = load_mesh("data/ShapeNet3_sp_bottle/modelS/") # initial shape
mesh = prepare_mesh(VERT, TRIV, "float32")

VERT_t, TRIV_t = load_mesh("data/ShapeNet1_bottle/modelT/") # target shape
evals_t = calc_evals(VERT_t, TRIV_t)

run_optimization(
    mesh=mesh,
    target_evals=evals_t,
    out_path="results/spheretobottle200/",
    params=params,
)
