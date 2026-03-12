from shape_library import *
from spectrum_alignment import *
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

params = OptimizationParams()
params.min_eval_loss = 0.0001
params.evals = [20]
params.numsteps = 2000
params.volume_reg = 0 ## Modified

[VERT, TRIV] = load_mesh('data/Drake_Sphere/'); # Original shape
mesh = prepare_mesh(VERT,TRIV,'float32')

[VERT_t, TRIV_t] = load_mesh('data/Bubble_Grasp_Deep_0.05x40/mesh_00019/') # Target shape
evals_t = calc_evals(VERT_t,TRIV_t)

run_optimization(mesh = mesh, target_evals = evals_t, out_path = 'results/testing_outreg', params = params)
