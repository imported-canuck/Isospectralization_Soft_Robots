from shape_library import *
from spectrum_alignment import *
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

params = OptimizationParams() # Create object of class OptimizationParams to store parameters
params.evals = [20] # Optimize the first (20) eigenvalues
params.numsteps = 5000 # Number of optimization steps
params.plot=False # Disables plotting during optimization

[VERT, TRIV] = load_mesh('data/circle/'); # Loads the initial shape
[VERT,TRIV] = resample(VERT, TRIV, 300) # Resample to 300 vertices

# mesh.vert is a list of vertex coordinates (accesed as VERT)
# mesh.triv is a list of triangle defined by a triplet of vertex indices (accesed as TRIV)

[VERT_t, TRIV_t] = load_mesh('data/bell/') # Load target shape
evals_t = calc_evals(VERT_t,TRIV_t) # Calculates first (20) laplacian eigenvalues and stores in array
mesh = prepare_mesh(VERT,TRIV,'float32') # Creates mesh using vertex coords and triangle coords, stores in tuple
run_optimization(mesh = mesh, target_evals = evals_t, out_path = 'results/bell', params = params)
# Iteratively adjusts the vertex values of mesh s.t. the eigenvalues approach those of target_evals

# [VERT_t, TRIV_t] = load_mesh('data/bell/')
# evals_t = calc_evals(VERT_t,TRIV_t)
# mesh = prepare_mesh(VERT,TRIV,'float32')
# run_optimization(mesh = mesh, target_evals = evals_t, out_path = 'results/bell', params = params)