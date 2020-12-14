'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this computer program. 
Using this computer program means that you agree to the terms in the LICENSE file (https://flame.is.tue.mpg.de/modellicense) included 
with the FLAME model. Any use not explicitly granted by the LICENSE is prohibited.

Copyright 2020 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its 
Max Planck Institute for Intelligent Systems. All rights reserved.

More information about FLAME is available at http://flame.is.tue.mpg.de.
For comments or questions, please email us at flame@tue.mpg.de
'''

import numpy as np
import chumpy as ch
from os.path import join

from smpl_webuser.serialization import load_model
from fitting.landmarks import load_embedding, landmark_error_3d
from fitting.util import load_binary_pickle, write_simple_obj, safe_mkdir, get_unit_factor

# -----------------------------------------------------------------------------

def fit_lmk3d( lmk_3d,                      # input landmark 3d
               model,                       # model
               lmk_face_idx, lmk_b_coords,  # landmark embedding
               weights,                     # weights for the objectives
               shape_num=300, expr_num=100, opt_options=None ):
    
    """ function: fit FLAME model to 3D landmarks

    input: 
        lmk_3d: input landmark 3D, in shape (N,3)
        model: FLAME face model
        lmk_face_idx, lmk_b_coords: landmark embedding, in face indices and barycentric coordinates
        weights: weights for each objective
        shape_num, expr_num: numbers of shape and expression compoenents used
        opt_options: optimizaton options

    output:
        model.r: fitted result vertices
        model.f: fitted result triangulations (fixed in this code)
        parms: fitted model parameters

    """

    # variables
    pose_idx       = np.union1d(np.arange(3), np.arange(6,9)) # global rotation and jaw rotation
    shape_idx      = np.arange( 0, min(300,shape_num) )        # valid shape component range in "betas": 0-299
    expr_idx       = np.arange( 300, 300+min(100,expr_num) )   # valid expression component range in "betas": 300-399
    used_idx       = np.union1d( shape_idx, expr_idx )
    model.betas[:] = np.random.rand( model.betas.size ) * 0.0  # initialized to zero
    model.pose[:]  = np.random.rand( model.pose.size ) * 0.0   # initialized to zero
    free_variables = [ model.trans, model.pose[pose_idx], model.betas[used_idx] ] 
    
    # weights
    print("fit_lmk3d(): use the following weights:")
    for kk in weights.keys():
        print("fit_lmk3d(): weights['%s'] = %f" % ( kk, weights[kk] ))

    # objectives
    # lmk
    lmk_err = landmark_error_3d( mesh_verts=model, 
                                 mesh_faces=model.f, 
                                 lmk_3d=lmk_3d, 
                                 lmk_face_idx=lmk_face_idx, 
                                 lmk_b_coords=lmk_b_coords, 
                                 weight=weights['lmk'] )
    # regularizer
    shape_err = weights['shape'] * model.betas[shape_idx] 
    expr_err  = weights['expr']  * model.betas[expr_idx] 
    pose_err  = weights['pose']  * model.pose[3:] # exclude global rotation
    objectives = {}
    objectives.update( { 'lmk': lmk_err, 'shape': shape_err, 'expr': expr_err, 'pose': pose_err } ) 

    # options
    if opt_options is None:
        print("fit_lmk3d(): no 'opt_options' provided, use default settings.")
        import scipy.sparse as sp
        opt_options = {}
        opt_options['disp']    = 1
        opt_options['delta_0'] = 0.1
        opt_options['e_3']     = 1e-4
        opt_options['maxiter'] = 2000
        sparse_solver = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options['maxiter'])[0]
        opt_options['sparse_solver'] = sparse_solver

    # on_step callback
    def on_step(_):
        pass
        
    # optimize
    # step 1: rigid alignment
    from time import time
    timer_start = time()
    print("\nstep 1: start rigid fitting...")
    ch.minimize( fun      = lmk_err,
                 x0       = [ model.trans, model.pose[0:3] ],
                 method   = 'dogleg',
                 callback = on_step,
                 options  = opt_options )
    timer_end = time()
    print("step 1: fitting done, in %f sec\n" % ( timer_end - timer_start ))

    # step 2: non-rigid alignment
    timer_start = time()
    print("step 2: start non-rigid fitting...")    
    ch.minimize( fun      = objectives,
                 x0       = free_variables,
                 method   = 'dogleg',
                 callback = on_step,
                 options  = opt_options )
    timer_end = time()
    print("step 2: fitting done, in %f sec\n" % ( timer_end - timer_start ))

    # return results
    parms = { 'trans': model.trans.r, 'pose': model.pose.r, 'betas': model.betas.r }
    return model.r, model.f, parms

# -----------------------------------------------------------------------------

def run_fitting():
    # input landmarks
    lmk_path = './data/scan_lmks.npy'
    # measurement unit of landmarks ['m', 'cm', 'mm']
    unit = 'm' 

    scale_factor = get_unit_factor('m') / get_unit_factor(unit)
    lmk_3d = scale_factor*np.load(lmk_path)
    print("loaded 3d landmark from:", lmk_path)

    # model
    model_path = './models/generic_model.pkl' # change to 'female_model.pkl' or 'male_model.pkl', if gender is known
    model = load_model(model_path)       # the loaded model object is a 'chumpy' object, check https://github.com/mattloper/chumpy for details
    print("loaded model from:", model_path)

    # landmark embedding
    lmk_emb_path = './models/flame_static_embedding.pkl' 
    lmk_face_idx, lmk_b_coords = load_embedding(lmk_emb_path)
    print("loaded lmk embedding")

    # output
    output_dir = './output'
    safe_mkdir(output_dir)

    # weights
    weights = {}
    # landmark term
    weights['lmk']   = 1.0   
    # shape regularizer (weight higher to regularize face shape more towards the mean)
    weights['shape'] = 1e-3
    # expression regularizer (weight higher to regularize facial expression more towards the mean)
    weights['expr']  = 1e-3
    # regularization of head rotation around the neck and jaw opening (weight higher for more regularization)
    weights['pose']  = 1e-2
    
    # number of shape and expression parameters (we do not recommend using too many parameters for fitting to sparse keypoints)
    shape_num = 100
    expr_num = 50

    # optimization options
    import scipy.sparse as sp
    opt_options = {}
    opt_options['disp']    = 1
    opt_options['delta_0'] = 0.1
    opt_options['e_3']     = 1e-4
    opt_options['maxiter'] = 2000
    sparse_solver = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options['maxiter'])[0]
    opt_options['sparse_solver'] = sparse_solver

    # run fitting
    mesh_v, mesh_f, parms = fit_lmk3d( lmk_3d=lmk_3d,                                         # input landmark 3d
                                       model=model,                                           # model
                                       lmk_face_idx=lmk_face_idx, lmk_b_coords=lmk_b_coords,  # landmark embedding
                                       weights=weights,                                       # weights for the objectives
                                       shape_num=shape_num, expr_num=expr_num, opt_options=opt_options ) # options

    # write result
    output_path = join( output_dir, 'fit_lmk3d_result.obj' )
    write_simple_obj( mesh_v=mesh_v, mesh_f=mesh_f, filepath=output_path, verbose=False )

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    run_fitting()

