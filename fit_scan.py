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

from psbody.mesh import Mesh
from smpl_webuser.serialization import load_model
from sbody.mesh_distance import ScanToMesh
from sbody.robustifiers import GMOf
from sbody.alignment.objectives import sample_from_mesh
from fitting.landmarks import load_embedding, landmark_error_3d, mesh_points_by_barycentric_coordinates
from fitting.util import load_binary_pickle, write_simple_obj, safe_mkdir, get_unit_factor

# -----------------------------------------------------------------------------

def compute_approx_scale(lmk_3d, model, lmk_face_idx, lmk_b_coords, opt_options=None):
    """ function: compute approximate scale to align scan and model

    input: 
        lmk_3d: input landmark 3d, in shape (N,3)
        model: FLAME face model
        lmk_face_idx, lmk_b_coords: landmark embedding, in face indices and barycentric coordinates
        opt_options: optimizaton options

    output:
        model.r: fitted result vertices
        model.f: fitted result triangulations (fixed in this code)
        parms: fitted model parameters

    """

    scale = ch.ones(1)
    scan_lmks = scale*ch.array(lmk_3d)
    model_lmks = mesh_points_by_barycentric_coordinates( model, model.f, lmk_face_idx, lmk_b_coords )
    lmk_err = scan_lmks-model_lmks

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

    ch.minimize( fun      = lmk_err,
                 x0       = [ scale, model.trans, model.pose[:3] ],
                 method   = 'dogleg',
                 callback = on_step,
                 options  = opt_options )
    return scale.r

# -----------------------------------------------------------------------------

def fit_scan(  scan,                        # input scan
               lmk_3d,                      # input scan landmarks
               model,                       # model
               lmk_face_idx, lmk_b_coords,  # landmark embedding
               weights,                     # weights for the objectives
               gmo_sigma,                   # weight of the robustifier
               shape_num=300, expr_num=100, opt_options=None ):
    
    """ function: fit FLAME model to a 3D scan

    input: 
        scan: input scan
        lmk_3d: input landmark 3d, in shape (N,3)
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
    shape_idx      = np.arange( 0, min(300,shape_num) )        # valid shape component range in "betas": 0-299
    expr_idx       = np.arange( 300, 300+min(100,expr_num) )   # valid expression component range in "betas": 300-399
    used_idx       = np.union1d( shape_idx, expr_idx )
    model.betas[:] = np.random.rand( model.betas.size ) * 0.0  # initialized to zero
    model.pose[:]  = np.random.rand( model.pose.size ) * 0.0   # initialized to zero
    free_variables = [model.trans, model.pose, model.betas[used_idx]] 
    
    # weights
    print("fit_scan(): use the following weights:")
    for kk in weights.keys():
        print("fit_scan(): weights['%s'] = %f" % ( kk, weights[kk] ))

    # objectives
    # landmark error
    lmk_err = landmark_error_3d(mesh_verts=model, mesh_faces=model.f,  lmk_3d=lmk_3d, lmk_face_idx=lmk_face_idx, lmk_b_coords=lmk_b_coords)

    # scan-to-mesh distance, measuring the distance between scan vertices and the closest points in the model's surface
    sampler = sample_from_mesh(scan, sample_type='vertices')
    s2m = ScanToMesh(scan, model, model.f, scan_sampler=sampler, rho=lambda x: GMOf(x, sigma=gmo_sigma))

    # regularizer
    shape_err = weights['shape'] * model.betas[shape_idx] 
    expr_err  = weights['expr']  * model.betas[expr_idx] 
    pose_err  = weights['pose']  * model.pose[3:] # exclude global rotation

    objectives = {'s2m': weights['s2m']*s2m, 'lmk': weights['lmk']*lmk_err, 'shape': shape_err, 'expr': expr_err, 'pose': pose_err} 

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
                 x0       = [ model.trans, model.pose[:3] ],
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


def run_fitting():
    # input scan
    scan_path = './data/scan.obj'

    # landmarks of the scan
    scan_lmk_path = './data/scan_lmks.npy'

    # measurement unit of landmarks ['m', 'cm', 'mm', 'NA'] 
    # When using option 'NA', the scale of the scan will be estimated by rigidly aligning model and scan landmarks
    scan_unit = 'm' 

    scan = Mesh(filename=scan_path)
    print("loaded scan from:", scan_path)

    lmk_3d = np.load(scan_lmk_path)
    print("loaded scan landmark from:", scan_lmk_path)

    # model
    model_path = './models/generic_model.pkl' # change to 'female_model.pkl' or 'male_model.pkl', if gender is known
    model = load_model(model_path)       # the loaded model object is a 'chumpy' object, check https://github.com/mattloper/chumpy for details
    print("loaded model from:", model_path)

    # landmark embedding
    lmk_emb_path = './models/flame_static_embedding.pkl' 
    lmk_face_idx, lmk_b_coords = load_embedding(lmk_emb_path)
    print("loaded lmk embedding")

    # scale scans and scan landmarks to be in the same local coordinate systems as the FLAME model
    if scan_unit.lower() == 'na':
        print('No scale specifiec - compute approximate scale based on the landmarks')
        scale_factor = compute_approx_scale(lmk_3d, model, lmk_face_idx, lmk_b_coords)
        print('Scale factor: %f' % scale_factor)
    else:
        scale_factor = get_unit_factor('m') / get_unit_factor(scan_unit)
        print('Scale factor: %f' % scale_factor)        

    scan.v[:] *= scale_factor
    lmk_3d[:] *= scale_factor

    # output
    output_dir = './output'
    safe_mkdir(output_dir)

    # weights
    weights = {}
    # scan vertex to model surface distance term
    weights['s2m']   = 2.0   
    # landmark term
    weights['lmk']   = 1e-2
    # shape regularizer (weight higher to regularize face shape more towards the mean)
    weights['shape'] = 1e-4
    # expression regularizer (weight higher to regularize facial expression more towards the mean)
    weights['expr']  = 1e-4
    # regularization of head rotation around the neck and jaw opening (weight higher for more regularization)
    weights['pose']  = 1e-3
    # Parameter of the Geman-McClure robustifier (higher weight for a larger bassin of attraction which makes it less robust to outliers)
    gmo_sigma = 1e-4

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
    mesh_v, mesh_f, parms = fit_scan(   scan=scan,                                             # input scan
                                        lmk_3d=lmk_3d,                                         # input landmark 3d
                                        model=model,                                           # model
                                        lmk_face_idx=lmk_face_idx, lmk_b_coords=lmk_b_coords,  # landmark embedding
                                        weights=weights,                                       # weights for the objectives
                                        gmo_sigma=gmo_sigma,                                   # parameter of the regularizer
                                        shape_num=300, expr_num=100, opt_options=opt_options ) # options

    # write result
    output_path = join( output_dir, 'fit_scan_result.obj' )
    write_simple_obj( mesh_v=mesh_v, mesh_f=mesh_f, filepath=output_path, verbose=False )
    print('output mesh saved to: ', output_path) 

    # output scaled scan for reference (output scan fit and the scan should be spatially aligned)
    output_path = join( output_dir, 'scan_scaled.obj' )    
    write_simple_obj( mesh_v=scan.v, mesh_f=scan.f, filepath=output_path, verbose=False )

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    run_fitting()
