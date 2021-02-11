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
import pickle as pickle

from fitting.util import load_binary_pickle

# -----------------------------------------------------------------------------

def merge_meshes(mesh_list):
    v = mesh_list[0].v
    f = mesh_list[0].f
    i = v.shape[0]
    for m in mesh_list[1:]:
        v = np.vstack((v, m.v))
        f = np.vstack((f, i+m.f))
        i = v.shape[0]
    return Mesh(v, f)

# -----------------------------------------------------------------------------

def landmarks_to_mesh(lmks, radius=0.001):
    meshes = []
    for lmk in lmks:
        sph = Sphere(lmk, radius).to_mesh([255.0, 0.0, 0.0])
        meshes.append(sph)
    return merge_meshes(meshes)

# -----------------------------------------------------------------------------

def load_embedding( file_path ):
    """ funciton: load landmark embedding, in terms of face indices and barycentric coordinates for corresponding landmarks
    note: the included example is corresponding to CMU IntraFace 49-point landmark format.
    """
    lmk_indexes_dict = load_binary_pickle( file_path )
    lmk_face_idx = lmk_indexes_dict[ 'lmk_face_idx' ].astype( np.uint32 )
    lmk_b_coords = lmk_indexes_dict[ 'lmk_b_coords' ]
    return lmk_face_idx, lmk_b_coords

# -----------------------------------------------------------------------------

def mesh_points_by_barycentric_coordinates( mesh_verts, mesh_faces, lmk_face_idx, lmk_b_coords ):
    """ function: evaluation 3d points given mesh and landmark embedding
    """
    dif1 = ch.vstack([(mesh_verts[mesh_faces[lmk_face_idx], 0] * lmk_b_coords).sum(axis=1),
                    (mesh_verts[mesh_faces[lmk_face_idx], 1] * lmk_b_coords).sum(axis=1),
                    (mesh_verts[mesh_faces[lmk_face_idx], 2] * lmk_b_coords).sum(axis=1)]).T
    return dif1

# -----------------------------------------------------------------------------

def landmark_error_3d( mesh_verts, mesh_faces, lmk_3d, lmk_face_idx, lmk_b_coords, weight=1.0 ):
    """ function: 3d landmark error objective
    """

    # select corresponding vertices
    v_selected = mesh_points_by_barycentric_coordinates( mesh_verts, mesh_faces, lmk_face_idx, lmk_b_coords )
    lmk_num  = lmk_face_idx.shape[0]

    # an index to select which landmark to use
    lmk_selection = np.arange(0,lmk_num).ravel() # use all

    # residual vectors
    lmk3d_obj = weight * ( v_selected[lmk_selection] - lmk_3d[lmk_selection] )

    return lmk3d_obj

# -----------------------------------------------------------------------------

def load_picked_points(filename):
    """
    Load a picked points file (.pp) containing 3D points exported from MeshLab.
    Returns a Numpy array of size Nx3
    """

    f = open(filename, 'r')

    def get_num(string):
        pos1 = string.find('\"')
        pos2 = string.find('\"', pos1 + 1)
        return float(string[pos1 + 1:pos2])

    def get_point(str_array):
        if 'x=' in str_array[0] and 'y=' in str_array[1] and 'z=' in str_array[2]:
            return [get_num(str_array[0]), get_num(str_array[1]), get_num(str_array[2])]
        else:
            return []

    pickedPoints = []
    for line in f:
        if 'point' in line:
            str = line.split()
            if len(str) < 4:
                continue
            ix = [i for i, s in enumerate(str) if 'x=' in s][0]
            iy = [i for i, s in enumerate(str) if 'y=' in s][0]
            iz = [i for i, s in enumerate(str) if 'z=' in s][0]
            pickedPoints.append(get_point([str[ix], str[iy], str[iz]]))
    f.close()
    return np.array(pickedPoints)
