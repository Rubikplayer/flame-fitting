'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this computer program. 
Using this computer program means that you agree to the terms in the LICENSE file (https://flame.is.tue.mpg.de/modellicense) included 
with the FLAME model. Any use not explicitly granted by the LICENSE is prohibited.

Copyright 2020 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its 
Max Planck Institute for Intelligent Systems. All rights reserved.

More information about FLAME is available at http://flame.is.tue.mpg.de.
For comments or questions, please email us at flame@tue.mpg.de
'''



# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.

# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os
import numpy as np
import pickle as pickle

# -----------------------------------------------------------------------------

def get_unit_factor(unit):
    if unit == 'mm':
        return 1000.0
    elif unit == 'cm':
        return 100.0
    elif unit == 'm':
        return 1.0
    else:
        raise ValueError('Unit not supported')

# -----------------------------------------------------------------------------

def load_binary_pickle( filepath ):
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding="latin1")
    return data

# -----------------------------------------------------------------------------

def save_binary_pickle( data, filepath ):
    with open( filepath, 'wb' ) as f:
        pickle.dump( data, f )

# -----------------------------------------------------------------------------

def load_simple_obj(filename):
    f = open(filename, 'r')

    def get_num(string, type):
        if type == 'int':
            return int(string)
        elif type == 'float':
            return float(string)
        else:
            print('Wrong type specified')

    vertices = []
    faces = []

    for line in f:
        str = line.split()
        if len(str) == 0:
            continue

        if str[0] ==  '#':
            continue
        elif str[0] == 'v':
            tmp_v = [get_num(s, 'float') for s in str[1:]]
            vertices.append( tmp_v )

        elif str[0] == 'f':
            tmp_f = [get_num(s, 'int')-1 for s in str[1:]]
            faces.append( tmp_f )

    f.close()
    return ( np.asarray(vertices), np.asarray(faces) )

# -----------------------------------------------------------------------------

def write_simple_obj( mesh_v, mesh_f, filepath, verbose=False ):
    with open( filepath, 'w') as fp:
        for v in mesh_v:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
        for f in mesh_f+1: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )
    if verbose:
        print('mesh saved to: ', filepath)

# -----------------------------------------------------------------------------

def safe_mkdir( file_dir ):
    if not os.path.exists( file_dir ):
        os.mkdir( file_dir ) 

