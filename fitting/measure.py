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
import matplotlib as mpl
import matplotlib.cm as cm

# -----------------------------------------------------------------------------

def mesh2mesh( mesh_v_1, mesh_v_2 ):
	return dist = np.linalg.norm( mesh_v_2 - mesh_v_1, axis=1 )

# -----------------------------------------------------------------------------

def distance2color( dist, vmin=0, vmax=0.001, cmap_name='jet' ):
    # vmin, vmax in meters
    norm = mpl.colors.Normalize( vmin=vmin, vmax=vmax )
    cmap = cm.get_cmap( name=cmap_name )
    colormapper = cm.ScalarMappable( norm=norm, cmap=cmap )
    rgba = colormapper.to_rgba( dist )
    color_3d = rgba[:,0:3]
    return color_3d
    