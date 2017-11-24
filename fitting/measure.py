'''
Util funcitons for measurement
Tianye Li <tianye.li@tuebingen.mpg.de>
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
    