#__all__ = ['landmark_function', 'scan_to_mesh_squared_function',
#           'mesh_to_scan_squared_function', 'scan_to_mesh_function',
#           'mesh_to_scan_function', 'full_mesh_to_scan_function', 'vecdiff_function']

from scipy.sparse import csc_matrix
from scipy import array
import scipy as sy
import scipy.sparse as sp
import numpy as np
import sbody.alignment.mesh_distance.mesh_distance as mesh_distance
import random
import copy

from sbody.matlab import *

def co3(x):
    return matlab.bsxfun(np.add, row(np.arange(3)), col(3 * (x)))


def triangle_area(v, f):
    return np.sqrt(np.sum(np.cross(v[f[:, 1], :] - v[f[:, 0], :], v[f[:, 2], :] - v[f[:, 0], :]) ** 2, axis=1)) / 2


def sample_categorical(samples, dist):
    a = np.random.multinomial(samples, dist)
    b = np.zeros(int(samples), dtype=int)
    upper = np.cumsum(a)
    lower = upper - a
    for value in range(len(a)):
        b[lower[value]: upper[value]] = value
    np.random.shuffle(b)
    return b


def sample_from_mesh(mesh, sample_type='edge_midpoints', num_samples=10000, vertex_indices_to_sample=None, seed=0):

    # print 'WARNING: sample_from_mesh needs testing, especially with edge-midpoints and uniformly-at-random'
    if sample_type == 'vertices':
        if vertex_indices_to_sample is None:
            sample_spec = {'point2sample': sy.sparse.eye(mesh.v.size, mesh.v.size)}  # @UndefinedVariable
        else:
            sample_ind = vertex_indices_to_sample
            IS = co3(array(range(0, sample_ind.size)))
            JS = co3(sample_ind)
            VS = np.ones(IS.size)
            point2sample = matlab.sparse(IS.flatten(), JS.flatten(), VS.flatten(), 3 * sample_ind.size, 3 * mesh.v.shape[0])
            sample_spec = {'point2sample': point2sample}
    elif sample_type == 'uniformly-from-vertices':
        # Note: this will never oversample: when num_samples is greater than number of verts,
        # then the vert indices are all included (albeit shuffled), and none left out
        # (because of how random.sample works)

        #print("SEED IS", seed, 'SIZE is', mesh.v.shape[0], '#elements is', int(min(num_samples, mesh.v.shape[0])))
        random.seed(seed)  # XXX uncomment when not debugging
        np.random.seed(seed)
        #sample_ind = np.array(random.sample(xrange(mesh.v.shape[0]), int(min(num_samples, mesh.v.shape[0]))))
        sample_ind = np.array(random.sample(range(mesh.v.shape[0]), int(min(num_samples, mesh.v.shape[0]))))
        #print("FIRST ELEMENTS ARE", sample_ind[:100])
        IS = co3(array(range(0, sample_ind.size)))
        JS = co3(sample_ind)
        VS = np.ones(IS.size)
        point2sample = matlab.sparse(IS.flatten(), JS.flatten(), VS.flatten(), 3 * sample_ind.size, 3 * mesh.v.shape[0])
        sample_spec = {'point2sample': point2sample}
    else:
        if sample_type == 'edge-midpoints':
            tri = np.tile(array(range(0, mesh.f.size[0])).reshape(-1, 1), 1, 3).flatten()
            IS = array(range(0, tri.size))
            JS = tri
            VS = np.ones(IS.size) / 3
            area2weight = matlab.sparse(IS, JS, VS, tri.size, mesh.f.shape[0])
            bary = np.tile([[.5, .5, 0], [.5, 0, .5], [0, .5, .5]], 1, mesh.f.shape[0])

        elif sample_type == 'uniformly-at-random':
            random.seed(seed)  # XXX uncomment when not debugging
            np.random.seed(seed)
            tri_areas = triangle_area(mesh.v, mesh.f)
            tri = sample_categorical(num_samples, tri_areas / tri_areas.sum())
            bary = np.random.rand(tri.size, 3)
            flip = np.sum(bary[:, 0:1] > 1)
            bary[flip, :2] = 1 - bary[flip, 1::-1]
            bary[:, 2] = 1 - np.sum(bary[:, :2], 1)
            area2weight = sy.sparse.eye(tri.size, tri.size)  # @UndefinedVariable
        else:
            raise 'Unknown sample_type'

        IS = []
        JS = []
        VS = []
        S = tri.size
        V = mesh.v.size / 3
        for cc in range(0, 3):
            for vv in range(0, 3):
                IS.append(np.arange(cc, 3 * S, 3))
                JS.append(cc + 3 * mesh.f[tri, vv])
                VS.append(bary[:, vv])

        IS = np.concatenate(IS)
        JS = np.concatenate(JS)
        VS = np.concatenate(VS)

        point2sample = matlab.sparse(IS, JS, VS, 3 * S, 3 * V)
        sample_spec = {'area2weight': area2weight, 'point2sample': point2sample, 'tri': tri, 'bary': bary}
    return sample_spec
