#!/usr/bin/env python

from chumpy import Ch, depends_on
import chumpy as ch
import numpy as np
from .alignment.objectives import sample_from_mesh
from .matlab.matlab import row, col
from .alignment.mesh_distance import sample2meshdist
from .robustifiers import SignedSqrt
import scipy.sparse as sp


def ScanToMesh(scan, mesh_verts, mesh_faces, rho=lambda x : x, scan_sampler=None, normalize=True, signed=False):
    """Returns a Ch object whose only dterm is 'mesh_verts'"""

    if scan_sampler is None:
        scan_sampler = scan

    sampler, n_samples = construct_sampler(scan_sampler, scan.v.size / 3)

    norm_const = np.sqrt(n_samples) if normalize else 1

    if signed:
        fn = lambda x : SignedSqrt(rho(x)) / norm_const
    else:
        fn = lambda x : ch.sqrt(rho(x)) / norm_const

    result = Ch(lambda mesh_verts : fn(MeshDistanceSquared(
        sample_verts=scan.v,
        sample_faces=scan.f,
        reference_verts=mesh_verts,
        reference_faces=mesh_faces,
        sampler=sampler,
        signed=signed
        )))

    result.mesh_verts = mesh_verts
    return result

def MeshToScan(scan, mesh_verts, mesh_faces, mesh_template_or_sampler, rho=lambda x : x, normalize=True, signed=False):
    """Returns a Ch object whose only dterm is 'mesh_verts'"""

    sampler, n_samples = construct_sampler(mesh_template_or_sampler, mesh_verts.size / 3)

    norm_const = np.sqrt(n_samples) if normalize else 1

    if signed:
        fn = lambda x : SignedSqrt(rho(x)) / norm_const
    else:
        fn = lambda x : ch.sqrt(rho(x)) / norm_const

    result = Ch(lambda mesh_verts : fn(MeshDistanceSquared(
        sample_verts=mesh_verts,
        sample_faces=mesh_faces,
        reference_verts=scan.v,
        reference_faces=scan.f,
        sampler=sampler,
        signed=signed
        )))

    result.mesh_verts = mesh_verts
    return result

def PtsToMesh(sample_verts, reference_verts, reference_faces, reference_template_or_sampler, rho=lambda x : x, normalize=True, signed=False):
    """Returns a Ch object whose dterms are 'reference_v' and 'sample_v'"""

    sampler = {'point2sample': sp.eye(sample_verts.size, sample_verts.size)}
    n_samples = sample_verts.size / 3

    norm_const = np.sqrt(n_samples) if normalize else 1

    if signed:
        fn = lambda x : SignedSqrt(rho(x)) / norm_const
    else:
        fn = lambda x : ch.sqrt(rho(x)) / norm_const

    result = Ch(lambda sample_v, reference_v : fn(MeshDistanceSquared(
        sample_verts=sample_v,
        reference_verts=reference_v,
        reference_faces=reference_faces,
        sampler=sampler,
        signed=signed
        )))

    result.reference_v = reference_verts
    result.sample_v = sample_verts
    return result

class ClampedSignedPtsToMesh(ch.Ch):
    dterms = 'reference_v', 'sample_v'
    terms = 'a_min', 'a_max', 'reference_f'

    def on_changed(self, which):
        dist = PtsToMesh(
            sample_verts=self.sample_v, reference_verts=self.reference_v, reference_faces=self.reference_f,
            signed=True, normalize=False)

        self.which_idxs = np.nonzero((dist.r >= self.a_min) & (dist.r <= self.a_max))[0]
        self.sparse_dist = PtsToMesh(
            sample_verts=np.asarray(self.sample_v)[self.which_idxs], reference_verts=self.reference_v, reference_faces=self.reference_f,
            signed=True, normalize=False)

        self.dist = ch.clip(dist, self.a_min, self.a_max)

    def compute_r(self):
        result = self.dist.r
        return result

    def compute_dr_wrt(self, wrt):
        if wrt is self.reference_v:
            tmp = self.sparse_dist.dr_wrt(self.sparse_dist.reference_v).tocoo()
            data = tmp.data
            IS = self.which_idxs[tmp.row]
            JS = tmp.col

            return sp.csc_matrix((tmp.data, (IS, JS)), shape=(self.r.size, wrt.size))
        elif wrt is self.sample_v:
            return self.dist.dr_wrt(wrt)

def construct_sampler(sampler_or_template, num_mesh_verts):
    if isinstance(sampler_or_template, dict):
        sampler = sampler_or_template
    else:
        sampler = sample_from_mesh(sampler_or_template, sample_type='uniformly-from-vertices', num_samples=1e+5)

    n_samples = sampler['point2sample'].shape[0] / 3 if 'point2sample' in sampler else num_mesh_verts
    return sampler, n_samples

class MeshDistanceSquared(Ch):
    terms = 'sampler', 'sample_faces', 'reference_faces', 'signed'
    dterms = 'sample_verts', 'reference_verts'

    def compute_r(self):
        result = np.sum((self.diff) ** 2, axis=1).ravel()
        if self.signed:
            result *= self.direction
        return result

    def compute_dr_wrt(self, wrt):
        if wrt not in (self.sample_verts, self.reference_verts):
            return

        if wrt is self.reference_verts:
            r, Dr_ref, Dr_sample = sample2meshdist.squared_distance(self.nearest_tri, self.nearest_part, self.reference_faces,
                self.reference_verts.r.reshape((-1, 3)), self.sample_points, compute_dref=True, compute_dsample=False)
            result = Dr_ref
        elif wrt is self.sample_verts:
            r, Dr_ref, Dr_sample = sample2meshdist.squared_distance(self.nearest_tri, self.nearest_part, self.reference_faces,
                self.reference_verts.r.reshape((-1, 3)), self.sample_points, compute_dref=False, compute_dsample=True, dsample_pattern=self.dsample_pattern)

            # this dot product takes about half the time in this function call. can it be fixed?
            result = Dr_sample.dot(self.ss_point2sample)

        if self.signed:
            result = sp.spdiags(self.direction, [0], self.direction.size, self.direction.size).dot(result)
        return result


    @depends_on(terms + dterms)
    def direction(self):
        from body.ch.ch_vert_normals import VertNormals, TriNormals
        fn = TriNormals(v=self.reference_verts, f=self.reference_faces).r.reshape((-1, 3))
        vn = VertNormals(f=self.reference_faces, num_verts=self.reference_verts.shape[0], v=self.reference_verts).r.reshape((-1, 3))

        nearest_normals = np.zeros_like(self.sample_points)

        "nearest_part tells you whether the closest point in triangle abc is in the interior (0), on an edge (ab:1,bc:2,ca:3), or a vertex (a:4,b:5,c:6)"
        cl_tri_idxs = np.nonzero(self.nearest_part == 0)[0]
        cl_vrt_idxs = np.nonzero(self.nearest_part > 3)[0]
        cl_edg_idxs = np.nonzero((self.nearest_part <= 3) & (self.nearest_part > 0))[0]

        nearest_tri = self.nearest_tri[cl_tri_idxs]
        nearest_normals[cl_tri_idxs] = fn[nearest_tri]

        nearest_tri = self.nearest_tri[cl_vrt_idxs]
        nearest_part = self.nearest_part[cl_vrt_idxs] - 4
        nearest_normals[cl_vrt_idxs] = vn[self.reference_faces[nearest_tri, nearest_part]]

        nearest_tri = self.nearest_tri[cl_edg_idxs]

        nearest_part = np.asarray(self.nearest_part[cl_edg_idxs] - 1, dtype=np.int64)
        nearest_normals[cl_edg_idxs] += vn[self.reference_faces[nearest_tri, nearest_part]]

        nearest_part = np.asarray(np.mod(self.nearest_part[cl_edg_idxs], 3), dtype=np.int64)
        nearest_normals[cl_edg_idxs] += vn[self.reference_faces[nearest_tri, nearest_part]]

        direction = np.sign(np.sum(self.diff * nearest_normals, axis=1))

        return direction.ravel()

    def on_changed(self, which):

        if not hasattr(self, 'signed'):
            self.signed = False

        if 'sample_verts' in which:
            assert(len(np.nonzero(np.isnan(self.sample_verts.r.ravel()))[0]) == 0)
            assert(len(np.nonzero(np.isinf(self.sample_verts.r.ravel()))[0]) == 0)

        if 'reference_verts' in which:
            assert(len(np.nonzero(np.isnan(self.reference_verts.r.ravel()))[0]) == 0)
            assert(len(np.nonzero(np.isinf(self.reference_verts.r.ravel()))[0]) == 0)

        # If we don't have a sampler, assign one automatically
        if not hasattr(self, 'sampler') or self.sampler is None:
            from psbody.mesh import Mesh
            sample_mesh = Mesh(v=self.sample_verts.r.reshape((-1, 3)), f=self.sample_faces)
            self.sampler = sample_from_mesh(sample_mesh, sample_type='uniformly-from-vertices', num_samples=1e+5)
            which.add('sampler')

        # Recompute the aabb tree for the reference mesh
        # (TODO: consider precomputing normals for reference mesh)
        if 'reference_verts' in which or 'reference_faces' in which:
            self.tree = _AabbTree(self.reference_verts.r.reshape((-1, 3)), self.reference_faces)

        if self.sampler['point2sample'] is not None:
            self.ss_point2sample = self.sampler['point2sample']
            # get points sampled from sample mesh
            self.sample_points = self.ss_point2sample.dot(col(self.sample_verts.r)).reshape(-1, 3)
        else:
            self.sample_points = sample_mesh
        self.num_sample_points = self.sample_points.shape[0]
        self.dsample_pattern = self.sampler.get('dsample_pattern', {})

        # For each sample point in the sample mesh, figure out which primitives
        # are nearest: vertices, edges, or triangles.
        self.nearest_tri, self.nearest_part, self.nearest_point = self.tree.nearest(self.sample_points, nearest_part=True)

        # fix types/shapes for r/c code
        self.nearest_tri = self.nearest_tri.ravel().astype(np.uint64)
        self.nearest_part = self.nearest_part.ravel().astype(np.uint64)
        self.reference_faces = self.reference_faces.astype(np.uint64)

        self.diff = self.sample_points - self.nearest_point




class _AabbTree(object):
    """Encapsulates an AABB (Axis Aligned Bounding Box) Tree """
    def __init__(self, v, f):
        import psbody.mesh.spatialsearch as spatialsearch
        self.cpp_handle = spatialsearch.aabbtree_compute(v.astype(np.float64).copy(order='C'), f.astype(np.uint32).copy(order='C'))

        if True:  # FOR PICKLING TEST
            self.vv = v
            self.ff = f

    def nearest(self, v_samples, nearest_part=False):
        "nearest_part tells you whether the closest point in triangle abc is in the interior (0), on an edge (ab:1,bc:2,ca:3), or a vertex (a:4,b:5,c:6)"
        import psbody.mesh.spatialsearch as spatialsearch
        f_idxs, f_part, v = spatialsearch.aabbtree_nearest(self.cpp_handle, np.array(v_samples, dtype=np.float64, order='C'))

        # if False:  # FOR VISUALIZING CORRESPONDENCES
        #     from psbody.mesh import Mesh
        #     from psbody.mesh.meshviewer import MeshViewer
        #     from psbody.mesh.lines import Lines
        #     mv = MeshViewer()
        #     mv.static_meshes = [Mesh(v=self.vv, f=[], vc=self.vv * 0 + .5)]
        #     v2 = np.hstack((v_samples, v)).reshape((-1, 3))
        #     e = np.arange(v2.shape[0]).reshape((-1, 2))
        #     ll = Lines(v=v2, e=e)
        #     mv.static_lines = [ll]
        #     import pdb; pdb.set_trace()


        return (f_idxs, f_part, v) if nearest_part else (f_idxs, v)

    def __getstate__(self):
        # This tree object is not pickable. We store the data to recreate
        # print "_AabbTree being pickled"
        pickable_dict = dict()
        pickable_dict['vv'] = self.vv
        pickable_dict['ff'] = self.ff

        return pickable_dict

    def __setstate__(self, d):
        # print "_AabbTree being unpickled"

        self.vv = d['vv']
        self.ff = d['ff']

        import psbody.mesh.spatialsearch as spatialsearch
        self.cpp_handle = spatialsearch.aabbtree_compute(self.vv.astype(np.float64).copy(order='C'), self.ff.astype(np.uint32).copy(order='C'))


def main():
    pass


if __name__ == '__main__':
    main()
