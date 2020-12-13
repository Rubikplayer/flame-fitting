__all__ = ['SampleMeshDistanceSquared']

import numpy as n
from sbody.alignment.mesh_distance import sample2meshdist
from sbody.matlab.matlab import col
# from functools import cached_property
from werkzeug.utils import cached_property

class SampleMeshDistanceSquared(object):
    def __init__(self, sample_mesh, sample_spec, reference_mesh):
        self.reference_mesh = reference_mesh
        self.sample_mesh = sample_mesh
        if not hasattr(reference_mesh, 'tree'):
            self.reference_mesh.tree = reference_mesh.compute_aabb_tree()

        if sample_spec['point2sample'] is not None:
            self.ss_point2sample = sample_spec['point2sample']
            # get points sampled from sample mesh
            self.sample_points = self.ss_point2sample.dot(col(sample_mesh.v)).reshape(-1, 3)
        else:
            self.sample_points = sample_mesh
        self.num_sample_points = self.sample_points.shape[0]
        self.dsample_pattern = sample_spec.get('dsample_pattern', {})

        # For each sample point in the sample mesh, figure out which primitives
        # are nearest: vertices, edges, or triangles.
        self.nearest_tri, self.nearest_part, self.nearest_point = self.reference_mesh.tree.nearest(self.sample_points, nearest_part=True)

        # fix types/shapes for r/c code
        self.nearest_tri = self.nearest_tri.flatten().astype(n.uint64)
        self.nearest_part = self.nearest_part.flatten().astype(n.uint64)
        self.reference_mesh.f = self.reference_mesh.f.astype(n.uint64)

    def _setup_for_derivative_computation(self):
        # exists for api compatibility with mesh_distance_lazy
        return

    @cached_property
    def r(self):
        return n.sum((self.sample_points - self.nearest_point) ** 2, axis=1)

    @cached_property
    def dr_reference_mesh(self):
        r, Dr_ref, Dr_sample = sample2meshdist.squared_distance(self.nearest_tri,
                                                                self.nearest_part,
                                                                self.reference_mesh.f,
                                                                self.reference_mesh.v,
                                                                self.sample_points,
                                                                compute_dref=True,
                                                                compute_dsample=False)
        return Dr_ref

    #@cached_property
    def dr_sample_mesh(self):
        r, Dr_ref, Dr_sample = sample2meshdist.squared_distance(self.nearest_tri,
                                                                self.nearest_part,
                                                                self.reference_mesh.f,
                                                                self.reference_mesh.v,
                                                                self.sample_points,
                                                                compute_dref=False,
                                                                compute_dsample=True,
                                                                dsample_pattern=self.dsample_pattern)
        # this dot product takes about half the time in this function call. can it be fixed?
        return Dr_sample.dot(self.ss_point2sample)
