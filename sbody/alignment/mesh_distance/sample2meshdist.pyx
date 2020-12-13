# cython: profile=True
import numpy as n
import scipy.sparse

cimport numpy as n
cimport cython
from cython cimport parallel

# maybe there's a better way than copy/paste to deal with template instantiation?
cdef extern from "sample2meshdist.h" namespace "instances":
    cdef cppclass Distance:
        double plane(double* x, double* a, double* b, double* c, 
                   double* dx, double* da, double* db, double* dc) nogil
        double line(double* x, double* a, double* b,  
                   double* dx, double* da, double* db) nogil
        double point(double* x, double* a,
                   double* dx, double *da) nogil
        double tri(int part, double* x, double* a, double* b, double* c, 
                   double* dx, double* da, double* db, double* dc) nogil
    
    cdef cppclass SquaredDistance:
        double plane(double* x, double* a, double* b, double* c, 
                   double* dx, double* da, double* db, double* dc) nogil
        double line(double* x, double* a, double* b,  
                   double* dx, double* da, double* db) nogil
        double point(double* x, double* a,
                   double* dx, double *da) nogil
        double tri(int part, double* x, double* a, double* b, double* c, 
                   double* dx, double* da, double* db, double* dc) nogil
     
    cdef cppclass GMDistance:
        GMDistance(double) except +
        double plane(double* x, double* a, double* b, double* c, 
                   double* dx, double* da, double* db, double* dc) nogil
        double line(double* x, double* a, double* b,  
                   double* dx, double* da, double* db) nogil
        double point(double* x, double* a,
                   double* dx, double *da) nogil
        double tri(int part, double* x, double* a, double* b, double* c, 
                   double* dx, double* da, double* db, double* dc) nogil

ctypedef fused SomeDistance:
    Distance
    SquaredDistance
    GMDistance

# computes [f(x1) f(x2) ... f(xn)] and derivatives where 
#   xi are samples
#   f is distance to the reference mesh, or squared/geman-mcclure distance according to the passed SomeDistance object
#   Always returns r,Dr_reference, Dr_samples. 
#   If compute_dref or compute_dsample is False, None will be returned in place of the corresponding derivative
from libcpp cimport bool
@cython.boundscheck(False)
@cython.wraparound(False)
cdef somedistance(SomeDistance distance_to,
         n.ndarray[n.uint64_t,ndim=1] nearest_tri, 
         n.ndarray[n.uint64_t,ndim=1] nearest_part,
         n.ndarray[n.uint64_t,ndim=2] ref_tri,
         n.ndarray[n.float64_t,ndim=2] ref_v,
         n.ndarray[n.float64_t,ndim=2] sample_v,
         bool compute_dref, bool compute_dsample,
         dsample_indices,
         dsample_indptr,
         dsample_sortdata):

    cdef int S = nearest_tri.size
    cdef n.ndarray[n.float64_t, ndim=1] r = n.zeros((S,))

    # Dr_refv
    cdef n.ndarray[n.uint64_t, ndim=1] dref_i = n.empty((9*S,), dtype=n.uint64)
    cdef n.ndarray[n.uint64_t, ndim=1] dref_j = n.empty((9*S,), dtype=n.uint64)
    cdef n.ndarray[n.float64_t, ndim=1] dref_v = n.zeros((9*S,))
    
    # Dr_samplev
    cdef n.ndarray[n.float64_t, ndim=1] dsample_v = n.zeros((3*S,))
    
    cdef int three = 3
    cdef int ss, vv, cc, tt
    for ss in parallel.prange(S, nogil=True):
        tt = nearest_tri[ss]    
        r[ss] = distance_to.tri(nearest_part[ss], &sample_v[ss,0], 
                &ref_v[ref_tri[tt,0],0], &ref_v[ref_tri[tt,1],0], &ref_v[ref_tri[tt,2],0],
                &dsample_v[ss*3] if compute_dsample else NULL, 
                &dref_v[ss*9+0] if compute_dref else NULL,
                &dref_v[ss*9+3] if compute_dref else NULL, 
                &dref_v[ss*9+6] if compute_dref else NULL)
        if compute_dref:
            for vv in range(three):
                for cc in range(three):
                    dref_i[9*ss + 3*vv + cc] = ss
                    dref_j[9*ss + 3*vv + cc] = 3*ref_tri[tt,vv]+cc
    Dr_refv, Dr_samplev = None, None
    if compute_dref:
        Dr_refv = scipy.sparse.coo_matrix((dref_v,(dref_i,dref_j)), (S,ref_v.size)).tocsc() 
    if compute_dsample:
        if dsample_indices is not None and dsample_indptr is not None and dsample_sortdata is not None: 
            # if we have precomputed the csc structure, use it
            Dr_samplev = scipy.sparse.csc_matrix((dsample_v[dsample_sortdata], dsample_indices, dsample_indptr), (S,3*S))
        else:
            js = n.arange(3*S)
            Dr_samplev = scipy.sparse.coo_matrix((dsample_v, (n.floor_divide(js,3), js)), (S,3*S)).tocsc()

    return r, Dr_refv, Dr_samplev

# as above, but computes the scalar function sum_i f(x_i) and its derivatives
@cython.boundscheck(False)
@cython.wraparound(False)
cdef somedistance_scalar(SomeDistance distance_to,
         n.ndarray[n.uint64_t,ndim=1] nearest_tri, 
         n.ndarray[n.uint64_t,ndim=1] nearest_part,
         n.ndarray[n.uint64_t,ndim=2] ref_tri,
         n.ndarray[n.float64_t,ndim=2] ref_v,
         n.ndarray[n.float64_t,ndim=2] sample_v):

    cdef double f = 0
    cdef n.ndarray[n.float64_t, ndim=1] dref = n.zeros((ref_v.size,))
    cdef n.ndarray[n.float64_t, ndim=1] dsample = n.zeros((sample_v.size,))
    
    cdef int ss, tt
    cdef int S = nearest_tri.size
    for ss in range(S):
        tt = nearest_tri[ss]    
        f += distance_to.tri(nearest_part[ss], &sample_v[ss,0], 
                &ref_v[ref_tri[tt,0],0], &ref_v[ref_tri[tt,1],0], &ref_v[ref_tri[tt,2],0],
                &dsample[3*ss], &dref[3*ref_tri[tt,0]], &dref[3*ref_tri[tt,1]], &dref[3*ref_tri[tt,2]])
    return f, dref, dsample

## type-specializations to make this stuff available in Python
## geman-mcclure stuff untested
def distance(nearest_tri, nearest_part, ref_tri, ref_v, sample_v, compute_dref=True, compute_dsample=True, dsample_pattern={}):
    cdef Distance f
    return somedistance[Distance](f, nearest_tri, nearest_part, ref_tri, ref_v, sample_v, compute_dref, compute_dsample,
        dsample_pattern.get('indices'), dsample_pattern.get('indptr'), dsample_pattern.get('sortdata'))

def squared_distance(nearest_tri, nearest_part, ref_tri, ref_v, sample_v, compute_dref=True, compute_dsample=True, dsample_pattern={}):
    cdef SquaredDistance f
    return somedistance[SquaredDistance](f, nearest_tri, nearest_part, ref_tri, ref_v, sample_v, compute_dref, compute_dsample,
        dsample_pattern.get('indices'), dsample_pattern.get('indptr'), dsample_pattern.get('sortdata'))

def gm_distance(sigma, nearest_tri, nearest_part, ref_tri, ref_v, sample_v, compute_dref=True, compute_dsample=True, dsample_pattern={}):
    cdef GMDistance* f = new GMDistance(sigma)
    return somedistance[GMDistance](cython.operator.dereference(f), nearest_tri, nearest_part, ref_tri, ref_v, sample_v, compute_dref, compute_dsample,
            dsample_pattern.get('indices'), dsample_pattern.get('indptr'), dsample_pattern.get('sortdata'))


def squared_distance_scalar(nearest_tri, nearest_part, ref_tri, ref_v, sample_v):
    cdef SquaredDistance f
    return somedistance_scalar[SquaredDistance](f, nearest_tri, nearest_part, ref_tri, ref_v, sample_v)

def gm_distance_scalar(sigma, nearest_tri, nearest_part, ref_tri, ref_v, sample_v):
    cdef GMDistance* f = new GMDistance(sigma)
    return somedistance_scalar[GMDistance](cython.operator.dereference(f), nearest_tri, nearest_part, ref_tri, ref_v, sample_v)

# simple point-primitive distances for testing
def pointPlaneDistance(n.ndarray[n.float64_t, ndim=1] x,
                       n.ndarray[n.float64_t, ndim=1] a,
                       n.ndarray[n.float64_t, ndim=1] b,
                       n.ndarray[n.float64_t, ndim=1] c):
    cdef Distance f
    cdef n.ndarray[n.float64_t, ndim=1] dx=n.zeros((3,)), da=n.zeros((3,)), db=n.zeros((3,)), dc=n.zeros((3,))
    fv = f.plane(&x[0], &a[0], &b[0], &c[0], &dx[0], &da[0], &db[0], &dc[0])
    return fv, dx, da, db, dc
def pointLineDistance(n.ndarray[n.float64_t, ndim=1] x,
                      n.ndarray[n.float64_t, ndim=1] a,
                      n.ndarray[n.float64_t, ndim=1] b):
    cdef Distance f
    cdef n.ndarray[n.float64_t, ndim=1] dx=n.zeros((3,)), da=n.zeros((3,)), db=n.zeros((3,))
    fv = f.line(&x[0], &a[0], &b[0], &dx[0], &da[0], &db[0])
    return fv, dx, da, db
def pointPointDistance(n.ndarray[n.float64_t, ndim=1] x,
                        n.ndarray[n.float64_t, ndim=1] a):
    cdef Distance f
    cdef n.ndarray[n.float64_t, ndim=1] dx=n.zeros((3,)), da=n.zeros((3,))
    fv = f.point(&x[0], &a[0], &dx[0], &da[0])
    return fv, dx, da


