__all__ = ['repmat', 'bsxfun', 'sub2ind', 'row', 'col', 'dot_v', 'dot_h', 'sparse']


def repmat(A, m, n):
    import numpy as np
    return np.tile(A, (m, n))


def bsxfun(oper, a, b):
    import numpy as np
    if a.shape[0] == b.shape[0] or a.shape[1] == b.shape[1]:
        return oper(a, b)
    elif min(a.shape) == 1 and min(b.shape) == 1:
        if a.shape[0] == 1:
            return oper(np.tile(a, (b.shape[0], 1)), b)
        else:
            return oper(np.tile(a, (1, b.shape[1], b)))
    else:
        raise '_bsxfun failure'


def sub2ind(siz, i, j):
    return j + i * siz[1]


def row(A):
    return A.reshape((1, -1))


def col(A):
    return A.reshape((-1, 1))


def dot_v(a, b):
    import numpy as np
    return np.sum(a * b, axis=0)


def dot_h(a, b):
    import numpy as np
    return np.sum(a * b, axis=1)


def sparse(i, j, data, m=None, n=None):
    import numpy as np
    from scipy.sparse import csc_matrix
    ij = np.vstack((i.flatten().reshape(1, -1), j.flatten().reshape(1, -1)))

    if m is None:
        return csc_matrix((data, ij))
    else:
        return csc_matrix((data, ij), shape=(m, n))
