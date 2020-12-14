from Cython.Distutils import build_ext
import platform
import os
from distutils.core import setup, Extension
from copy import deepcopy
from numpy import get_include as numpy_include

sourcefiles = ['sample2meshdist.pyx']
additional_options = {'include_dirs': []}

if platform.system().lower() in ['darwin', 'linux']:
    import sysconfig
    extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
    extra_compile_args += ["-std=c++11"]
    additional_options['extra_compile_args'] = extra_compile_args

if platform.system().lower() in ['darwin']:
    extra_compile_args+=['-stdlib=libc++'] 
    extra_link_args=['-stdlib=libc++'] 

# Add path of EIGEN here
EIGEN_DIR = './eigen'

def setup_extended(parallel=True, numpy_includes=True, usr_local_includes=True, **kwargs):
    """Like "setup" from distutils
    Created by Matthew Loper on 2012-10-10.
    Copyright (c) 2012 MPI. All rights reserved.
    """

    kwargs = deepcopy(kwargs)
    if numpy_includes:
        for m in kwargs['ext_modules']:
            m.include_dirs.append(numpy_include())

    if usr_local_includes:
        for m in kwargs['ext_modules']:
            m.include_dirs.append('/usr/local/include')
    m.include_dirs.append(EIGEN_DIR)
    print(kwargs)
    setup(**kwargs)

setup_extended(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("sample2meshdist", sourcefiles, language="c++", **additional_options)],
    include_dirs=['.'],
)
