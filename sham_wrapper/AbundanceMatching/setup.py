import os
import numpy as np
from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize

sources = ['fiducial_deconv_wrapper.pyx', 'src/fiducial_deconvolute.c']
ext = Extension(name="fiducial_deconv_wrapper", sources=sources, 
    include_dirs=[np.get_include()])
setup(name="cython_wrapper",ext_modules = cythonize([ext]))
