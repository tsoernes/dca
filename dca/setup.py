from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from Cython.Distutils import build_ext

ext_modules = [Extension("gridfuncs", ["gridfuncs.pyx"])]

setup(
  name = 'DCA',
  cmdclass = {'build_ext': build_ext},
  include_dirs = [np.get_include()],
  ext_modules = ext_modules
)
