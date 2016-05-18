from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize("kid_readout/roach/decode.pyx"))

