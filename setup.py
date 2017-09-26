from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name='KID Readout',
      description='Code for the Columbia CMB ROACH KID readout',
      url='https://github.com/ColumbiaCMB/kid_readout',
      packages=['kid_readout'],
      ext_modules=cythonize('kid_readout/roach/decode.pyx'),
      include_dirs=[numpy.get_include()])
