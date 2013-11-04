from __future__ import division

import numpy as np

from kid_readout.utils import easync

class Dataset(object):
    
    def __init__(self, filename):
        self.nc = easync.EasyNetCDF4(filename)
        
