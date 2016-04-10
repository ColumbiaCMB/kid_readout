__author__ = 'gjones'

import os
import time
from kid_readout.measurement.io.nc import IO as NCFile

from kid_readout.analysis.resources.local_settings import BASE_DATA_DIR

def new_nc_file(suffix='',directory=BASE_DATA_DIR):
    if not suffix.startswith('_'):
        suffix = '_' + suffix
    root_path = os.path.join(directory,time.strftime('%Y-%m-%d_%H%M%S')+suffix+'.nc')
    return NCFile(root_path)

