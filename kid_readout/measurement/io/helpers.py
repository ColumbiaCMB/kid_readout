import os
import time

from kid_readout.measurement.io.nc import NCFile
from kid_readout.settings import BASE_DATA_DIR

__author__ = 'gjones'


def new_nc_file(suffix='', directory=BASE_DATA_DIR):
    if suffix and not suffix.startswith('_'):
        suffix = '_' + suffix
    root_path = os.path.join(directory, time.strftime('%Y-%m-%d_%H%M%S') + suffix + '.nc')
    return NCFile(root_path)

