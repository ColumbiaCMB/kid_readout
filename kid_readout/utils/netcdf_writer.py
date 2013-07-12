import numpy as np
import time
import sys
import netCDF4

class NetCDFWriter():
    def __init__(self, parent):
        self.parent = parent
        
    def write_data_chunk(self,chunk):
        pass