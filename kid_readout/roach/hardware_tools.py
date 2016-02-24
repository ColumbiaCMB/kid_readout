import numpy as np

import tools
from interface import RoachInterface
from heterodyne import RoachHeterodyne
from r2heterodyne import Roach2Heterodyne
from attenuator import Attenuator


def r2_with_board2(lo_freq=1000):
    attenuator = Attenuator()
    r2 = r2heterodyne.Roach2Heterodyne(roachip='r2kid', adc_valon = '/dev/ttyUSB4', lo_valon = '/dev/ttyUSB3', attenuator=attenuator)
    r2.set_lo(lo_freq)
    return r2
        
   
