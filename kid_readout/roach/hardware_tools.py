import numpy as np

import tools
from interface import RoachInterface
from heterodyne import RoachHeterodyne
from r2heterodyne import Roach2Heterodyne
from attenuator import Attenuator


def r2_with_board2(lo_freq=1000, atten=10):
    attenuator = Attenuator(attenuation=atten)
    r2 = Roach2Heterodyne(roachip='r2kid', adc_valon = '/dev/ttyUSB4', lo_valon = '/dev/ttyUSB3', attenuator=attenuator)
    r2.set_lo(lo_freq)
    return r2
        
   
def r1_with_board2(lo_freq=1000, atten=10):
    attenuator = Attenuator(attenuation=atten)
    r1 = RoachHeterodyne(roachip='roach', adc_valon='/dev/ttyUSB0', lo_valon='/dev/ttyUSB3', attenuator=attenuator)
    r1.set_lo(lo_freq)
    return r1
    
