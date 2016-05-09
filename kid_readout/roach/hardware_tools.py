import numpy as np

from interface import RoachInterface
from heterodyne import RoachHeterodyne
from r2heterodyne import Roach2Heterodyne
from attenuator import Attenuator
from kid_readout.settings import ROACH1_IP, ROACH1_VALON, ROACH2_IP, ROACH2_VALON, MARK2_VALON


def r2_with_mk2(lo_freq=1000, atten=None):
    attenuator = Attenuator(attenuation=atten)
    r2 = Roach2Heterodyne(roachip=ROACH2_IP, adc_valon=ROACH2_VALON, lo_valon=MARK2_VALON, attenuator=attenuator)
    r2.set_lo(lo_freq)
    return r2
        
   
def r1_with_mk2(lo_freq=1000, atten=None):
    attenuator = Attenuator(attenuation=atten)
    r1 = RoachHeterodyne(roachip=ROACH1_IP, adc_valon=ROACH1_VALON, lo_valon=MARK2_VALON, attenuator=attenuator)
    r1.initialize()
    r1.set_lo(lo_freq)
    return r1
    

def r1_with_mk1(lo_freq=1000):
    r1 = RoachHeterodyne(roachip=ROACH1_IP, adc_valon=ROACH1_VALON, lo_valon=None, attenuator=None)
    r1.initialize()
    r1.set_lo(lo_freq)
    return r1
