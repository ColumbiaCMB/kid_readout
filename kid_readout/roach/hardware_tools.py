from heterodyne import RoachHeterodyne
from r2heterodyne import Roach2Heterodyne
from attenuator import Attenuator
from kid_readout.settings import *


def r2_with_mk2(**kwargs):
    attenuator = Attenuator()
    if MARK2_VALON is None:  # If lo_valon is None the roach will use its internal valon, with unfortunate results.
        raise ValueError("MARK2_VALON is None.")
    r2 = Roach2Heterodyne(roachip=ROACH2_IP, adc_valon=ROACH2_VALON, host_ip=ROACH2_GBE_HOST_IP, lo_valon=MARK2_VALON,
                          attenuator=attenuator,**kwargs)
    return r2


def r2_with_mk1(**kwargs):
    r2 = Roach2Heterodyne(roachip=ROACH2_IP, adc_valon=ROACH2_VALON, host_ip=ROACH2_GBE_HOST_IP, lo_valon=None,
                          attenuator=None,**kwargs)
    return r2

   
def r1_with_mk2(**kwargs):
    attenuator = Attenuator()
    if MARK2_VALON is None:  # If lo_valon is None the roach will use its internal valon, with unfortunate results.
        raise ValueError("MARK2_VALON is None.")
    r1 = RoachHeterodyne(roachip=ROACH1_IP, adc_valon=ROACH1_VALON, host_ip=ROACH1_HOST_IP, lo_valon=MARK2_VALON,
                         attenuator=attenuator, **kwargs)
    return r1
    

def r1_with_mk1(**kwargs):
    r1 = RoachHeterodyne(roachip=ROACH1_IP, adc_valon=ROACH1_VALON, host_ip=ROACH1_HOST_IP, lo_valon=None,
                         attenuator=None,**kwargs)
    return r1
