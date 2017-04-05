from kid_readout.roach import attenuator, heterodyne, r2heterodyne
from kid_readout.settings import *


def r2_with_mk2(**kwargs):
    if MARK2_VALON is None:  # If lo_valon is None the roach will use its internal valon, with unfortunate results.
        raise ValueError("MARK2_VALON is None.")
    r2 = r2heterodyne.Roach2Heterodyne(roachip=ROACH2_IP, adc_valon=ROACH2_VALON, host_ip=ROACH2_GBE_HOST_IP,
                                       lo_valon=MARK2_VALON, attenuator=attenuator.Attenuator(), **kwargs)
    return r2


def r2_with_mk1(**kwargs):
    r2 = r2heterodyne.Roach2Heterodyne(roachip=ROACH2_IP, adc_valon=ROACH2_VALON, host_ip=ROACH2_GBE_HOST_IP,
                                       lo_valon=None, attenuator=None, **kwargs)
    return r2


def r1_with_mk2(**kwargs):
    return r1h14_with_mk2(**kwargs)


def r1_with_mk1(**kwargs):
    return r1h14_with_mk1(**kwargs)


def r1h14_with_mk2(**kwargs):
    if MARK2_VALON is None:  # If lo_valon is None the roach will use its internal valon, with unfortunate results.
        raise ValueError("MARK2_VALON is None.")
    r1 = heterodyne.RoachHeterodyne(roachip=ROACH1_IP, adc_valon=ROACH1_VALON, host_ip=ROACH1_HOST_IP,
                                    lo_valon=MARK2_VALON, attenuator=attenuator.Attenuator(), **kwargs)
    return r1
    

def r1h14_with_mk1(**kwargs):
    r1 = heterodyne.RoachHeterodyne(roachip=ROACH1_IP, adc_valon=ROACH1_VALON, host_ip=ROACH1_HOST_IP, lo_valon=None,
                                    attenuator=None, **kwargs)
    return r1


def r1h11_with_mk2(**kwargs):
    if MARK2_VALON is None:  # If lo_valon is None the roach will use its internal valon, with unfortunate results.
        raise ValueError("MARK2_VALON is None.")
    return heterodyne.Roach1Heterodyne11(roachip=ROACH1_IP, adc_valon=ROACH1_VALON, host_ip=ROACH1_HOST_IP,
                                         lo_valon=MARK2_VALON, attenuator=attenuator.Attenuator(), **kwargs)


def r1h11_with_mk1(**kwargs):
    return heterodyne.Roach1Heterodyne11(roachip=ROACH1_IP, adc_valon=ROACH1_VALON, host_ip=ROACH1_HOST_IP,
                                         lo_valon=None, attenuator=None, **kwargs)


def r1h11a_with_mk2(**kwargs):
    if MARK2_VALON is None:  # If lo_valon is None the roach will use its internal valon, with unfortunate results.
        raise ValueError("MARK2_VALON is None.")
    return heterodyne.Roach1Heterodyne11Antialiased(roachip=ROACH1_IP, adc_valon=ROACH1_VALON, host_ip=ROACH1_HOST_IP,
                                                    lo_valon=MARK2_VALON, attenuator=attenuator.Attenuator(), **kwargs)


def r1h11a_with_mk1(**kwargs):
    return heterodyne.Roach1Heterodyne11Antialiased(roachip=ROACH1_IP, adc_valon=ROACH1_VALON, host_ip=ROACH1_HOST_IP,
                                                    lo_valon=None, attenuator=None, **kwargs)


def r1h09a_with_mk2(**kwargs):
    if MARK2_VALON is None:  # If lo_valon is None the roach will use its internal valon, with unfortunate results.
        raise ValueError("MARK2_VALON is None.")
    return heterodyne.Roach1Heterodyne09Antialiased(roachip=ROACH1_IP, adc_valon=ROACH1_VALON, host_ip=ROACH1_HOST_IP,
                                                    lo_valon=MARK2_VALON, attenuator=attenuator.Attenuator(), **kwargs)


def r1h09a_with_mk1(**kwargs):
    return heterodyne.Roach1Heterodyne09Antialiased(roachip=ROACH1_IP, adc_valon=ROACH1_VALON, host_ip=ROACH1_HOST_IP,
                                                    lo_valon=None, attenuator=None, **kwargs)


