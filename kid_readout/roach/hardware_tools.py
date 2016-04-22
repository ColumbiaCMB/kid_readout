import numpy as np

from interface import RoachInterface
from heterodyne import RoachHeterodyne
from r2heterodyne import Roach2Heterodyne
from attenuator import Attenuator

mark2_valon = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A101FK1H-if00-port0"
roach2_valon = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A101FK1K-if00-port0"


def r2_with_mk2(lo_freq=1000, atten=None):
    attenuator = Attenuator(attenuation=atten)
    r2 = Roach2Heterodyne(roachip='r2kid', adc_valon=roach2_valon, lo_valon=mark2_valon, attenuator=attenuator)
    r2.set_lo(lo_freq)
    return r2
        
   
def r1_with_mk2(lo_freq=1000, atten=None):
    attenuator = Attenuator(attenuation=atten)
    r1 = RoachHeterodyne(roachip='roach', adc_valon='/dev/ttyUSB0', lo_valon=mark2_valon, attenuator=attenuator)
    r1.initialize()
    r1.set_lo(lo_freq)
    return r1
    

def r1_with_mk1(lo_freq=1000):
    r1 = RoachHeterodyne(roachip='roach', adc_valon='/dev/ttyUSB0', lo_valon=None, attenuator=None)
    r1.initialize()
    r1.set_lo(lo_freq)
    return r1
