"""
Classes to interface to ROACH2 hardware for KID readout systems
"""
__author__ = 'gjones'

import time
import socket

import numpy as np
import scipy.signal

import udp_catcher
import tools
from interface import RoachInterface
from baseband import RoachBaseband


try:
    import numexpr

    have_numexpr = True
except ImportError:
    have_numexpr = False



class Roach2Baseband(RoachBaseband):
    def __init__(self,roach=None, wafer=0, roachip='r2kid', adc_valon=None, host_ip=None, initialize=True,
                 nfs_root='/srv/roach_boot/etch'):
        super(Roach2Baseband,self).__init__(roach=roach,wafer=wafer,roachip=roachip, adc_valon=adc_valon,
                                            host_ip=host_ip, initialize=initialize, nfs_root=nfs_root)

        self.lo_frequency = 0.0
        self.heterodyne = False
        self.boffile = 'r2bb2xpfb14mcr21_2015_Oct_07_1708.bof'
        self.boffile = 'r2bb2xpfb14mcr21_2015_Oct_08_1422.bof'  # This boffile was giving ADC glitches 2015-10-19
        self.boffile = 'r2bb2xpfb14mcr21_2015_Oct_08_2247.bof'

        self.wafer = wafer
        self.raw_adc_ns = 2 ** 12  # number of samples in the raw ADC buffer
        self.nfft = 2 ** 14
        self._fpga_output_buffer = 'ppout%d' % wafer

        self._general_setup()

        if initialize:
            self.initialize()

    def initialize(self, fs=512.0, cal_qdr=True, use_config=True):
        super(Roach2Baseband,self).initialize(fs=fs,start_udp=False,use_config=use_config)
        if cal_qdr:
            import qdr
            q = qdr.Qdr(self.r,'qdr0')
            q.qdr_cal(verbosity=1)

    def load_waveform(self, wave, start_offset=0, fast=True):
        """
        Load waveform

        wave : array of 16-bit (dtype='i2') integers with waveform

        fast : boolean
            decide what method for loading the dram
        """
        data = np.zeros((2 * wave.shape[0],), dtype='>i2')
        offset = (1-self.wafer) * 2
        data[offset::4] = wave[::2]
        data[offset + 1::4] = wave[1::2]
        #start_offset = start_offset * data.shape[0]
        # self.r.write_int('dram_mask', data.shape[0]/4 - 1)
        self.r.blindwrite('qdr0_memory', data.tostring())
        self._unpause_dram()

    def _pause_dram(self):
        self.r.write_int('qdr_en',0)
    def _unpause_dram(self):
        self.r.write_int('qdr_en',1)