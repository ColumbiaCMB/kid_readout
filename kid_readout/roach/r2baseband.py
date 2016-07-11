"""
Classes to interface to ROACH2 hardware for KID readout systems
"""
__author__ = 'gjones'

import time
import socket

import numpy as np
import scipy.signal
import logging

import udp_catcher
import tools
from interface import RoachInterface
from baseband import RoachBaseband

logger = logging.getLogger(__name__)

try:
    import numexpr

    have_numexpr = True
except ImportError:
    have_numexpr = False



class Roach2Baseband(RoachBaseband):
    def __init__(self,roach=None, wafer=0, roachip='r2kid', adc_valon=None, host_ip=None, initialize=True,
                 nfs_root='/srv/roach_boot/etch'):
        super(Roach2Baseband,self).__init__(roach=roach,wafer=wafer,roachip=roachip, adc_valon=adc_valon,
                                            host_ip=host_ip, initialize=False, nfs_root=nfs_root)

        self.lo_frequency = 0.0
        self.heterodyne = False
        self.is_roach2 = True
        self.boffile = 'r2bb2xpfb14mcr23_2015_Oct_27_1357.bof'

        self.wafer = wafer
        self.raw_adc_ns = 2 ** 12  # number of samples in the raw ADC buffer
        self.nfft = 2 ** 14
        self._fpga_output_buffer = 'ppout%d' % wafer

        self._general_setup()

        if initialize:
            self.initialize()

    def initialize(self, fs=512.0, use_config=True, force_cal_qdr=False):
        reprogrammed = super(Roach2Baseband,self).initialize(fs=fs,start_udp=False,use_config=use_config)
        logger.debug("Checking QDR calibration")
        import qdr
        q = qdr.Qdr(self.r,'qdr0')
        qdr_is_calibrated = q.qdr_cal_check()
        if qdr_is_calibrated:
            logger.debug("QDR is calibrated")
        if not qdr_is_calibrated or force_cal_qdr or reprogrammed:
            logger.debug("Calibrating QDR")
            q.qdr_cal()
            logger.info("Succesfully recalibrated QDR")

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