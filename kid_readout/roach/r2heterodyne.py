"""
Classes to interface to ROACH2 hardware for KID readout systems
"""
__author__ = 'gjones'

import time
import socket

import numpy as np
import scipy.signal

import kid_readout.roach.r2_udp_catcher
import tools
from interface import RoachInterface
from heterodyne import RoachHeterodyne


try:
    import numexpr

    have_numexpr = True
except ImportError:
    have_numexpr = False



class Roach2Heterodyne(RoachHeterodyne):
    def __init__(self,roach=None, wafer=0, roachip='r2kid', adc_valon=None, host_ip='10.0.0.1', initialize=True,
                 nfs_root='/srv/roach_boot/etch', lo_valon=None, iq_delay=0, attenuator=None):
        super(Roach2Heterodyne,self).__init__(roach=roach,wafer=wafer,roachip=roachip, adc_valon=adc_valon,
                                            host_ip=host_ip, nfs_root=nfs_root, lo_valon=lo_valon, attenuator=attenuator)

        self.lo_frequency = 0.0
        self.heterodyne = True
        self.boffile = 'r2iq2xpfb14mcr12gb_2016_Feb_20_0948.bof'

        self.wafer = wafer
        self.raw_adc_ns = 2 ** 12  # number of samples in the raw ADC buffer
        self.nfft = 2 ** 14
        self._fpga_output_buffer = 'ppout%d' % wafer
        self.iq_delay = iq_delay

        self._general_setup()

        if initialize:
            self.initialize()

    def initialize(self, fs=512.0, cal_qdr=True, use_config=True):
        super(Roach2Heterodyne,self).initialize(fs=fs,start_udp=False,use_config=use_config)
        self.r.write_int('destip',np.fromstring(socket.inet_aton(self.host_ip),dtype='>u4')[0])
        self.r.write_int('destport',55555)
        self.r.write_int('txrst',3)
        self.r.write_int('txrst',2)
        self.r.tap_start('gbe','one_GbE',0x021111123456,0x0A000002,12345)

        if cal_qdr:
            import qdr
            q = qdr.Qdr(self.r,'qdr0')
            q.qdr_cal(verbosity=1)

    def set_tone_bins(self, bins, nsamp, amps=None, load=True, normfact=None,phases=None):
        super(Roach2Heterodyne,self).set_tone_bins(bins=bins, nsamp=nsamp, amps=amps, load=load, normfact=normfact, phases=phases)

    def load_waveforms(self, i_wave, q_wave, fast=True, start_offset=0):
        """
        Load waveforms for the two DACs

        i_wave,q_wave : arrays of 16-bit (dtype='i2') integers with waveforms for the two DACs

        fast : boolean
            decide what method for loading the dram
        """
        #somehow the r2 qdr has the dac0/1 outputs switched...
        data = np.zeros((2 * i_wave.shape[0],), dtype='>i2')
        data[0::4] = q_wave[::2]
        data[1::4] = q_wave[1::2]
        data[2::4] = i_wave[::2]
        data[3::4] = i_wave[1::2]

        self.r.blindwrite('qdr0_memory', data.tostring())
        self._unpause_dram()

    def _pause_dram(self):
        self.r.write_int('qdr_en',0)
    def _unpause_dram(self):
        self.r.write_int('qdr_en',1)


    def get_raw_adc(self):
        """
        Grab raw ADC samples
        returns: s0,s1
        s0 and s1 are the samples from adc 0 and adc 1 respectively
        Each sample is a 12 bit signed integer (cast to a numpy float)
        """
        self.r.write_int('adc_snap_ctrl', 0)
        self.r.write_int('adc_snap_ctrl', 5)
        s0 = (np.fromstring(self.r.read('adc_snap_bram', self.raw_adc_ns * 2 * 2), dtype='>i2'))
        sb = s0.view('>i4')
        i = sb[::2].copy().view('>i2') / 16.
        q = sb[1::2].copy().view('>i2') / 16.
        return i, q

    def get_data_udp(self, nread=2, demod=True):
        data, seqnos = kid_readout.roach.r2_udp_catcher.get_udp_data(self, npkts=nread,
                                                                     nchans=self.fft_bins.shape[1],
                                                                     addr=(self.host_ip, 55555))  # , stream_reg, addr)
        if demod:
            data = self.demodulate_data(data)
        return data, seqnos
