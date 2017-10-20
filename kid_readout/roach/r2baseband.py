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
import kid_readout.roach.r2_udp_catcher
from kid_readout import settings

logger = logging.getLogger(__name__)

try:
    import numexpr

    have_numexpr = True
except ImportError:
    have_numexpr = False


class Roach2Baseband(RoachBaseband):

    MEMORY_SIZE_BYTES = 2 ** 23  # 8 MB

    def __init__(self,roach=None, wafer=0, roachip='r2kid', adc_valon=settings.ROACH2_VALON, host_ip=settings.ROACH2_GBE_HOST_IP,
                 initialize=True, nfs_root='/srv/roach_boot/etch'):
        super(Roach2Baseband,self).__init__(roach=roach,wafer=wafer,roachip=roachip, adc_valon=adc_valon,
                                            host_ip=host_ip, initialize=False, nfs_root=nfs_root)

        self.lo_frequency = 0.0
        self.heterodyne = False
        self.is_roach2 = True
#        self.boffile = 'r2bb2xpfb14mcr23_2015_Oct_27_1357.bof'
        self.boffile = 'r2bb2xpfb14mcr25_2016_Oct_01_2233.bof'
        self.channel_selection_offset=3

        self.wafer = wafer
        self.raw_adc_ns = 2 ** 12  # number of samples in the raw ADC buffer
        self.nfft = 2 ** 14
        self.fpga_cycles_per_filterbank_frame = 2**14
        self._fpga_output_buffer = 'ppout%d' % wafer

        self._general_setup()

        if initialize:
            self.initialize()

    def initialize(self, fs=512.0, use_config=True, force_cal_qdr=False):
        reprogrammed = super(Roach2Baseband,self).initialize(fs=fs,start_udp=False,use_config=use_config)
        self.r.write_int('destip',np.fromstring(socket.inet_aton(self.host_ip),dtype='>u4')[0])
        self.r.write_int('destport',55555)
        self.r.write_int('txrst',3)
        self.r.write_int('txrst',2)
        try:
            self.r.tap_stop('gbe')
            logger.debug("Stopped tap interface")
        except RuntimeError:
            pass
        self.r.tap_start('gbe','one_GbE',0x021111123456,0x0A000002,12345)
        logger.debug("Started tap interface")

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

    def get_data(self, nread=2, demod=True):
        # TODO This is a temporary hack until we get the system simulation code in place
        if self._using_mock_roach:
            data = (np.random.standard_normal((nread * 4096, self.num_tones)) +
                    1j * np.random.standard_normal((nread * 4096, self.num_tones)))
            if self.r.sleep_for_fake_data:
                time.sleep(nread / self.blocks_per_second)
            seqnos = np.arange(data.shape[0])
            return data, seqnos
        else:
            return self.get_data_udp(nread=nread, demod=demod)

    def get_data_udp(self, nread=2, demod=True, fast=False):
        data, seq_nos = kid_readout.roach.r2_udp_catcher.get_udp_data(self, npkts=nread,
                                                                     nchans=self.readout_selection.shape[0],
                                                                     addr=(self.host_ip, 55555), fast=fast)

        if self.phase0 is None:
            self.phase0 = seq_nos[0]
        if demod:
            seq_nos -= self.phase0
            if fast:
                data = self.demodulate_stream(data, seq_nos)
            else:
                data = self.demodulate_data(data)
        return data, seq_nos

    def select_fft_bins(self, readout_selection=None, sync=True):
        """
        Select which subset of the available FFT bins to read out

        Initially we can only read out from a subset of the FFT bins, so this function selects which bins to read out right now
        This also takes care of writing the selection to the FPGA with the appropriate tweaks

        The readout selection is stored to self.readout_selection
        The FPGA readout indexes is stored in self.fpga_fft_readout_indexes
        The bins that we are reading out is stored in self.readout_fft_bins

        readout_selection : array of ints
            indexes into the self.fft_bins array to specify the bins to read out
        """
        if readout_selection is None:
            readout_selection = range(self.fft_bins.shape[1])
        idxs = self.fft_bin_to_index(self.fft_bins[self.bank,readout_selection])
        order = idxs.argsort()
        idxs = idxs[order]
        if np.any(np.diff(idxs)==0):
            failures = np.flatnonzero(np.diff(idxs)==0)
            raise ValueError("Selected filterbank channels overlap.\n"
                             "The requested channel indexes are: %s\n"
                             "and the failing channel indexes are: %s" %
                             (str(idxs), '; '.join([('%d,%d'% (x,x+1)) for x in idxs[failures]])))
        self.readout_selection = np.array(readout_selection)[order]
        self.fpga_fft_readout_indexes = idxs
        self.readout_fft_bins = self.fft_bins[self.bank, self.readout_selection]

        binsel = np.zeros((self.nfft,), dtype='u1')
        binsel_index = np.mod(self.fpga_fft_readout_indexes-self.channel_selection_offset,self.nfft)
        binsel[binsel_index] = 1
        self.r.write('chans', binsel.tostring())
        if sync:
            self._sync()

    def demodulate_data(self, data):
        """
        Demodulate the data from the FFT bin

        This function assumes that self.select_fft_bins was called to set up the necessary class attributes

        data : array of complex data

        returns : demodulated data in an array of the same shape and dtype as *data*
        """
        bank = self.bank
        hardware_delay = self.hardware_delay_estimate*1e6
        demod = np.zeros_like(data)
        t = np.arange(data.shape[0])
        for n, ich in enumerate(self.readout_selection):
            phi0 = self.phases[ich]
            k = self.tone_bins[bank, ich]
            m = self.fft_bins[bank, ich]
            if m >= self.nfft // 2:
                sign = -1.0
            else:
                sign = 1.0
            nfft = self.nfft
            ns = self.tone_nsamp
            f_tone = k * self.fs / float(ns)
            foffs = (2 * k * nfft - m * ns) / float(ns)
            wc = self._window_response(foffs / 2.0) * (self.tone_nsamp / 2.0 ** 18)
            #print "chan",m,"tone",k,"sign",sign,"foffs",foffs
            demod[:, n] = (wc * np.exp(sign * 1j * (2 * np.pi * foffs * t + phi0) - sign *
                                       2j*np.pi*f_tone*hardware_delay)
                           * data[:, n])
            if m >= self.nfft // 2:
                demod[:, n] = np.conjugate(demod[:, n])
        return self.wavenorm*np.conjugate(demod)

    @property
    def blocks_per_second_per_channel(self):
        chan_rate = self.fs * 1e6 / (2 * self.nfft)  # samples per second for one tone_index
        samples_per_channel_per_block = 1024 #1024 samples per packet
        return chan_rate / samples_per_channel_per_block
