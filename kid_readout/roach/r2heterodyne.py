"""
Classes to interface to ROACH2 hardware for KID readout systems
"""
__author__ = 'gjones'

import socket
import logging

import numpy as np
from scipy import signal

import kid_readout.roach.r2_udp_catcher
from heterodyne import RoachHeterodyne
from kid_readout.roach.demodulator import Demodulator

logger = logging.getLogger(__name__)

try:
    import numexpr
    have_numexpr = True
except ImportError:
    have_numexpr = False


class Roach2Heterodyne(RoachHeterodyne):

    MEMORY_SIZE_BYTES = 2 ** 23  # 8 MB

    def __init__(self, roach=None, wafer=0, roachip='r2kid', adc_valon=None, host_ip=None, initialize=True,
                 nfs_root='/srv/roach_boot/etch', lo_valon=None, attenuator=None, use_config=True):
        super(Roach2Heterodyne, self).__init__(roach=roach, wafer=wafer, roachip=roachip, adc_valon=adc_valon,
                                               host_ip=host_ip, nfs_root=nfs_root, lo_valon=lo_valon,
                                               attenuator=attenuator)
        self.lo_frequency = 0.0
        self.heterodyne = True
        self.is_roach2 = True
        self.boffile = 'r2iq2xpfb14mcr18gb_2016_Jun_30_1104.bof'
        self.iq_delay = 0
        self.channel_selection_offset = 3
        self.wafer = wafer
        self.raw_adc_ns = 2 ** 12  # number of samples in the raw ADC buffer
        self.nfft = 2 ** 14
        self.fpga_cycles_per_filterbank_frame = 2**13
        self._fpga_output_buffer = 'ppout%d' % wafer
        self.phase0 = None      #initial sequence number, if none then no data has been read in yet
        self._general_setup()
        if initialize:
            self.initialize(use_config=use_config)

    def initialize(self, fs=512.0, use_config=True, force_cal_qdr=False):
        reprogrammed = super(Roach2Heterodyne,self).initialize(fs=fs,start_udp=False,use_config=use_config)
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
            logger.info("Successfully recalibrated QDR")

    def max_num_waveforms(self, num_tone_samples):
        """The ROACH2 code currently allows for only one waveform."""
        return 1

    def set_tone_bins(self, bins, nsamp, amps=None, load=True, normfact=None, phases=None, preset_norm=True):
        super(Roach2Heterodyne,self).set_tone_bins(bins=bins, nsamp=nsamp, amps=amps, load=load, normfact=normfact, phases=phases, preset_norm=preset_norm)

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
                data = self.demodulate_data(data, seq_nos)
            data = data*self.wavenorm
        return data, seq_nos

    @property
    def blocks_per_second_per_channel(self):
        chan_rate = self.fs * 1e6 / (self.nfft)  # samples per second for one tone_index
        samples_per_channel_per_block = 1024
        return chan_rate / samples_per_channel_per_block


class Roach2Heterodyne11(Roach2Heterodyne):
    """
    Compared to Roach2Heterodyne, this class has more bandwidth (250 kHz) per channel and a sharper channel filter.

    The filter (8-tap Hamming) transmission is down to about 0.55 at the edges of the PFB bin, so using a tone too far
    from the bin center may produce strange results.
    """

    def __init__(self, roach=None, wafer=0, roachip='r2kid', adc_valon=None, host_ip=None, initialize=False,
                 use_config=False, nfs_root='/srv/roach_boot/etch', lo_valon=None, attenuator=None):
        """
        Class to represent the heterodyne readout system (high-frequency with IQ mixers)

        roach: an FpgaClient instance for communicating with the ROACH.
                If not specified, will try to instantiate one connected to *roachip*
        wafer: 0
                Not used for heterodyne system
        roachip: (optional). Network address of the ROACH if you don't want to provide an FpgaClient
        adc_valon: a Valon class, a string, or None
                Provide access to the Valon class which controls the Valon synthesizer which provides
                the ADC and DAC sampling clock.
                The default None value will use the valon.find_valon function to locate a synthesizer
                and create a Valon class for you.
                You can alternatively pass a string such as '/dev/ttyUSB0' to specify the port for the
                synthesizer, which will then be used for creating a Valon class.
                Finally, for test suites, you can directly pass a Valon class or a class with the same
                interface.
        """
        super(Roach2Heterodyne11, self).__init__(roach=roach, roachip=roachip, adc_valon=adc_valon, host_ip=host_ip,
                                                 nfs_root=nfs_root, lo_valon=lo_valon)
        self.lo_frequency = 0.0
        self.heterodyne = True
        self.boffile = 'r2iq2xpfb11mcr19gb_2017_Jan_13_1357.bof'
        self.iq_delay = 0
        self.channel_selection_offset = 3
        self.wafer = wafer
        self.raw_adc_ns = 2 ** 12  # number of samples in the raw ADC buffer
        self.nfft = 2 ** 11
        self.fpga_cycles_per_filterbank_frame = 2 ** 10
        self._fpga_output_buffer = 'ppout%d' % wafer
        self._general_setup()
        self.demodulator = Demodulator(nfft=self.nfft, num_taps=8, window=signal.hamming,
                                       hardware_delay_samples=self.hardware_delay_estimate * self.fs * 1e6)
        self.attenuator = attenuator
        if initialize:
            self.initialize(use_config=use_config)


class Roach2Heterodyne11NarrowChannel(Roach2Heterodyne):
    """
    This class is identical to Roach1Heterodyne11 except that the channel filter frequencies are scaled by 0.8 to
    reduce aliasing.

    The filter (8-tap Hamming) transmission is down to about 0.2 at the edges of the PFB bin, so using a tone too far
    from the bin center may produce strange results.
    """

    def __init__(self, roach=None, wafer=0, roachip='roach', adc_valon=None, host_ip=None, initialize=False,
                 use_config=False, nfs_root='/srv/roach_boot/etch', lo_valon=None, attenuator=None):
        """
        Class to represent the heterodyne readout system (high-frequency with IQ mixers)

        roach: an FpgaClient instance for communicating with the ROACH.
                If not specified, will try to instantiate one connected to *roachip*
        wafer: 0
                Not used for heterodyne system
        roachip: (optional). Network address of the ROACH if you don't want to provide an FpgaClient
        adc_valon: a Valon class, a string, or None
                Provide access to the Valon class which controls the Valon synthesizer which provides
                the ADC and DAC sampling clock.
                The default None value will use the valon.find_valon function to locate a synthesizer
                and create a Valon class for you.
                You can alternatively pass a string such as '/dev/ttyUSB0' to specify the port for the
                synthesizer, which will then be used for creating a Valon class.
                Finally, for test suites, you can directly pass a Valon class or a class with the same
                interface.
        """
        super(Roach2Heterodyne11NarrowChannel, self).__init__(roach=roach, roachip=roachip, adc_valon=adc_valon,
                                                              host_ip=host_ip, nfs_root=nfs_root, lo_valon=lo_valon)

        self.lo_frequency = 0.0
        self.heterodyne = True
        self.boffile = 'r2iq2xpfb11mcr20gb_2017_Sep_26_1236.bof'
        self.iq_delay = 0
        self.channel_selection_offset = 3
        self.wafer = wafer
        self.raw_adc_ns = 2 ** 12  # number of samples in the raw ADC buffer
        self.nfft = 2 ** 11
        self.fpga_cycles_per_filterbank_frame = 2 ** 10
        self._fpga_output_buffer = 'ppout%d' % wafer
        self.window_frequency_scale = 0.8
        self._general_setup()
        self.demodulator = Demodulator(nfft=self.nfft, num_taps=8, window=signal.hamming,
                                       hardware_delay_samples=self.hardware_delay_estimate * self.fs * 1e6,
                                       window_frequency_scale=self.window_frequency_scale)
        self.attenuator = attenuator
        if initialize:
            self.initialize(use_config=use_config)
