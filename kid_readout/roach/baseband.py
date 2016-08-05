"""
Classes to interface to ROACH hardware for KID readout systems
"""

import time
import socket

import numpy as np
import scipy.signal

import udp_catcher
import tools
from interface import RoachInterface
from kid_readout.settings import ROACH1_VALON, ROACH1_IP, ROACH1_HOST_IP

import logging
logger = logging.getLogger(__name__)

try:
    import numexpr

    have_numexpr = True
except ImportError:
    have_numexpr = False



class RoachBaseband(RoachInterface):

    def __init__(self, roach=None, wafer=0, roachip=ROACH1_IP, adc_valon=ROACH1_VALON, host_ip=ROACH1_HOST_IP,
                 initialize=True, nfs_root='/srv/roach_boot/etch'):
        """
        Class to represent the baseband readout system (low-frequency (150 MHz), no mixers)

        roach: an FpgaClient instance for communicating with the ROACH.
                If not specified, will try to instantiate one connected to *roachip*
        wafer: 0 or 1.
                In baseband mode, each of the two DAC and ADC connections can be used independantly to
                readout a single wafer each. This parameter indicates which connection you want to use.
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
        host_ip: Override IP address to which the ROACH should send it's data. If left as None,
                the host_ip will be set appropriately based on the HOSTNAME.
        initialize: Default True, will call self.initialize() which will try to load state from saved config
                Set to False if you don't want this to happen.
        """
        super(RoachBaseband,self).__init__(roach=roach, roachip=roachip, adc_valon=adc_valon, host_ip=host_ip,
                 nfs_root=nfs_root)

        self.lo_frequency = 0.0
        self.heterodyne = False
        self.boffile = 'bb2xpfb14mcr17b_2015_Apr_21_1159.bof'

        self.wafer = wafer
        self.raw_adc_ns = 2 ** 12  # number of samples in the raw ADC buffer
        self.nfft = 2 ** 14
        self._fpga_output_buffer = 'ppout%d' % wafer

        self._general_setup()
        if initialize:
            self.initialize()

    def load_waveform(self, wave, start_offset=0, fast=True):
        """
        Load waveform
        
        wave : array of 16-bit (dtype='i2') integers with waveform
        
        fast : boolean
            decide what method for loading the dram 
        """
        data = np.zeros((2 * wave.shape[0],), dtype='>i2')
        offset = self.wafer * 2
        data[offset::4] = wave[::2]
        data[offset + 1::4] = wave[1::2]
        start_offset = start_offset * data.shape[0]
        # self.r.write_int('dram_mask', data.shape[0]/4 - 1)
        self._load_dram(data, start_offset=start_offset, fast=fast)

    def set_tone_freqs(self, freqs, nsamp, amps=None, load=True, normfact=None, readout_selection=None,
                       phases=None, preset_norm=True):
        """
        Set the stimulus tones to generate
        
        freqs : array of frequencies in MHz
            For baseband system, these must be positive
        nsamp : int, must be power of 2
            number of samples in the playback buffer. Frequency resolution will be fs/nsamp
        amps : optional array of floats, same length as freqs array
            specify the relative amplitude of each tone. Can set to zero to read out a portion
            of the spectrum with no stimulus tone.
        load : bool (debug only). 
            If false, don't actually load the waveform, just calculate it.
                    
        returns:
        actual_freqs : array of the actual frequencies after quantization based on nsamp
        """
        freqs = np.atleast_1d(freqs).astype('float')
        bins = np.round((freqs / self.fs) * nsamp).astype('int')
        actual_freqs = self.fs * bins / float(nsamp)
        self.set_tone_bins(bins, nsamp, amps=amps, load=load, normfact=normfact,phases=phases, preset_norm=preset_norm)
        self.fft_bins = self.calc_fft_bins(bins, nsamp)
        self.select_bank(0)
        self.select_fft_bins(readout_selection)
        self.save_state()
        return actual_freqs

    set_tone_baseband_freqs = set_tone_freqs

    def add_tone_freqs(self, freqs, amps=None, overwrite_last=False, preset_norm=True):
        freqs = np.atleast_1d(freqs).astype('float')
        if freqs.shape[0] != self.tone_bins.shape[1]:
            raise ValueError("freqs array must contain same number of tones as original waveforms")
        if overwrite_last:  # Delete the last waveform and readout selection entry.
            self.tone_bins = self.tone_bins[:-1, :]
            self.fft_bins = self.fft_bins[:-1, :]
        nsamp = self.tone_nsamp
        bins = np.round((freqs / self.fs) * nsamp).astype('int')
        actual_freqs = self.fs * bins / float(nsamp)
        self.add_tone_bins(bins, amps=amps, preset_norm=preset_norm)
        self.fft_bins = np.vstack((self.fft_bins, self.calc_fft_bins(bins, nsamp)))
        self.save_state()
        return actual_freqs

    def set_tone_bins(self, bins, nsamp, amps=None, load=True, normfact=None, phases=None, preset_norm=True):
        """
        Set the stimulus tones by specific integer bins
        
        bins : array of bins at which tones should be placed
            For Heterodyne system, negative frequencies should be placed in canonical FFT order
            If 2d, interpret as (nwaves,ntones)
        nsamp : int, must be power of 2
            number of samples in the playback buffer. Frequency resolution will be fs/nsamp
        amps : optional array of floats, same length as bins array
            specify the relative amplitude of each tone. Can set to zero to read out a portion
            of the spectrum with no stimulus tone.
        load : bool (debug only). If false, don't actually load the waveform, just calculate it.
        """

        if bins.ndim == 1:
            bins.shape = (1, bins.shape[0])
        nwaves = bins.shape[0]
        spec = np.zeros((nwaves, nsamp // 2 + 1), dtype='complex')
        self.tone_bins = bins.copy()
        self.tone_nsamp = nsamp
        if phases is None:
            phases = np.random.random(bins.shape[1]) * 2 * np.pi
        self.phases = phases.copy()
        if amps is None:
            amps = 1.0
        self.amps = amps
        for k in range(nwaves):
            spec[k, bins[k, :]] = amps * np.exp(1j * phases)
        wave = np.fft.irfft(spec, axis=1)
        if preset_norm and not normfact:
            self.wavenorm = tools.calc_wavenorm(bins.shape[1], nsamp, baseband=True)
        else:
            self.wavenorm = np.abs(wave).max()
            if normfact is not None:
                wn = (2.0 / normfact) * len(bins) / float(nsamp)
                logger.debug("Using user provide waveform normalization resulting in wavenorm %f versus optimal %f. "
                             "Ratio is %f" % (wn,self.wavenorm,self.wavenorm/wn))
                self.wavenorm = wn
        qwave = np.round((wave / self.wavenorm) * (2 ** 15 - 1024)).astype('>i2')
        qwave.shape = (qwave.shape[0] * qwave.shape[1],)
        self.qwave = qwave
        if load:
            self.load_waveform(qwave)
        self.save_state()

    def add_tone_bins(self, bins, amps=None, preset_norm=True):
        nsamp = self.tone_nsamp
        spec = np.zeros((nsamp // 2 + 1,), dtype='complex')
        self.tone_bins = np.vstack((self.tone_bins, bins))
        phases = self.phases
        if amps is None:
            amps = 1.0
        # self.amps = amps  # TODO: Need to figure out how to deal with this

        spec[bins] = amps * np.exp(1j * phases)
        wave = np.fft.irfft(spec)
        if preset_norm:
            self.wavenorm = tools.calc_wavenorm(self.tone_bins.shape[1], nsamp, baseband=True)
        else:
            self.wavenorm = np.abs(wave).max()
        qwave = np.round((wave / self.wavenorm) * (2 ** 15 - 1024)).astype('>i2')
        # self.qwave = qwave  # TODO: Deal with this, if we ever use it
        start_offset = self.tone_bins.shape[0] - 1
        self.load_waveform(qwave, start_offset=start_offset)
        self.save_state()

    def calc_fft_bins(self, tone_bins, nsamp):
        """
        Calculate the FFT bins in which the tones will fall
        
        tone_bins : array of integers
            the tone bins (0 to nsamp - 1) which contain tones
        
        nsamp : length of the playback buffer
        
        returns : fft_bins, array of integers. 
        """
        tone_bins_per_fft_bin = nsamp / (2. * self.nfft)  # factor of 2 because real signal
        fft_bins = np.round(tone_bins / float(tone_bins_per_fft_bin)).astype('int')
        return fft_bins

    def fft_bin_to_index(self, bins):
        """
        Convert FFT bins to FPGA indexes
        """
        top_half = bins > self.nfft // 2
        idx = bins.copy()
        idx[top_half] = self.nfft - bins[top_half] + self.nfft // 2
        return idx

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
            readout_selection = np.arange(self.fft_bins.shape[1])
        bank = self.bank
        offset = 2
        idxs = self.fft_bin_to_index(self.fft_bins[bank, readout_selection])
        order = idxs.argsort()
        idxs = idxs[order]
        self.readout_selection = np.array(readout_selection)[order]
        self.fpga_fft_readout_indexes = idxs
        self.readout_fft_bins = self.fft_bins[bank, self.readout_selection]

        binsel = np.zeros((self.fpga_fft_readout_indexes.shape[0] + 1,), dtype='>i4')
        binsel[:-1] = np.mod(self.fpga_fft_readout_indexes - offset, self.nfft)
        binsel[-1] = -1
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
                sign = 1.0
            else:
                sign = -1.0
            nfft = self.nfft
            ns = self.tone_nsamp
            f_tone = k * self.fs / float(ns)
            foffs = (2 * k * nfft - m * ns) / float(ns)
            wc = self._window_response(foffs / 2.0) * (self.tone_nsamp / 2.0 ** 18)
            demod[:, n] = (wc * np.exp(sign * 1j * (2 * np.pi * foffs * t + phi0) - sign *
                                       2j*np.pi*f_tone*hardware_delay)
                           * data[:, n])
            if m >= self.nfft // 2:
                demod[:, n] = np.conjugate(demod[:, n])
        return demod

    def set_loopback(self,enable):
        if enable:
            self.r.write_int('loopback',1)
            self.loopback = True
        else:
            self.r.write_int('loopback',0)
            self.loopback = False


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

    @property
    def blocks_per_second_per_channel(self):
        chan_rate = self.fs * 1e6 / (2 * self.nfft)  # samples per second for one tone_index
        samples_per_channel_per_block = 4096
        return chan_rate / samples_per_channel_per_block


    def get_data_seconds(self, nseconds, demod=True, pow2=True):
        """
        Capture data for specified length of time (using the udp interface)

        nseconds: Number of seconds

        demod: bool, Should the data be demodulated (default True)

        pow2: bool, If true, force the data length to the nearest power of 2
        """
        chan_rate = self.fs * 1e6 / (2 * self.nfft)  # samples per second for one tone_index
        samples_per_channel_per_block = 4096
        seconds_per_block = samples_per_channel_per_block / chan_rate
        blocks = int(np.round(nseconds / seconds_per_block))
        if pow2:
            lg2 = np.round(np.log2(blocks))
            if lg2 < 0:
                lg2 = 0
            blocks = 2 ** lg2

        return self.get_data_udp(blocks, demod=demod)

    def get_data_udp(self, nread=2, demod=True):
        chan_offset = 1
        nch = self.fpga_fft_readout_indexes.shape[0]
        data, seqnos = udp_catcher.get_udp_data(self, npkts=nread * 16, streamid=np.random.randint(1,2**15),
                                                chans=self.fpga_fft_readout_indexes + chan_offset,
                                                nfft=self.nfft, addr=(self.host_ip, 12345))  # , stream_reg, addr)
        if demod:
            data = self.demodulate_data(data)
        return data, seqnos


    def get_data_seconds_katcp(self, nseconds, demod=True, pow2=True):
        """
        Capture data for specified length of time using the katcp interface
        
        nseconds: Number of seconds
        
        demod: bool, Should the data be demodulated (default True)
        
        pow2: bool, If true, force the data length to the nearest power of 2
        """
        chan_rate = self.fs * 1e6 / (2 * self.nfft)  # samples per second per tone_index
        nch = self.fpga_fft_readout_indexes.shape[0]
        seconds_per_block = (1024 * nch) / chan_rate
        blocks = int(np.round(nseconds / seconds_per_block))
        if pow2:
            lg2 = np.round(np.log2(blocks))
            if lg2 < 0:
                lg2 = 0
            blocks = 2 ** lg2
        return self.get_data_katcp(blocks, demod=demod)

    def get_data_katcp(self, nread=10, demod=True):
        """
        Get a chunk of data
        
        nread: number of 4096 sample frames to read
        
        demod: should the data be demodulated before returning? Default, yes
        
        returns  dout,addrs

        dout: complex data stream. Real and imaginary parts are each 16 bit signed
            integers (but cast to numpy complex)

        addrs: counter values when each frame was read. Can be used to check that
            frames are contiguous
        """
        self.r.write_int('streamid',0)  # The code below assumes streamid=0. If we want to use other streamids later,
                                        #  will need to update the code below to mask off the streamid info
        bufname = 'ppout%d' % self.wafer
        chan_offset = 1
        draw, addr, ch = self._read_data(nread, bufname)
        if not np.all(ch == ch[0]):
            logger.error("all channel registers not the same; this case not yet supported.")
            return draw, addr, ch
        if not np.all(np.diff(addr) < 8192):
            logger.warning("address skip!")
        nch = self.readout_selection.shape[0]
        dout = draw.reshape((-1, nch))
        shift = np.flatnonzero(self.fpga_fft_readout_indexes == (ch[0] - chan_offset))[0] - (nch - 1)
        dout = np.roll(dout, shift, axis=1)
        if demod:
            dout = self.demodulate_data(dout)
        return dout, addr

    def _window_response(self, fr):
        res = np.interp(np.abs(fr) * 2 ** 7, np.arange(2 ** 7), self._window_mag)
        res = 1 / res
        return res
