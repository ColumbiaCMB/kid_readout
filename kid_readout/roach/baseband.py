"""
Classes to interface to ROACH hardware for KID readout systems
"""
# Long-term interface changes:
# TODO: switch to properties where possible
# TODO: give more explicit access to the tone banks
# TODO: raise RoachError for Roach-specific violations

# TODO: add explicit integer division and use from __future__ import division
import time
import socket

import numpy as np
import scipy.signal

import udp_catcher
import tools
from interface import RoachInterface


try:
    import numexpr

    have_numexpr = True
except ImportError:
    have_numexpr = False



class RoachBaseband(RoachInterface):

    def __init__(self, roach=None, wafer=0, roachip='roach', adc_valon=None, host_ip=None, initialize=True,
                 nfs_root='/srv/roach_boot/etch'):
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
                the host_ip will be set appropriately based on the hostname.
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
        bins = np.round((freqs / self.fs) * nsamp).astype('int')
        actual_freqs = self.fs * bins / float(nsamp)
        self.set_tone_bins(bins, nsamp, amps=amps, load=load, normfact=normfact,phases=phases, preset_norm=preset_norm)
        self.fft_bins = self.calc_fft_bins(bins, nsamp)
        self.select_bank(0)
        if readout_selection is not None:
            self.select_fft_bins(readout_selection)
        self.save_state()
        return actual_freqs

    set_tone_baseband_freqs = set_tone_freqs

    def add_tone_freqs(self, freqs, amps=None, overwrite_last=False, preset_norm=True):
        if freqs.shape[0] != self.tone_bins.shape[1]:
            raise ValueError("freqs array must contain same number of tones as original waveforms")
        # This is a hack that doesn't handle bank selection at all and may have additional problems.
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
            For Heterodyne system, negative frequencies should be placed in cannonical FFT order
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
        spec = np.zeros((nwaves, nsamp / 2 + 1), dtype='complex')
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
        if preset_norm:
            self.wavenorm = tools.calc_wavenorm(bins.shape[1], nsamp, baseband=True)
        else:
            self.wavenorm = np.abs(wave).max()
        if normfact is not None:
            wn = (2.0 / normfact) * len(bins) / float(nsamp)
            print "ratio of current wavenorm to optimal:", self.wavenorm / wn
            self.wavenorm = wn
        qwave = np.round((wave / self.wavenorm) * (2 ** 15 - 1024)).astype('>i2')
        qwave.shape = (qwave.shape[0] * qwave.shape[1],)
        self.qwave = qwave
        if load:
            self.load_waveform(qwave)
        self.save_state()

    def add_tone_bins(self, bins, amps=None, preset_norm=True):
        nsamp = self.tone_nsamp
        spec = np.zeros((nsamp / 2 + 1,), dtype='complex')
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
        top_half = bins > self.nfft / 2
        idx = bins.copy()
        idx[top_half] = self.nfft - bins[top_half] + self.nfft / 2
        return idx

    def select_fft_bins(self, readout_selection, sync=True):
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
            if m >= self.nfft / 2:
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
            if m >= self.nfft / 2:
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
    def blocks_per_second(self):
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
        bufname = 'ppout%d' % self.wafer
        chan_offset = 1
        draw, addr, ch = self._read_data(nread, bufname)
        if not np.all(ch == ch[0]):
            print "all channel registers not the same; this case not yet supported"
            return draw, addr, ch
        if not np.all(np.diff(addr) < 8192):
            print "address skip!"
        nch = self.readout_selection.shape[0]
        dout = draw.reshape((-1, nch))
        shift = np.flatnonzero(self.fpga_fft_readout_indexes == (ch[0] - chan_offset))[0] - (nch - 1)
        dout = np.roll(dout, shift, axis=1)
        if demod:
            dout = self.demodulate_data(dout)
        return dout, addr

    def _set_fs(self, fs, chan_spacing=2.0):
        """
        Set sampling frequency in MHz
        Note, this should generally not be called without also reprogramming the ROACH
        Use initialize() instead        
        """
        if self.adc_valon is None:
            print "Could not set Valon; none available"
            return
        self.adc_valon.set_frequency_a(fs, chan_spacing=chan_spacing)  # for now the baseband readout uses both valon
        #  outputs,
        self.fs = float(fs)


class RoachBasebandWide(RoachBaseband):

    def __init__(self, roach=None, wafer=0, roachip='roach', adc_valon=None, host_ip=None):
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
        """
        if roach:
            self.r = roach
        else:
            from corr.katcp_wrapper import FpgaClient
            self.r = FpgaClient(roachip)
            t1 = time.time()
            timeout = 10
            while not self.r.is_connected():
                if (time.time() - t1) > timeout:
                    raise Exception("Connection timeout to roach")
                time.sleep(0.1)

        if adc_valon is None:
            import valon
            ports = valon.find_valons()
            if len(ports) == 0:
                self.adc_valon_port = None
                self.adc_valon = None
                print "Warning: No valon found!"
            else:
                for port in ports:
                    try:
                        self.adc_valon_port = port
                        self.adc_valon = valon.Synthesizer(port)
                        f = self.adc_valon.get_frequency_a()
                        break
                    except:
                        pass
        elif type(adc_valon) is str:
            import valon
            self.adc_valon_port = adc_valon
            self.adc_valon = valon.Synthesizer(self.adc_valon_port)
        else:
            self.adc_valon = adc_valon

        if host_ip is None:
            hostname = socket.gethostname()
            if hostname == 'detectors':
                host_ip = '192.168.4.2'
            else:
                host_ip = '192.168.1.1'

        self.host_ip = host_ip
        self.adc_atten = 31.5
        self.dac_atten = -1
        self.fft_gain = 0
        self.fft_bins = None
        self.tone_nsamp = None
        self.tone_bins = None
        self.phases = None
        self.bof_pid = None
        self.roachip = roachip
        try:
            self.fs = self.adc_valon.get_frequency_a()
        except:
            print "warning couldn't get valon frequency, assuming 512 MHz"
            self.fs = 512.0
        self.wafer = wafer
        self.dac_ns = 2 ** 16  # number of samples in the dac buffer
        self.raw_adc_ns = 2 ** 12  # number of samples in the raw ADC buffer
        self.nfft = 2 ** 11
        # self.boffile = 'bb2xpfb12mcr5_2013_Oct_29_1658.bof'
        #self.boffile = 'bb2xpfb11mcr7_2013_Nov_04_1309.bof'
        #self.boffile = 'bb2xpfb11mcr8_2013_Nov_04_2151.bof'
        self.boffile = 'bb2xpfb11mcr11_2014_Feb_01_1106.bof'
        #self.boffile = 'bb2xpfb11mcr12_2014_Feb_26_1028.bof'
        self._fpga_output_buffer = 'ppout%d' % wafer
        self._window_mag = tools.compute_window(npfb=2 * self.nfft, taps=2, wfunc=scipy.signal.flattop)

    def demodulate_data(self, data):
        """
        Demodulate the data from the FFT bin
        
        This function assumes that self.select_fft_bins was called to set up the necessary class attributes
        
        data : array of complex data
        
        returns : demodulated data in an array of the same shape and dtype as *data*
        """
        bank = self.bank
        demod = np.zeros_like(data)
        t = np.arange(data.shape[0])
        for n, ich in enumerate(self.readout_selection):
            phi0 = self.phases[ich]
            k = self.tone_bins[bank, ich]
            m = self.fft_bins[bank, ich]
            if m >= self.nfft / 2:
                sign = 1.0
            else:
                sign = -1.0
            nfft = self.nfft
            ns = self.tone_nsamp
            foffs = (2 * k * nfft - m * ns) / float(ns)
            wc = self._window_response(foffs / 2) * (self.tone_nsamp / 2.0 ** 18)
            if have_numexpr:
                pi = np.pi
                this_data = data[:, n]
                demod[:, n] = numexpr.evaluate('wc*exp(sign*1j*(2*pi*foffs*t + phi0)) * this_data')
                if m >= self.nfft / 2:
                    np.conj(demod[:, n], out=demod[:, n])
            else:
                demod[:, n] = wc * np.exp(sign * 1j * (2 * np.pi * foffs * t + phi0)) * data[:, n]
                if m >= self.nfft / 2:
                    demod[:, n] = np.conjugate(demod[:, n])
        return demod


class RoachBasebandWide10(RoachBasebandWide):

    def __init__(self, roach=None, wafer=0, roachip='roach', adc_valon=None):
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
        """
        if roach:
            self.r = roach
        else:
            from corr.katcp_wrapper import FpgaClient
            self.r = FpgaClient(roachip)
            t1 = time.time()
            timeout = 10
            while not self.r.is_connected():
                if (time.time() - t1) > timeout:
                    raise Exception("Connection timeout to roach")
                time.sleep(0.1)

        if adc_valon is None:
            import valon
            ports = valon.find_valons()
            if len(ports) == 0:
                self.adc_valon_port = None
                self.adc_valon = None
                print "Warning: No valon found!"
            else:
                for port in ports:
                    try:
                        self.adc_valon_port = port
                        self.adc_valon = valon.Synthesizer(port)
                        f = self.adc_valon.get_frequency_a()
                        break
                    except:
                        pass
        elif type(adc_valon) is str:
            import valon
            self.adc_valon_port = adc_valon
            self.adc_valon = valon.Synthesizer(self.adc_valon_port)
        else:
            self.adc_valon = adc_valon

        self.adc_atten = -1
        self.dac_atten = -1
        self.bof_pid = None
        self.roachip = roachip
        try:
            self.fs = self.adc_valon.get_frequency_a()
        except:
            print "warning couldn't get valon frequency, assuming 512 MHz"
            self.fs = 512.0
        self.wafer = wafer
        self.dac_ns = 2 ** 16  # number of samples in the dac buffer
        self.raw_adc_ns = 2 ** 12  # number of samples in the raw ADC buffer
        self.nfft = 2 ** 10
        # self.boffile = 'bb2xpfb10mcr8_2013_Nov_18_0706.bof'
        self.boffile = 'bb2xpfb10mcr11_2014_Jan_20_1049.bof'
        self._fpga_output_buffer = 'ppout%d' % wafer
        self._window_mag = tools.compute_window(npfb=2 * self.nfft, taps=2, wfunc=scipy.signal.flattop)

    def demodulate_data(self, data):
        """
        Demodulate the data from the FFT bin
        
        This function assumes that self.select_fft_bins was called to set up the necessary class attributes
        
        data : array of complex data
        
        returns : demodulated data in an array of the same shape and dtype as *data*
        """
        demod = np.zeros_like(data)
        t = np.arange(data.shape[0])
        for n, ich in enumerate(self.readout_selection):
            phi0 = self.phases[ich]
            k = self.tone_bins[ich]
            m = self.fft_bins[ich]
            if m >= self.nfft / 2:
                sign = 1.0
            else:
                sign = -1.0
            nfft = self.nfft
            ns = self.tone_nsamp
            foffs = (2 * k * nfft - m * ns) / float(ns)
            wc = self._window_response(foffs / 2) * (self.tone_nsamp / 2.0 ** 18)
            demod[:, n] = wc * np.exp(sign * 1j * (2 * np.pi * foffs * t + phi0)) * data[:, n]
            if m >= self.nfft / 2:
                demod[:, n] = np.conjugate(demod[:, n])
        return demod




def test_sweep(ri):
    data = []
    tones = []
    ri.r.write_int('sync', 0)
    ri.r.write_int('sync', 1)
    ri.r.write_int('sync', 0)
    for k in range(ri.fft_bins.shape[0] / 4):
        ri.select_fft_bins(range(k * 4, (k + 1) * 4))
        time.sleep(0.1)
        d, addr = ri.get_data(2)
        data.append(d)
        tones.append(ri.tone_bins[ri.readout_selection])
    tones = np.concatenate(tones)
    order = tones.argsort()
    davg = np.concatenate([x.mean(0) for x in data])
    davg = davg[order]
    tones = tones[order]
    return tones, davg, data
