import socket
import time

import numpy as np
import scipy.signal
from kid_readout.roach.interface import RoachInterface
from kid_readout.roach.tools import compute_window
import kid_readout.roach.udp_catcher
import requests


try:
    import numexpr

    have_numexpr = True
except ImportError:
    have_numexpr = False


class RoachHeterodyne(RoachInterface):

    def __init__(self, roach=None, wafer=0, roachip='roach', adc_valon=None, host_ip=None,
                 nfs_root='/srv/roach_boot/etch', lo_valon=None):
        """
        Class to represent the heterodyne readout system (high-frequency (1.5 GHz), IQ mixers)

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
        super(RoachHeterodyne,self).__init__(roach=roach, roachip=roachip, adc_valon=adc_valon, host_ip=host_ip,
                 nfs_root=nfs_root, lo_valon=lo_valon)

        self.lo_frequency = 0.0
        self.heterodyne = True
        #self.boffile = 'iq2xpfb14mcr6_2015_May_11_2241.bof'
        self.boffile = 'iq2xpfb14mcr7_2015_Nov_25_0907.bof'

        self.wafer = wafer
        self.raw_adc_ns = 2 ** 12  # number of samples in the raw ADC buffer
        self.nfft = 2 ** 14
        self._fpga_output_buffer = 'ppout%d' % wafer

        self._general_setup()

        self.demodulator = Demodulator()
        self.attenuator = Attenuator()

    def load_waveforms(self, i_wave, q_wave, fast=True, start_offset=0):
        """
        Load waveforms for the two DACs

        i_wave,q_wave : arrays of 16-bit (dtype='i2') integers with waveforms for the two DACs

        fast : boolean
            decide what method for loading the dram
        """
        data = np.zeros((2 * i_wave.shape[0],), dtype='>i2')
        data[0::4] = i_wave[::2]
        data[1::4] = i_wave[1::2]
        data[2::4] = q_wave[::2]
        data[3::4] = q_wave[1::2]
        self._load_dram(data, fast=fast, start_offset=start_offset*data.shape[0])

    def set_tone_freqs(self, freqs, nsamp, amps=None):
        baseband_freqs = freqs-self.lo_frequency
        actual_baseband_freqs = self.set_tone_baseband_freqs(baseband_freqs,nsamp,amps=amps)
        return actual_baseband_freqs + self.lo_frequency

    def set_tone_baseband_freqs(self, freqs, nsamp, amps=None):
        """
        Set the stimulus tones to generate

        freqs : array of frequencies in MHz
            For Heterodyne system, these can be positive or negative to produce tones above and
            below the local oscillator frequency.
        nsamp : int, must be power of 2
            number of samples in the playback buffer. Frequency resolution will be fs/nsamp
        amps : optional array of floats, same length as freqs array
            specify the relative amplitude of each tone. Can set to zero to read out a portion
            of the spectrum with no stimulus tone.

        returns:
        actual_freqs : array of the actual frequencies after quantization based on nsamp
        """
        bins = np.round((freqs / self.fs) * nsamp).astype('int')
        actual_freqs = self.fs * bins / float(nsamp)
        bins[bins < 0] = nsamp + bins[bins < 0]
        self.set_tone_bins(bins, nsamp, amps=amps)
        self.fft_bins = self.calc_fft_bins(bins, nsamp)
        if self.fft_bins.shape[1] > 4:
            readout_selection = range(4)
        else:
            readout_selection = range(self.fft_bins.shape[1])
        self.select_bank(0)
        self.select_fft_bins(readout_selection)
        self.save_state()
        return actual_freqs

    def add_tone_freqs(self, freqs, amps=None, overwrite_last=False):
        baseband_freqs = freqs-self.lo_frequency
        actual_baseband_freqs = self.add_tone_baseband_freqs(baseband_freqs, amps=amps, overwrite_last=overwrite_last)
        return actual_baseband_freqs + self.lo_frequency

    def add_tone_baseband_freqs(self, freqs, amps=None, overwrite_last=False):
        if freqs.shape[0] != self.tone_bins.shape[1]:
            raise ValueError("freqs array must contain same number of tones as original waveforms")
        # This is a hack that doesn't handle bank selection at all and may have additional problems.
        if overwrite_last:  # Delete the last waveform and readout selection entry.
            self.tone_bins = self.tone_bins[:-1, :]
            self.fft_bins = self.fft_bins[:-1, :]
        nsamp = self.tone_nsamp
        bins = np.round((freqs / self.fs) * nsamp).astype('int')
        actual_freqs = self.fs * bins / float(nsamp)
        bins[bins < 0] = nsamp + bins[bins < 0]
        self.add_tone_bins(bins, amps=amps)
        self.fft_bins = np.vstack((self.fft_bins, self.calc_fft_bins(bins, nsamp)))
        self.save_state()
        return actual_freqs


    def set_tone_bins(self, bins, nsamp, amps=None, load=True, normfact=None,phases=None,iq_delay=0):
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
        spec = np.zeros((nwaves, nsamp), dtype='complex')
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
        wave = np.fft.ifft(spec, axis=1)
        self.wavenorm = np.abs(wave).max()
        if normfact is not None:
            wn = (2.0 / normfact) * len(bins) / float(nsamp)
            print "ratio of current wavenorm to optimal:", self.wavenorm / wn
            self.wavenorm = wn
        q_rwave = np.round((wave.real / self.wavenorm) * (2 ** 15 - 1024)).astype('>i2')
        q_iwave = np.round((wave.imag / self.wavenorm) * (2 ** 15 - 1024)).astype('>i2')
        q_iwave = np.roll(q_iwave, iq_delay, axis=1)    
        q_rwave.shape = (q_rwave.shape[0] * q_rwave.shape[1],)
        q_iwave.shape = (q_iwave.shape[0] * q_iwave.shape[1],)
        self.q_rwave = q_rwave
        self.q_iwave = q_iwave
        if load:
            self.load_waveforms(q_rwave,q_iwave)
        self.save_state()

    def add_tone_bins(self, bins, amps=None):
        nsamp = self.tone_nsamp
        spec = np.zeros((nsamp,), dtype='complex')
        self.tone_bins = np.vstack((self.tone_bins, bins))
        phases = self.phases
        if amps is None:
            amps = 1.0
        # self.amps = amps  # TODO: Need to figure out how to deal with this

        spec[bins] = amps * np.exp(1j * phases)
        wave = np.fft.ifft(spec)
        q_rwave = np.round((wave.real / self.wavenorm) * (2 ** 15 - 1024)).astype('>i2')
        q_iwave = np.round((wave.imag / self.wavenorm) * (2 ** 15 - 1024)).astype('>i2')
        start_offset = self.tone_bins.shape[0] - 1
        self.load_waveforms(q_rwave, q_iwave, start_offset=start_offset)
        self.save_state()


    def calc_fft_bins(self, tone_bins, nsamp):
        """
        Calculate the FFT bins in which the tones will fall

        tone_bins: array of integers
            the tone bins (0 to nsamp - 1) which contain tones

        nsamp : length of the playback bufffer

        returns: fft_bins, array of integers.
        """
        tone_bins_per_fft_bin = nsamp / (self.nfft)
        fft_bins = np.round(tone_bins / float(tone_bins_per_fft_bin)).astype('int')
        return fft_bins

    def fft_bin_to_index(self, bins):
        """
        Convert FFT bins to FPGA indexes
        """
        idx = bins.copy()
        return idx

    def select_fft_bins(self, readout_selection):
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
        offset = 2
        idxs = self.fft_bin_to_index(self.fft_bins[self.bank,readout_selection])
        order = idxs.argsort()
        idxs = idxs[order]
        self.readout_selection = np.array(readout_selection)[order]
        self.fpga_fft_readout_indexes = idxs
        self.readout_fft_bins = self.fft_bins[self.bank, self.readout_selection]

        binsel = np.zeros((self.fpga_fft_readout_indexes.shape[0] + 1,), dtype='>i4')
        # evenodd = np.mod(self.fpga_fft_readout_indexes,2)
        #binsel[:-1] = np.mod(self.fpga_fft_readout_indexes/2-offset,self.nfft/2)
        #binsel[:-1] += evenodd*2**16
        binsel[:-1] = np.mod(self.fpga_fft_readout_indexes - offset, self.nfft)
        binsel[-1] = -1
        self.r.write('chans', binsel.tostring())

    def demodulate_data(self,data):
        bank = self.bank
        demod = np.zeros_like(data)
        for n, ich in enumerate(self.readout_selection):
            demod[:,n] = self.demodulator.demodulate(data[:,n],
                                            tone_bin=self.tone_bins[bank,ich],
                                            tone_num_samples=self.tone_nsamp,
                                            tone_phase=self.phases[ich],
                                            fft_bin=self.fft_bins[bank,ich])
        return demod

    def demodulate_data_original(self, data):
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
            k = self.tone_bins[bank,ich]
            m = self.fft_bins[bank,ich]
            if m >= self.nfft / 2:
                sign = -1.0
                doconj = True
            else:
                sign = -1.0
                doconj = False
            nfft = self.nfft
            ns = self.tone_nsamp
            foffs = (k * nfft - m * ns) / float(ns)
            demod[:, n] = np.exp(sign * 1j * (2 * np.pi * foffs * t + phi0)) * data[:, n]
            if doconj:
                demod[:, n] = np.conjugate(demod[:, n])
        return demod

    def get_data(self, nread=2, demod=True):
        return self.get_data_udp(nread=nread, demod=demod)

    def get_data_udp(self, nread=2, demod=True):
        chan_offset = 2
        nch = self.fpga_fft_readout_indexes.shape[0]
        udp_channel = (self.fpga_fft_readout_indexes//2 + chan_offset) % (self.nfft//2)
        data, seqnos = kid_readout.roach.udp_catcher.get_udp_data(self, npkts=nread * 16 * nch, streamid=1,
                                                chans=udp_channel,
                                                nfft=self.nfft//2, addr=(self.host_ip, 12345))  # , stream_reg, addr)
        if demod:
            data = self.demodulate_data(data)
        return data, seqnos

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
        print "getting data"
        bufname = 'ppout%d' % self.wafer
        chan_offset = 2
        draw, addr, ch = self._read_data(nread, bufname)
        if not np.all(ch == ch[0]):
            print "all channel registers not the same; this case not yet supported"
            return draw, addr, ch
        if not np.all(np.diff(addr) < 8192):
            print "address skip!"
        nch = self.readout_selection.shape[0]
        dout = draw.reshape((-1, nch))
        shift = np.flatnonzero(self.fpga_fft_readout_indexes / 2 == (ch[0] - chan_offset))[0] - (nch - 1)
        print shift
        dout = np.roll(dout, shift, axis=1)
        if demod:
            dout = self.demodulate_data(dout)
        return dout, addr

    def set_lo(self, lomhz=1200.0, chan_spacing=2.0, modulator_lo_power=5, demodulator_lo_power=5):
        """
        Set the local oscillator frequency for the IQ mixers

        lomhz: float, frequency in MHz

        lo_level: LO power level on the valon. options are [-4, -1, 2, 5]
        """
        #TODO: Fix this after valon is updated
        if self.lo_valon is None:
            self.adc_valon.set_rf_level(8,2)
            self.adc_valon.set_frequency_b(lomhz, chan_spacing=chan_spacing)
            self.lo_frequency = lomhz
            self.save_state()
        else:
            #out1 goes to demod at 0dBm
            #out2 goes to mod at 5dBm
            power_settings = [-4, -1, 2, 5]
            if demodulator_lo_power in power_settings:
                self.lo_valon.set_rf_level(0,demodulator_lo_power) 
            else:
                print "demodulator_lo_level not available, using full power" 
                self.lo_valon.set_rf_level(0,5) 
            if modulator_lo_power in power_settings:
                self.lo_valon.set_rf_level(8,modulator_lo_power)
            else:
                print "modulator_lo_level not available, using full power" 
                self.lo_valon.set_rf_level(8,5)
            self.lo_valon.set_frequency_a(lomhz, chan_spacing=chan_spacing)
            self.lo_valon.set_frequency_b(lomhz, chan_spacing=chan_spacing)
            self.lo_frequency = lomhz
            self.save_state()

    def set_dac_attenuator(self, attendb):
        if attendb < 0 or attendb > 63:
            raise ValueError("DAC Attenuator must be between 0 and 63 dB. Value given was: %s" % str(attendb))

        if attendb > 31.5:
            attena = 31.5
            attenb = attendb - attena
        else:
            attena = attendb
            attenb = 0
        self.set_attenuator(attena, le_bit=0x01)
        self.set_attenuator(attenb, le_bit=0x80)
        self.dac_atten = int(attendb * 2) / 2.0

    def set_adc_attenuator(self, attendb):
        if attendb < 0 or attendb > 31.5:
            raise ValueError("ADC Attenuator must be between 0 and 31.5 dB. Value given was: %s" % str(attendb))
        self.set_attenuator(attendb, le_bit=0x02)
        self.adc_atten = int(attendb * 2) / 2.0
        
    def get_fftout_snap(self):
        self.r.wselfte_int('fftout_ctrl',0)
        self.r.wselfte_int('fftout_ctrl',1)
        self._sync()
        while self.r.read_int('fftout_status') != 0x8000:
            pass
        return np.fromstselfng(self.r.read('fftout_bram',2**15),dtype='>i2').astype('float').view('complex')

    def _set_fs(self, fs, chan_spacing=2.0):
        """
        Set sampling frequency in MHz
        Note, this should generally not be called without also reprogramming the ROACH
        Use initialize() instead
        """
        if self.adc_valon is None:
            print "Could not set Valon; none available"
            return
        self.adc_valon.set_frequency_a(fs, chan_spacing=chan_spacing)
        self.fs = fs


class Demodulator(object):
    def __init__(self,nfft=2**14,num_taps=2,window=scipy.signal.flattop,interpolation_factor=64):
        self.nfft = nfft
        self.num_taps = num_taps
        self.window_function = window
        self.interpolation_factor = interpolation_factor
        self._window_frequency,self._window_response = self.compute_window_frequency_response(self.compute_pfb_window(),
                                                                       interpolation_factor=interpolation_factor)

    def compute_pfb_window(self):
        raw_window = self.window_function(self.nfft*self.num_taps)
        sinc = np.sinc(np.arange(self.nfft*self.num_taps)/(1.*self.nfft) - self.num_taps/2.0)
        return raw_window*sinc

    def compute_window_frequency_response(self,window,interpolation_factor=64):
        response = np.abs(np.fft.fftshift(np.fft.fft(window,window.shape[0]*interpolation_factor)))
        response = response/response.max()
        normalized_frequency = (np.arange(-len(response)/2.,len(response)/2.)/interpolation_factor)/2.
        return normalized_frequency,response

    def compute_pfb_response(self,normalized_frequency):
        return 1/np.interp(normalized_frequency,self._window_frequency,self._window_response)

    def demodulate(self,data,tone_bin,tone_num_samples,tone_phase,fft_bin):
        phi0 = tone_phase
        k = tone_bin
        m = fft_bin
        nfft = self.nfft
        ns = tone_num_samples
        foffs = tone_offset_frequency(tone_bin,tone_num_samples,fft_bin,nfft)
        wc = self.compute_pfb_response(foffs)
        t = np.arange(data.shape[0])
        demod = wc*np.exp(-1j * (2 * np.pi * foffs * t + phi0)) * data
        return demod


def tone_offset_frequency(tone_bin,tone_num_samples,fft_bin,nfft):
    k = tone_bin
    m = fft_bin
    nfft = nfft
    ns = tone_num_samples
    return nfft * (k / float(ns)) - m

class Attenuator(object):
    def __init__(self,tempip='http://192.168.1.211/'):
        self.ip = tempip
        self.att = self.get_att()

    def get_att(self):
        return float(self.get_query("ATT??"))
    
    def set_att(self, newatt):
        if newatt > 62:
            print "Setting attenuation too high.  Max is 62"
        qatt = "SETATT="+str(newatt)
        return float(self.get_query(qatt))
    
    def get_query(self, query):
        query = self.ip + query
        q = requests.get(query)
        return q.content
