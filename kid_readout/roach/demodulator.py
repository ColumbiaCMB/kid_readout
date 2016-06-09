import numpy as np
import scipy.signal
import numexpr

class Demodulator(object):
    def __init__(self,nfft=2**14,num_taps=2,window=scipy.signal.flattop,interpolation_factor=64,
                 hardware_delay_samples=0):
        self.nfft = nfft
        self.num_taps = num_taps
        self.window_function = window
        self.interpolation_factor = interpolation_factor
        self.hardware_delay_samples = hardware_delay_samples
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

    def demodulate(self,data,tone_bin,tone_num_samples,tone_phase,fft_bin,nchan,seq_nos=None):
        phi0 = tone_phase
        nfft = self.nfft
        ns = tone_num_samples
        foffs = tone_offset_frequency(tone_bin,tone_num_samples,fft_bin,nfft)
        wc = self.compute_pfb_response(foffs)
        t = np.arange(data.shape[0])
        demod = wc*np.exp(-1j * (2 * np.pi * foffs * t + phi0)) * data
        if type(seq_nos) is np.ndarray:
            pphase = packet_phase(seq_nos[0],foffs,nchan,nfft,ns)
            demod *= pphase
        if self.hardware_delay_samples != 0:
            demod *= np.exp(2j*np.pi*self.hardware_delay_samples*tone_bin/tone_num_samples)
        return demod

class StreamDemodulator(Demodulator):
    def __init__(self,tone_bins, phases, tone_nsamp, fft_bins, nfft=2**14,num_taps=2,window=scipy.signal.flattop,
                 interpolation_factor=64,
                 hardware_delay_samples=0):
        super(StreamDemodulator,self).__init__(nfft=nfft,num_taps=num_taps,window=window,
                                               interpolation_factor=interpolation_factor,
                                               hardware_delay_samples=hardware_delay_samples)

        self.tone_bins = tone_bins
        self.tone_nsamp = tone_nsamp
        self.phases = phases
        self.fft_bins = fft_bins
        self.offset_frequencies = tone_offset_frequency(self.tone_bins, self.tone_nsamp, self.fft_bins, self.nfft)
        self.max_period = get_foffs_period(self.offset_frequencies)

    def demodulate_stream(self, data, sequence_numbers):
        """
        Demodulate a stream of data from all channels

        Parameters
        ----------
        data : array of complex64 (num_samples,num_channels)
        sequence_numbers : array of uint32

        Returns
        -------
        demodulated data in same shape and dtype as input data

        """
        offset_frequencies = tone_offset_frequency(self.tone_bins, self.tone_nsamp, self.fft_bins, self.nfft)
        wave_period = get_foffs_period(offset_frequencies)
        if (data.shape[0] % wave_period):
            data = data[: wave_period*(data.shape[0]//wave_period),:]
        demod_wave, pphase, wc = self.create_demodulation_waveform(data.shape, sequence_numbers, offset_frequencies, wave_period)
        return numexpr.evaluate("pphase * wc * demod_wave * data")

    def create_demodulation_waveform(self, data_shape, seq_nos, foffs, wave_period):
        # Handles dropped packets if they are included as NaNs.
        # If do not want to use NaNs then need to calculate phase from seq_nos...
        wc = self.demodulator.compute_pfb_response(foffs)
        pphase = packet_phase(seq_nos[0], foffs, self.readout_selection.shape[0], self.nfft, self.tone_nsamp)
        t = np.arange(wave_period)
        wave = np.exp(-1j * (2 * np.pi * np.outer(t, foffs) + pphase + self.phases))
        wave_mat = np.lib.stride_tricks.as_strided(wave, shape=data_shape, strides=(0,wave.strides[1]))
        return wave_mat, pphase, wc


def packet_phase(seq_no,foffs,nchan,nfft,ns):
    packet_bins = 1024    #this is hardcoded from the roach. number of fft bins that fit in 1 udp packet
    packet_counts = nfft * packet_bins
    chan_counts = packet_counts / nchan
    shift = int(np.log2(chan_counts)) - 1
    modn = ns / chan_counts
    if modn == 0:
        modn = 1
    multy = ns / nfft
    seq_no = seq_no >> shift
    seq_no %= modn
    return np.exp(-1j * 2. * np.pi * seq_no * foffs * multy / modn)

def tone_offset_frequency(tone_bin,tone_num_samples,fft_bin,nfft):
    k = tone_bin
    m = fft_bin
    nfft = nfft
    ns = tone_num_samples
    return nfft * (k / float(ns)) - m

def get_foffs_period(foffs):
    mask = foffs != 0
    try:
        period = int(np.max(1/foffs[mask]))
    except ValueError:
        period = 1
    return period