import numpy as np
import scipy.signal

from kid_readout.roach.calculate import packet_phase, tone_offset_frequency, get_offset_frequencies_period


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
        offset_frequency = tone_offset_frequency(tone_bin, tone_num_samples, fft_bin, nfft)
        wc = self.compute_pfb_response(offset_frequency)
        t = np.arange(data.shape[0])
        demod = wc*np.exp(-1j * (2 * np.pi * offset_frequency * t + phi0)) * data
        if type(seq_nos) is np.ndarray:
            pphase = packet_phase(seq_nos[0], offset_frequency, nchan, nfft, ns)
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
        self.num_channels = self.tone_bins.shape[0]
        self.tone_nsamp = tone_nsamp
        self.phases = phases
        self.fft_bins = fft_bins
        self.offset_frequencies = tone_offset_frequency(self.tone_bins, self.tone_nsamp, self.fft_bins, self.nfft)
        self.max_period = get_offset_frequencies_period(self.offset_frequencies)

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
        demod_wave, wave_period = self.create_demodulation_waveform(data.shape, sequence_numbers)
        if (data.shape[0] % wave_period):
            data = data[: wave_period*(data.shape[0]//wave_period),:]
        return demod_wave * data

    def create_demodulation_waveform(self, data_shape, seq_nos):
        # Handles dropped packets if they are included as NaNs.
        # If do not want to use NaNs then need to calculate phase from seq_nos...
        offset_frequencies = tone_offset_frequency(self.tone_bins, self.tone_nsamp, self.fft_bins, self.nfft)
        wave_period = get_offset_frequencies_period(offset_frequencies)
        wc = self.compute_pfb_response(offset_frequencies)
        pphase = packet_phase(seq_nos[0], offset_frequencies, self.num_channels, self.nfft, self.tone_nsamp)
        hardware_delay = -self.hardware_delay_samples*self.tone_bins/float(self.tone_nsamp)
        t = np.arange(data_shape[0])#wave_period)
        wave = wc*np.exp(-1j * (2 * np.pi * (np.outer(t, offset_frequencies) + hardware_delay) + self.phases)) * pphase
        #wave_mat = np.lib.stride_tricks.as_strided(wave, shape=data_shape, strides=(0,wave.strides[1]))
        wave_mat=wave
        return wave_mat, wave_period


