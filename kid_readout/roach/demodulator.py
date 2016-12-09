import numpy as np
import scipy.signal
import types

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
        if type(self.window_function) is types.FunctionType:
            raw_window = self.window_function(self.nfft*self.num_taps)
        else:
            raw_window = scipy.signal.get_window(self.window_function, self.nfft*self.num_taps,fftbins=False)
        sinc = np.sinc(np.arange(self.nfft*self.num_taps)/(1.*self.nfft) - self.num_taps/2.0)
        return raw_window*sinc

    def compute_window_frequency_response(self,window,interpolation_factor=64):
        response = np.abs(np.fft.fftshift(np.fft.fft(window,window.shape[0]*interpolation_factor)))
        response = response/response.max()
        normalized_frequency = (np.arange(-len(response)/2., len(response)/2.) /
                                float(interpolation_factor * self.num_taps))
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
            pphase = np.exp(1j*packet_phase(seq_nos[0], offset_frequency, nchan, ns/nfft, nfft))
            demod *= pphase
        if self.hardware_delay_samples != 0:
            demod *= np.exp(2j*np.pi*self.hardware_delay_samples*tone_bin/tone_num_samples)
        return demod

def get_stream_demodulator_from_roach_state(state, state_arrays):
    return StreamDemodulator(tone_bins=state_arrays.tone_bin,
                             phases=state_arrays.tone_phase,
                             fft_bins=state_arrays.filterbank_bin,
                             tone_nsamp=state.num_tone_samples,
                             nfft=state.num_filterbank_channels,
                             hardware_delay_samples=state.hardware_delay_samples,
                             reference_sequence_number=state.reference_sequence_number
                             )

class StreamDemodulator(Demodulator):
    def __init__(self,tone_bins, phases, tone_nsamp, fft_bins, nfft=2**14,num_taps=2,window=scipy.signal.flattop,
                 interpolation_factor=64,
                 hardware_delay_samples=0,reference_sequence_number=0):
        super(StreamDemodulator,self).__init__(nfft=nfft,num_taps=num_taps,window=window,
                                               interpolation_factor=interpolation_factor,
                                               hardware_delay_samples=hardware_delay_samples)

        self.tone_bins = tone_bins
        self.num_channels = self.tone_bins.shape[0]
        self.tone_nsamp = tone_nsamp
        self.phases = phases
        self.fft_bins = fft_bins
        self.reference_sequence_number = reference_sequence_number

        self.samples_per_packet = 1024
        self.samples_per_channel_per_packet = self.samples_per_packet // self.num_channels
        self.sequence_number_increment_per_packet = self.samples_per_channel_per_packet * nfft // 2  # The 2 here is
        # because the sequence number increments once per fpga clock and there are nfft // 2 fpga clocks per sample
        self.offset_frequencies = tone_offset_frequency(self.tone_bins, self.tone_nsamp, self.fft_bins, self.nfft)
        self.max_period = get_offset_frequencies_period(self.offset_frequencies)
        self.pfb_response_correction = self.compute_pfb_response(self.offset_frequencies)


        self.demodulation_lookup = self.create_demodulation_lookup()

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
        demod_wave= self.create_demodulation_waveform(data.shape, sequence_numbers)
        #if (data.shape[0] % wave_period):
        #    data = data[: wave_period*(data.shape[0]//wave_period),:]
        return demod_wave * data

    def create_demodulation_waveform(self, data_shape, seq_nos):
        # Handles dropped packets if they are included as NaNs.
        # If do not want to use NaNs then need to calculate phase from seq_nos...
        pphase = np.exp(1j*packet_phase(seq_nos[0], self.offset_frequencies, self.num_channels,
                                        self.tone_nsamp//self.nfft,
                                        self.nfft))
        hardware_delay = -self.hardware_delay_samples*self.tone_bins/float(self.tone_nsamp)
        t = np.arange(data_shape[0])#wave_period)
        wave = (self.pfb_response_correction
                * np.exp(-1j * (2 * np.pi * (np.outer(t, self.offset_frequencies) + hardware_delay) + self.phases))
                * pphase)
        #wave_mat = np.lib.stride_tricks.as_strided(wave, shape=data_shape, strides=(0,wave.strides[1]))
        wave_mat=wave
        return wave_mat

    def create_demodulation_lookup(self):
        hardware_delay = -self.hardware_delay_samples*self.tone_bins/float(self.tone_nsamp)
        num_time_steps = self.max_period
        if num_time_steps * self.num_channels < 2 * self.samples_per_packet:
            num_time_steps = 2 * self.samples_per_packet // self.num_channels
        t = np.arange(num_time_steps)
        wave = (self.pfb_response_correction
                * np.exp(-1j * (2 * np.pi * (np.outer(t, self.offset_frequencies) + hardware_delay) + self.phases)))
        wave.shape = (np.prod(wave.shape),)
        return wave

    def lookup_index_from_sequence_num(self,sequence_num):
        lut_size = self.demodulation_lookup.shape[0]
        packet_number = (sequence_num - self.reference_sequence_number)//self.sequence_number_increment_per_packet
        offset = (packet_number*self.samples_per_packet) % lut_size
        return offset

    def decode_and_demodulate_packets(self,packets, assume_not_contiguous=False):
        num_packets = len(packets)
        packet_buffer = np.empty((num_packets,4100),dtype=np.uint8)
        sequence_number_buffer = np.empty((num_packets,),dtype=np.uint32)
        output_buffer = np.empty((num_packets,self.samples_per_packet),dtype=np.complex64)
        bad_packets = self.unpack_packets_into_buffer(packets,packet_buffer)
        skips = self.decode_and_demodulate_packet_buffer(packet_buffer,sequence_number_buffer,output_buffer,
                                                 assume_not_contiguous=assume_not_contiguous)
        return sequence_number_buffer,output_buffer

    def unpack_packets_into_buffer(self,packets,packet_buffer, expected_packet_length=4100):
        index = 0
        bad_packets = 0
        for packet in packets:
            if len(packet) == expected_packet_length:
                packet_buffer[index,:] = np.frombuffer(packet,dtype=np.uint8)
                index += 1
            else:
                bad_packets += 1
        if bad_packets:
            print "got %d bad packets" % bad_packets
        return bad_packets

    def decode_and_demodulate_packet_buffer(self,packet_buffer,sequence_number_buffer,output_buffer,
                                            assume_not_contiguous=False):
        packets_per_buffer = packet_buffer.shape[0]
        sequence_number_buffer[:] = packet_buffer.view('<u4')[:,-1]
        seq_num_diff = np.diff(sequence_number_buffer)
        is_contiguous = np.all(seq_num_diff==self.sequence_number_increment_per_packet)
        data = packet_buffer.view('<i2').astype(np.float32).view(np.complex64)[:,:-1] # note! we need to strip off
        # the sequence number after casting to avoid two memory copies
        if not is_contiguous or assume_not_contiguous:
            for k in range(packets_per_buffer):
                offset = self.lookup_index_from_sequence_num(sequence_number_buffer[k])
                output_buffer[k,:] = data[k,:] * self.demodulation_lookup[offset:offset+self.samples_per_packet]
            skips = (seq_num_diff[seq_num_diff!=self.sequence_number_increment_per_packet]
                     // self.sequence_number_increment_per_packet - 1)
            skips = skips.sum()
            if skips:
                print "skipped %d packets" % skips

        else:
            offset = self.lookup_index_from_sequence_num(sequence_number_buffer[0])

            waveform = np.lib.stride_tricks.as_strided(self.demodulation_lookup[offset:offset+self.samples_per_packet],
                                                       shape=(packets_per_buffer,self.samples_per_packet),
                                                       strides=(0,self.demodulation_lookup.strides[0]))
            np.multiply(data,waveform,out=output_buffer)
            skips = 0
        return skips

