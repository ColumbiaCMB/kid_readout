"""
Calculate frequencies and sample rates using roach state.

All frequencies and rates in the roach state dictionary are assumed to be in Hz, not MHz.

These functions should eventually be merged into the roach interface.
"""
from __future__ import division
import numpy as np


def baseband_frequency(roach_state, tone_bin):
    temporary_frequency = roach_state['adc_sample_rate'] * tone_bin / roach_state['num_tone_samples']
    if roach_state['heterodyne']:
        return np.where(temporary_frequency >= roach_state['adc_sample_rate'] / 2,
                        temporary_frequency - roach_state['adc_sample_rate'],
                        temporary_frequency)
    else:
        return temporary_frequency


def frequency(roach_state, tone_bin):
    if roach_state['heterodyne']:
        return roach_state['lo_frequency'] + baseband_frequency(roach_state, tone_bin)
    else:
        return baseband_frequency(roach_state, tone_bin)


def stream_sample_rate(roach_state):
    if roach_state['heterodyne']:
        # In the heterodyne case, the number of complex samples per FFT is just num_filterbank_channels.
        return roach_state['adc_sample_rate'] / roach_state['num_filterbank_channels']
    else:
        # In the baseband case, the number of real samples per FFT is 2 * num_filterbank_channels.
        return roach_state['adc_sample_rate'] / (2 * roach_state['num_filterbank_channels'])

def modulation_period_samples(roach_state):
    if roach_state.modulation_output != 2:
        return 0
    else:
        if roach_state.heterodyne:
            return 2**(roach_state.modulation_rate+1)
        else:
            return 2**(roach_state.modulation_rate)

def packet_phase(packet_sequence_number, offset_frequencies, num_channels, maximum_period, num_filterbank_channels):
    samples_per_packet = 1024  # hardcoded in roach FPGA design. 4096 payload bytes and 4 bytes per sample
    samples_per_channel_per_packet = samples_per_packet // num_channels
    # If each packet holds at least as many samples per channel as the longest possible demodulated period,
    # then no packet phase correction is needed
    if samples_per_channel_per_packet >= maximum_period:
        return np.zeros_like(offset_frequencies)
    fft_frame_number = packet_sequence_number // num_filterbank_channels
    sample_number = fft_frame_number * samples_per_channel_per_packet
    reduced_sample_number = sample_number % maximum_period
    phase = -2*np.pi*reduced_sample_number*offset_frequencies
    return phase

def packet_phase_original(seq_no,offset_frequencies,nchan,nfft,ns):
    packet_bins = 1024    #this is hardcoded from the roach. number of fft bins that fit in 1 udp packet
    packet_counts = nfft * packet_bins
    chan_counts = packet_counts // nchan
    shift = int(np.log2(chan_counts)) - 1
    modn = ns / chan_counts
    if modn == 0:
        modn = 1
    multy = ns / nfft
    seq_no = seq_no >> shift
    seq_no %= modn
    return np.exp(-1j * 2. * np.pi * seq_no * offset_frequencies * multy / modn)


def tone_offset_frequency(tone_bin,tone_num_samples,fft_bin,nfft):
    k = tone_bin
    m = fft_bin
    ns = tone_num_samples
    return nfft * (k / float(ns)) - m


def get_offset_frequencies_period(offset_frequencies):
    period = 1
    while not np.all(np.round(offset_frequencies*period)==offset_frequencies*period):
        period = period * 2
    return period