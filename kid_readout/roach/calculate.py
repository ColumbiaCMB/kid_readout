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


def audio_sample_rate(roach_state):
    if roach_state['heterodyne']:
        # In the heterodyne case, the number of complex samples per FFT is just num_filterbank_channels.
        return roach_state['adc_sample_rate'] / roach_state['num_filterbank_channels']
    else:
        # In the baseband case, the number of real samples per FFT is 2 * num_filterbank_channels.
        return roach_state['adc_sample_rate'] / (2 * roach_state['num_filterbank_channels'])
