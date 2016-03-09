"""
Calculate frequencies and sample rates using roach state.

All frequencies and rates in the roach state dictionary are assumed to be in Hz, not MHz.
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


# TODO: explain this factor of 2
def output_sample_rate(roach_state):
    if roach_state['heterodyne']:
        return roach_state['adc_sample_rate'] / roach_state['num_filterbank_channels']
    else:
        return 1e6 * roach_state['adc_sample_rate'] / (2 * roach_state['num_filterbank_channels'])


