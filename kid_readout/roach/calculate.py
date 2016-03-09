from __future__ import division
import numpy as np


def baseband_frequency_MHz(roach_state, tone_bin):
    temporary_frequency_MHz = roach_state['adc_sample_rate_MHz'] * tone_bin / roach_state['num_tone_samples']
    if roach_state['heterodyne']:
        return np.where(temporary_frequency_MHz >= roach_state['adc_sample_rate_MHz'] / 2,
                        temporary_frequency_MHz - roach_state['adc_sample_rate_MHz'],
                        temporary_frequency_MHz)
    else:
        return temporary_frequency_MHz


def frequency_MHz(roach_state, tone_bin):
    if roach_state['heterodyne']:
        return roach_state['local_oscillator_frequency_MHz'] + baseband_frequency_MHz(roach_state, tone_bin)
    else:
        return baseband_frequency_MHz(roach_state, tone_bin)


# TODO: explain this factor of 2
def output_sample_rate(roach_state):
    if roach_state['heterodyne']:
        return 1e6 * roach_state['adc_sample_rate_MHz'] / roach_state['nfft']
    else:
        return 1e6 * roach_state['adc_sample_rate_MHz'] / (2 * roach_state['nfft'])


