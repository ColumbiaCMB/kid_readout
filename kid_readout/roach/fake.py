"""
This module has functions for generating fake roach state. These could be moved elsewhere.
"""
from __future__ import division
import numpy as np


def baseband_active_state_arrays(f_min=100e6, f_max=200e6, num_tones=16, num_tone_samples=2 ** 20,
                                 num_filterbank_channels=2**14):
    frequency = np.linspace(f_min, f_max, num_tones)
    adc_dac_sample_rate = 512e6
    tone_bin = np.round(frequency / adc_dac_sample_rate * num_tone_samples).astype(np.int)
    # This calculation taken from baseband:
    tone_bins_per_filterbank_bin = num_tone_samples / (2 * num_filterbank_channels)
    filterbank_bin = np.round(tone_bin / tone_bins_per_filterbank_bin).astype(np.int)
    state_arrays = {'tone_bin': tone_bin,
                    'tone_amplitude': np.ones(num_tones),
                    'tone_phase': np.zeros(num_tones),
                    'tone_index': tone_bin.copy(),  # this is true for almost all legacy data.
                    'filterbank_bin': filterbank_bin}
    return state_arrays


def baseband_state(num_tones=16, num_tone_samples=2 ** 20):
    adc_dac_sample_rate = 512e6
    roach_state = {'boffile': 'boffile',
                   'heterodyne': False,
                   'adc_sample_rate': adc_dac_sample_rate,
#                   'dac_sample_rate': adc_dac_sample_rate,
                   'lo_frequency': 0.,
                   'num_tones': num_tones,
                   'modulation_rate': 0,
                   'modulation_output': 0,
                   'waveform_normalization': 1.,
                   'num_tone_samples': num_tone_samples,
                   'num_filterbank_channels': 2**14,
                   'dac_attenuation': 31.5,
                   'adc_attenuation': 31.5,
                   'bank': 0}
    return roach_state
