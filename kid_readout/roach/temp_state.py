"""
This module has functions for generating fake roach state. These could be moved elsewhere.
"""
from __future__ import division
import numpy as np


def state(ri):
    """
    Return two dicts containing state information from the given roach class.

    The first dictionary contains arrays, and the second contains scalars.

    :param ri: an instance of a subclass of roach.Interface
    :return: two dicts containing state information
    """
    array = {}
    other = {}
    array['tone_bin'] = ri.tonebin.copy()
    if isinstance(ri.amps, np.ndarray):
        array['amplitude'] = ri.amps.copy()
    else:
        array['amplitude'] = ri.amps * np.ones(ri.tonebin.size)
    array['phase'] = ri.phases.copy()
    # TODO: calculate this correctly!
    array['tone_index'] = ri.tonebin.copy()
    # TODO: verify!
    array['fft_bin'] = ri.fft_bins[ri.readout_selection]
    # TODO: figure out which are the fundamental roach values and extract only these; the others should be calculated.
    return array, other


def fake_baseband(f_min=100e6, f_max=200e6, num_tones=16, num_tone_samples=2**19, num_filterbank_channels=16384):
    frequency = np.linspace(f_min, f_max, num_tones)
    adc_dac_sample_rate = 512e6
    tone_bin = np.round(frequency / adc_dac_sample_rate * num_tone_samples).astype(np.int)
    # This calculation taken from baseband:
    tone_bins_per_fft_bin = num_tone_samples / (2 * num_filterbank_channels)
    fft_bin = np.round(tone_bin / tone_bins_per_fft_bin).astype(np.int)
    array = {'tone_bin': tone_bin,
             'amplitude': np.ones(num_tones),
             'phase': np.zeros(num_tones),
             'tone_index': tone_bin.copy(),  # This is true for almost all legacy data.
             'fft_bin': fft_bin}
    other = {'boffile': 'boffile',
             'delay_estimate': 0,
             'heterodyne': False,
             'adc_attenuation': 31.5,
             'dac_attenuation': 31.5,
             'output_attenuation': 41.5,
             'num_tones': num_tones,
             'modulation': {'rate': 0,
                            'output': 0},
             'adc_sample_rate': adc_dac_sample_rate,
             'dac_sample_rate': adc_dac_sample_rate,
             'num_filterbank_channels': num_filterbank_channels,
             'num_tone_samples': num_tone_samples,
             'wavenorm': 1}
    return array, other