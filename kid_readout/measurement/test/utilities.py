"""
This module contains helper functions and data structures for running tests.
"""
import numpy as np
from kid_readout.measurement import core
from kid_readout.roach import baseband
from kid_readout.roach.tests import mock_roach, mock_valon

corners = {'zero_int': 0,
           'zero_float': 0.,
           'one_int': 1,
           'one_float': 1.,
           'minus_one_int': -1,
           'minus_one_float': -1.,
           'two_int': 2,
           'two_float': 2.,
           'None': None,
           'True': True,
           'False': False,
           'empty_list': [],
           'int_list': [-1, 0, 1, 2],
           'float_list': [-0.1, 1, np.pi],
           'str_list': ['zero', 'one', 'two', ''],
           'bool_list': [False, True, False],
           'none_dict': {'None': None},
           'dict_dict': {'1': 1, 'dict': {'None': None, 'False': False, 'another_dict': {}}},
           'list_dict': {'empty_list': [],
                         'int_list': [-1, 0, 1, 2],
                         'float_list': [-0.1, 1, np.pi],
                         'str_list': ['zero', 'one', 'two', ''],
                         'bool_list': [False, True, False]}}


def get_measurement():
    m = core.Measurement(state=corners)
    for k, v in corners.items():
        if k not in ('True', 'False', 'None'):
            setattr(m, k, v)
    return m


def make_stream(tone_index=0, frequency=None, num_tone_samples=2**16, blocks=2, state=None, description=''):
    if frequency is None:
        frequency = np.linspace(100, 200, 16)
    mr = mock_roach.MockRoach('roach')
    mv = mock_valon.MockValon()
    ri = baseband.RoachBaseband(roach=mr, adc_valon=mv, initialize=False)
    ri.set_tone_freqs(frequency, nsamp=num_tone_samples)
    ri.select_fft_bins(np.arange(frequency.size))
    stream_array = ri.get_measurement(blocks, state=state, description=description)
    return stream_array.stream(tone_index)


def make_stream_array(frequency=None, num_tone_samples=2**16, blocks=2, state=None, description=''):
    if frequency is None:
        frequency = np.linspace(100, 200, 16)
    mr = mock_roach.MockRoach('roach')
    mv = mock_valon.MockValon()
    ri = baseband.RoachBaseband(roach=mr, adc_valon=mv, initialize=False)
    ri.set_tone_freqs(frequency, nsamp=num_tone_samples)
    ri.select_fft_bins(np.arange(frequency.size))
    return ri.get_measurement(blocks, state=state, description=description)


def make_sweep():
    pass


def make_sweep_array():
    pass


def make_sweep_stream():
    pass


def make_sweep_stream_array():
    pass