"""
This module contains helper functions and data structures for running tests.
"""
import numpy as np

from kid_readout.measurement import core, basic
from kid_readout.measurement.acquire import acquire
from kid_readout.roach import baseband  # TODO: incorporate heterodyne, r2heterodyne
from kid_readout.roach.tests import mock_roach, mock_valon

bad_node_paths = ['', ' ', '"', ':', '\\', '?', '!', 'bad-hyphen', '0number', '//', '/bad/end/']
good_node_paths = ['/', 'relative', '/absolute', '/2/good', 'underscore_is_fine/_/__really__', '0/12/345']

corners = {'zero_int': 0,
           'zero_float': 0.,
           'one_int': 1,
           'one_float': 1.,
           'minus_one_int': -1,
           'minus_one_float': -1.,
           'two_int': 2,
           'two_float': 2.,
           'none': None,
           'true': True,
           'false': False,
           'empty_list': [],
           'int_list': [-1, 0, 1, 2],
           'float_list': [-0.1, 1, np.pi],
           'str_list': ['zero', 'one', 'two', ''],
           'bool_list': [False, True, False],
           'none_dict': {'none': None},
           'dict_dict': {'one': 1, 'a_dict': {'none': None, 'false': False, 'another_dict': {}}},
           'list_dict': {'empty_list': [],
                         'int_list': [-1, 0, 1, 2],
                         'float_list': [-0.1, 1, np.pi],
                         'str_list': ['zero', 'one', 'two', ''],
                         'bool_list': [False, True, False]}}


class CornerCases(core.Measurement):

    _version = 0

    def __init__(self,
                 zero_int=0,
                 zero_float=0.,
                 one_int=1,
                 one_float=1.,
                 minus_one_int=-1,
                 minus_one_float=-1.,
                 two_int=2,
                 two_float=2.,
                 none=None,
                 true=True,
                 false=False,
                 empty_list=[],
                 int_list=[-1, 0, 1, 2],
                 float_list=[-0.1, 1, np.pi],
                 str_list=['zero', 'one', 'two', ''],
                 bool_list=[False, True, False],
                 none_dict={'none': None},
                 dict_dict={'one': 1, 'a_dict': {'none': None, 'false': False, 'another_dict': {}}},
                 list_dict={'empty_list': [],
                            'int_list': [-1, 0, 1, 2],
                            'float_list': [-0.1, 1, np.pi],
                            'str_list': ['zero', 'one', 'two', ''],
                            'bool_list': [False, True, False]},
                 state=corners, description='CornerCases'):
        self.zero_int = zero_int
        self.zero_float = zero_float
        self.one_int = one_int
        self.one_float = one_float
        self.minus_one_int = minus_one_int
        self.minus_one_float = minus_one_float
        self.two_int = two_int
        self.two_float = two_float
        self.none = none
        self.true = true
        self.false = false
        self.empty_list = empty_list
        self.int_list = int_list
        self.float_list = float_list
        self.str_list = str_list
        self.bool_list = bool_list
        self.none_dict = none_dict
        self.dict_dict = dict_dict
        self.list_dict = list_dict
        super(CornerCases, self).__init__(state=state, description=description)


def fake_stream_array(num_tones=16, num_tone_samples=2 ** 21, length_seconds=0.01,
                      state={'I_am_a': 'fake stream array'}, description='fake stream array'):
    frequency = np.linspace(100, 200, num_tones)
    ri = baseband.RoachBaseband(roach=mock_roach.MockRoach('roach'), adc_valon=mock_valon.MockValon(), initialize=False)
    ri.set_tone_freqs(frequency, nsamp=num_tone_samples)
    ri.select_fft_bins(np.arange(frequency.size))
    return ri.get_measurement(length_seconds, state=state, description=description)


def fake_single_stream(num_tone_samples=2 ** 21, length_seconds=0.01,
                       state={'I_am_a': 'fake single stream'}, description='fake single stream'):
    stream_array = fake_stream_array(num_tones=1, num_tone_samples=num_tone_samples,
                                     length_seconds=length_seconds, state=state, description=description)
    return stream_array[0]


def fake_sweep_array(num_tones=16, num_waveforms=32, num_tone_samples=2 ** 21, length_seconds=0.01,
                     state={'I_am_a': 'fake sweep array'}, description='fake sweep array'):
    ri = baseband.RoachBaseband(roach=mock_roach.MockRoach('roach'), adc_valon=mock_valon.MockValon(), initialize=False)
    center_frequencies = np.linspace(100, 200, num_tones)
    offsets = np.linspace(-100e-3, 100e-3, num_waveforms)
    tone_banks = [center_frequencies + offset for offset in offsets]
    return acquire.run_sweep(ri=ri, tone_banks=tone_banks, num_tone_samples=num_tone_samples,
                              length_seconds=length_seconds, state=state, description=description)


def fake_single_sweep(num_waveforms=32, num_tone_samples=2 ** 21, length_seconds=0.01,
                     state={'I_am_a': 'fake single sweep'}, description='fake single sweep'):
    sweep_array = fake_sweep_array(num_tones=1, num_waveforms=num_waveforms, num_tone_samples=num_tone_samples,
                                   length_seconds=length_seconds, state=state, description=description)
    return sweep_array[0]


def fake_sweep_stream_array(num_tones=16, sweep_num_waveforms=32, sweep_num_tone_samples=2 ** 21,
                            sweep_length_seconds=0.01, stream_num_tone_samples=2 ** 21, stream_length_seconds=0.01,
                            state={'I_am_a': 'fake sweep stream array'}, description='fake sweep stream array'):
    sweep_array = fake_sweep_array(num_tones=num_tones, num_waveforms=sweep_num_waveforms,
                                   num_tone_samples=sweep_num_tone_samples, length_seconds=sweep_length_seconds)
    stream_array = fake_stream_array(num_tones=num_tones, num_tone_samples=stream_num_tone_samples,
                                     length_seconds=stream_length_seconds)
    return basic.SweepStreamArray(sweep_array=sweep_array, stream_array=stream_array, state=state,
                                  description=description)


def fake_single_sweep_stream(sweep_num_waveforms=32, sweep_num_tone_samples=2 ** 21, sweep_length_seconds=0.01,
                             stream_num_tone_samples=2 ** 21, stream_length_seconds=0.01,
                             state={'I_am_a': 'fake single sweep stream'}, description='fake single sweep stream'):
    ssa = fake_sweep_stream_array(num_tones=1, sweep_num_waveforms=sweep_num_waveforms,
                                  sweep_num_tone_samples=sweep_num_tone_samples,
                                  sweep_length_seconds=sweep_length_seconds,
                                  stream_num_tone_samples=stream_num_tone_samples,
                                  stream_length_seconds=stream_length_seconds, state=state, description=description)
    return ssa[0]
