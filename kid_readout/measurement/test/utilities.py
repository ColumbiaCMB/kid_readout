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
           'none': None,
           'true': True,
           'false': False,
           'empty_list': [],
           'int_list': [-1, 0, 1, 2],
           'float_list': [-0.1, 1, np.pi],
           'str_list': ['zero', 'one', 'two', ''],
           'bool_list': [False, True, False],
           'none_dict': {'none': None},
           'dict_dict': {'1': 1, 'dict': {'none': None, 'false': False, 'another_dict': {}}},
           'list_dict': {'empty_list': [],
                         'int_list': [-1, 0, 1, 2],
                         'float_list': [-0.1, 1, np.pi],
                         'str_list': ['zero', 'one', 'two', ''],
                         'bool_list': [False, True, False]}}


class CornerMeasurement(core.Measurement):

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
                 dict_dict={'1': 1, 'dict': {'none': None, 'false': False, 'another_dict': {}}},
                 list_dict={'empty_list': [],
                            'int_list': [-1, 0, 1, 2],
                            'float_list': [-0.1, 1, np.pi],
                            'str_list': ['zero', 'one', 'two', ''],
                            'bool_list': [False, True, False]},
                 state=corners, description='CornerMeasurement'):
        super(CornerMeasurement, self).__init__(state=state, description=description)
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