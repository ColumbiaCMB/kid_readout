"""
This module has classes that contain simultaneous multiple-channel measurements.
"""
from __future__ import division

from collections import OrderedDict

import numpy as np
import pandas as pd

from kid_readout.measurement import core
from kid_readout.measurement.single import Stream, Sweep, ResonatorSweep, SweepStream
from kid_readout.roach import calculate
from kid_readout.roach.tests import fake


class StreamArray(core.Measurement):
    """
    This class represents simultaneously-sampled data from multiple channels.
    """

    dimensions = OrderedDict([('tone_bin', ('tone_bin',)),
                              ('tone_amplitude', ('tone_bin',)),
                              ('tone_phase', ('tone_bin',)),
                              ('tone_index', ('tone_index',)),
                              ('filterbank_bin', ('tone_index',)),
                              ('epoch', ('epoch',)),
                              ('s21', ('tone_index', 'epoch'))])

    def __init__(self, tone_bin, tone_amplitude, tone_phase, tone_index, filterbank_bin, epoch, s21, roach_state,
                 state=None, analyze=False, description='StreamArray'):
        """
        Return a new StreamArray instance. The integer array tone_index contains the indices of tone_bin,
        tone_amplitude,
        and tone_phase for the tones demodulated to produce the time-ordered s21 data.

        The tone_bin, tone_amplitude, tone_phase, tone_index, filterbank_bin, and epoch arrays are 1-D, while s21 is
        2-D with
        s21.shape == (tone_index.size, epoch.size)

        :param tone_bin: an array of integers representing the frequencies of the tones played during the measurement.
        :param tone_amplitude: an array of floats representing the amplitudes of the tones played during the
          measurement.
        :param tone_phase: an array of floats representing the radian phases of the tones played during the measurement.
        :param tone_index: an intarray for which tone_bin[tone_index] corresponds to the frequency used to produce s21.
        :param filterbank_bin: an integer that is the filter bank bin in which the tone lies.
        :param epoch: an array of floats representing the unix timestamp when the data was recorded.
        :param s21: a 2-D array of complex floats containing the data, demodulated or not..
        :param roach_state: a dict containing state information for the roach.
        :param state: a dict containing all non-roach state information.
        :param analyze: if True, call the analyze() method at the end of instantiation.
        :param description: a string describing this measurement.
        :return: a new StreamArray instance.
        """
        self.tone_bin = tone_bin
        self.tone_amplitude = tone_amplitude
        self.tone_phase = tone_phase
        self.tone_index = tone_index
        self.filterbank_bin = filterbank_bin
        self.epoch = epoch
        self.s21 = s21
        self.roach_state = core.to_state_dict(roach_state)
        self._frequency = None
        self._baseband_frequency = None
        self._stream_sample_rate = None
        self._s21_mean = None
        self._s21_mean_error = None
        super(StreamArray, self).__init__(state=state, analyze=analyze, description=description)

    def analyze(self):
        self.baseband_frequency
        self.frequency
        self.stream_sample_rate
        self.s21_mean
        self.s21_mean_error

    @property
    def frequency(self):
        if self._frequency is None:
            self._frequency = calculate.frequency(self.roach_state, self.tone_bin[self.tone_index])
        return self._frequency

    @property
    def frequency_MHz(self):
        return 1e-6 * self.frequency

    @property
    def baseband_frequency(self):
        if self._baseband_frequency is None:
            self._baseband_frequency = calculate.baseband_frequency(self.roach_state, self.tone_bin[self.tone_index])
        return self._baseband_frequency

    @property
    def baseband_frequency_MHz(self):
        return 1e-6 * self.baseband_frequency

    @property
    def stream_sample_rate(self):
        if self._stream_sample_rate is None:
            self._stream_sample_rate = calculate.stream_sample_rate(self.roach_state)
        return self._stream_sample_rate

    @property
    def s21_mean(self):
        if self._s21_mean is None:
            self._s21_mean = self.s21.mean(axis=1)
        return self._s21_mean

    @property
    def s21_mean_error(self):
        if self._s21_mean_error is None:
            self._s21_mean_error = ((self.s21.real.std(axis=1) + 1j * self.s21.imag.std(axis=1)) /
                                    self.s21.shape[1] ** (1 / 2))
        return self._s21_mean_error

    def __getitem__(self, key):
        """
        Return a StreamArray containing only the data corresponding to the times given in the slice. If no start (stop)
        time is given, the value is taken to be -inf (+inf). The returned StreamArray has the same state.

        The indexing follows the Python convention that the first value is inclusive and the second is exclusive:
        start <= epoch < stop
        Thus, the two slices streamarray[t0:t1] and streamarray[t1:t2] will contain all the data occurring at or after
        t0 and before t2, with no duplication. This means that
        streamarray[streamarray.epoch.min():streamarray.epoch.max()]
        will include all but the last sample.

        Passing a slice with a non-unity step size is not implemented and will raise a ValueError.
        """
        if isinstance(key, slice):
            if key.start is None:
                start = -np.inf
            else:
                start = key.start
            if key.stop is None:
                stop = np.inf
            else:
                stop = key.stop
            if key.step is not None:
                raise ValueError("Step size is not supported: {}".format(key))
            start_index = np.searchsorted(self.epoch, (start,), side='left')
            stop_index = np.searchsorted(self.epoch, (stop,), side='right')  # This index is not included
            return StreamArray(tone_bin=self.tone_bin, tone_amplitude=self.tone_amplitude, tone_phase=self.tone_phase,
                               tone_index=self.tone_index, filterbank_bin=self.filterbank_bin,
                               epoch=self.epoch[start_index:stop_index], s21=self.s21[:, start_index:stop_index],
                               roach_state=self.state, description=self.description)
        else:
            raise ValueError("Invalid slice: {}".format(key))

    def stream(self, tone_index):
        """
        Return a Stream object containing the data at the frequency corresponding to the given integer tone_index.
        """
        if isinstance(tone_index, int):
            return Stream(tone_bin=self.tone_bin, tone_amplitude=self.tone_amplitude, tone_phase=self.tone_phase,
                          tone_index=self.tone_index[tone_index], filterbank_bin=self.filterbank_bin[tone_index],
                          epoch=self.epoch, s21=self.s21[tone_index, :], roach_state=self.roach_state, state=self.state)
        else:
            raise ValueError("Invalid tone index: {}".format(tone_index))


class SweepArray(core.Measurement):
    """
    This class contains a group of stream arrays.
    """

    def __init__(self, stream_arrays, state=None, analyze=False, description='SweepArray'):
        self.stream_arrays = core.MeasurementTuple(stream_arrays)
        for sa in self.stream_arrays:
            sa._parent = self
        super(SweepArray, self).__init__(state=state, analyze=analyze, description=description)

    def sweep(self, index):
        if isinstance(index, int):
            return Sweep(streams=(sa.stream(index) for sa in self.stream_arrays), state=self.state)
        else:
            raise ValueError("Invalid index: {}".format(index))

    @property
    def num_channels(self):
        try:
            if np.any(np.diff([sa.tone_index.size for sa in self.stream_arrays])):
                raise ValueError("Channel numbers differ between stream arrays.")
            else:
                return self.stream_arrays[0].tone_index.size
        except IndexError:
            return 0


class ResonatorSweepArray(SweepArray):
    """
    This class represents a set of groups of streams.
    """

    def __init__(self, stream_arrays, state=None, analyze=False, description='ResonatorSweepArray'):
        super(ResonatorSweepArray, self).__init__(stream_arrays=stream_arrays, state=state, analyze=analyze,
                                                  description=description)

    def sweep(self, index):
        if isinstance(index, int):
            return ResonatorSweep((sa.stream(index) for sa in self.stream_arrays), state=self.state)
        else:
            raise ValueError("Invalid index: {}".format(index))


class SweepStreamArray(core.Measurement):

    def __init__(self, sweep_array, stream_array, state=None, analyze=False, description='SweepStreamArray'):
        if sweep_array.num_channels != stream_array.tone_index.size:
            raise core.MeasurementError("The number of SweepArray channels does not match the StreamArray number.")
        self.sweep_array = sweep_array
        self.sweep_array._parent = self
        self.stream_array = stream_array
        self.stream_array._parent = self
        super(SweepStreamArray, self).__init__(state=state, analyze=analyze, description=description)

    def analyze(self):
        self.sweep_array.analyze()
        self.stream_array.analyze()

    @property
    def num_channels(self):
        return self.sweep_array.num_channels

    def sweep_stream(self, index):
        """
        Return a SweepStream object containing the data at the frequency corresponding to the given integer index.
        """
        if isinstance(index, int):
            return SweepStream(sweep=self.sweep_array.sweep(index), stream=self.stream_array.stream(index),
                               state=self.state)
        else:
            raise ValueError("Invalid index: {}".format(index))

    def to_dataframe(self):
        dataframes = []
        for n in range(self.num_channels):
            dataframes.append(self.sweep_stream(n).to_dataframe())
        return pd.concat(dataframes, ignore_index=True)


# Functions for generating fake measurements.


# TODO: broken!
def make_stream_array(mean=0, rms=1, length=1, t0=0, active_state_arrays=None, roach_state=None, state=None):
    variables = {}
    if active_state_arrays is None:
        active_state_arrays = fake.active_state_arrays()
    if roach_state is None:
        roach_state = fake.state()
    variables.update(active_state_arrays)
    num_samples = length * calculate.stream_sample_rate(roach_state)
    variables['s21'] = mean + rms * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    variables['epoch'] = np.linspace(t0, t0 + length, num_samples)
    variables['roach_state'] = roach_state
    variables['state'] = state
    return StreamArray(**variables)
