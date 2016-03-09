"""
This module contains classes that represent simultaneous multiple-channel measurements.
"""
from __future__ import division
from collections import OrderedDict
import numpy as np
import pandas as pd
from kid_readout.measurement.core import Measurement, MeasurementTuple, MeasurementError
from kid_readout.measurement.single import Stream, Sweep, ResonatorSweep, SweepStream
from kid_readout.roach import calculate


class StreamArray(Measurement):
    """
    This class represents simultaneously-sampled data from multiple channels.
    """

    dimensions = OrderedDict([('tone_bin', ('tone_bin',)),
                              ('amplitude', ('tone_bin',)),
                              ('phase', ('tone_bin',)),
                              ('fft_bin', ('tone_bin',)),
                              ('epoch', ('epoch',)),
                              ('s21', ('tone_bin', 'epoch'))])

    def __init__(self, tone_bin, amplitude, phase, fft_bin, epoch, s21, state, analyze=False,
                 description='StreamArray'):
        """
        The tone_bin, amplitude, phase, fft_bin, and epoch arrays are 1-D, while s21 is 2-D. The arrays must obey
        s21.shape == (tone_bin.size, epoch.size)
        """
        self.tone_bin = tone_bin
        self.amplitude = amplitude
        self.phase = phase
        self.fft_bin = fft_bin
        self.epoch = epoch
        self.s21 = s21
        self._frequency = None
        self._baseband_frequency = None
        self._output_sample_rate = None
        self._s21_mean = None
        self._s21_mean_error = None
        super(StreamArray, self).__init__(state=state, analyze=analyze, description=description)

    def analyze(self):
        self.baseband_frequency
        self.frequency
        self.output_sample_rate
        self.s21_mean
        self.s21_mean_error

    @property
    def frequency(self):
        if self._frequency is None:
            self._frequency = calculate.frequency_MHz(self.state.roach, self.tone_bin)
        return self._frequency

    @property
    def baseband_frequency(self):
        if self._baseband_frequency is None:
            self._baseband_frequency = calculate.baseband_frequency_MHz(self.state.roach, self.tone_bin)
        return self._baseband_frequency

    @property
    def output_sample_rate(self):
        if self._output_sample_rate is None:
            self._output_sample_rate = calculate.output_sample_rate(self.state.roach)
        return self._output_sample_rate

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
            return StreamArray(tone_bin=self.tone_bin, amplitude=self.amplitude, phase=self.phase,
                               fft_bin=self.fft_bin, epoch=self.epoch[start_index:stop_index],
                               s21=self.s21[:, start_index:stop_index], state=self.state, description=self.description)
        else:
            raise ValueError("Invalid slice: {}".format(key))

    def stream(self, index):
        """
        Return a Stream object containing the data at the frequency corresponding to the given integer index.
        """
        if isinstance(index, int):
            return Stream(tone_bin=self.tone_bin[index], amplitude=self.amplitude[index], phase=self.phase[index],
                          fft_bin=self.fft_bin[index], epoch=self.epoch, s21=self.s21[index, :], state=self.state)
        else:
            raise ValueError("Invalid index: {}".format(index))


class SweepArray(Measurement):
    """
    This class represents a set of groups of streams.
    """

    def __init__(self, stream_arrays, state, analyze=False, description='SweepArray'):
        self.stream_arrays = MeasurementTuple(stream_arrays)
        for sa in self.stream_arrays:
            sa._parent = self
        super(SweepArray, self).__init__(state=state, analyze=analyze, description=description)

    def sweep(self, index):
        if isinstance(index, int):
            return Sweep(streams=(sa.stream(index) for sa in self.stream_arrays), state=self.state)
        else:
            raise ValueError("Invalid index: {}".format(index))

    @property
    def n_channels(self):
        try:
            return self.stream_arrays[0].tone_bin.size
        except IndexError:
            return 0


class ResonatorSweepArray(SweepArray):
    """
    This class represents a set of groups of streams.
    """

    def __init__(self, stream_arrays, state, analyze=False, description='ResonatorSweepArray'):
        super(ResonatorSweepArray, self).__init__(stream_arrays=stream_arrays, state=state, analyze=analyze,
                                                  description=description)

    def sweep(self, index):
        if isinstance(index, int):
            return ResonatorSweep((sa.stream(index) for sa in self.stream_arrays), state=self.state)
        else:
            raise ValueError("Invalid index: {}".format(index))


class SweepStreamArray(Measurement):

    def __init__(self, sweep_array, stream_array, state, analyze=False, description='SweepStreamArray'):
        if sweep_array.n_channels != stream_array.tone_bin.size:
            raise MeasurementError("The number of SweepArray channels does not match the StreamArray number.")
        self.sweep_array = sweep_array
        self.sweep_array._parent = self
        self.stream_array = stream_array
        self.stream_array._parent = self
        super(SweepStreamArray, self).__init__(state=state, analyze=analyze, description=description)

    def analyze(self):
        pass

    @property
    def n_channels(self):
        return self.sweep_array.n_channels

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
        for n in range(self.n_channels):
            dataframes.append(self.sweep_stream(n).to_dataframe())
        return pd.concat(dataframes, ignore_index=True)

