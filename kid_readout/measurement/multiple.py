"""
This module has classes that contain simultaneous multiple-channel measurements.
"""
from __future__ import division
from collections import OrderedDict
import numpy as np
import pandas as pd
from kid_readout.measurement import core
from kid_readout.measurement.single import Stream, Sweep, ResonatorSweep, SweepStream
from kid_readout.roach import calculate, temp_state


class StreamArray(core.Measurement):
    """
    This class represents simultaneously-sampled data from multiple channels.
    """

    dimensions = OrderedDict([('tone_bin', ('tone_bin',)),
                              ('amplitude', ('tone_bin',)),
                              ('phase', ('tone_bin',)),
                              ('fft_bin', ('tone_bin',)),
                              ('tone_index', ('tone_index',)),
                              ('epoch', ('epoch',)),
                              ('s21', ('tone_index', 'epoch'))])

    def __init__(self, tone_bin, amplitude, phase, tone_index, fft_bin, epoch, s21, state, analyze=False,
                 description='StreamArray'):
        """
        Return a new Stream instance. The integer array tone_index contains the indices of tone_bin, amplitude, and phase
        for the tones demodulated to produce the time-ordered s21 data.

        The tone_bin, amplitude, phase, tone_index, fft_bin, and epoch arrays are 1-D, while s21 is 2-D. The arrays must
        obey s21.shape == (tone_index.size, epoch.size)

        :param tone_bin: an array of integers representing the frequencies of the tones played during the measurement.
        :param amplitude: an array of floats representing the amplitudes of the tones played during the measurement.
        :param phase: an array of floats representing the radian phases of the tones played during the measurement.
        :param tone_index: an intarray for which tone_bin[tone_index] corresponds to the frequency used to produce s21.
        :param fft_bin: an integer that is the fft bin in which the tone lies.
        :param epoch: an array of floats representing the unix timestamp when the data was recorded.
        :param s21: a 2-D array of complex floats containing the demodulated data.
        :param state: a dict containing state information for the roach and other hardware.
        :param analyze: if True, call the analyze() method at the end of instantiation.
        :param description: a string describing this measurement.
        :return: a new StreamArray instance.
        """
        self.tone_bin = tone_bin
        self.amplitude = amplitude
        self.phase = phase
        self.tone_index = tone_index
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
            self._frequency = calculate.frequency(self.state.roach, self.tone_bin[self.tone_index])
        return self._frequency

    @property
    def frequency_MHz(self):
        return 1e-6 * self.frequency

    @property
    def baseband_frequency(self):
        if self._baseband_frequency is None:
            self._baseband_frequency = calculate.baseband_frequency(self.state.roach, self.tone_bin[self.tone_index])
        return self._baseband_frequency

    @property
    def baseband_frequency_MHz(self):
        return 1e-6 * self.baseband_frequency

    @property
    def output_sample_rate(self):
        if self._output_sample_rate is None:
            self._output_sample_rate = calculate.audio_sample_rate(self.state.roach)
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
                               tone_index=self.tone_index, fft_bin=self.fft_bin,
                               epoch=self.epoch[start_index:stop_index], s21=self.s21[:, start_index:stop_index],
                               state=self.state, description=self.description)
        else:
            raise ValueError("Invalid slice: {}".format(key))

    def stream(self, index):
        """
        Return a Stream object containing the data at the frequency corresponding to the given integer index.
        """
        if isinstance(index, int):
            return Stream(tone_bin=self.tone_bin, amplitude=self.amplitude, phase=self.phase,
                          tone_index=self.tone_index[index], fft_bin=self.fft_bin[index], epoch=self.epoch,
                          s21=self.s21[index, :], state=self.state)
        else:
            raise ValueError("Invalid index: {}".format(index))


class SweepArray(core.Measurement):
    """
    This class contains a group of stream arrays.
    """

    def __init__(self, stream_arrays, state, analyze=False, description='SweepArray'):
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

    def __init__(self, stream_arrays, state, analyze=False, description='ResonatorSweepArray'):
        super(ResonatorSweepArray, self).__init__(stream_arrays=stream_arrays, state=state, analyze=analyze,
                                                  description=description)

    def sweep(self, index):
        if isinstance(index, int):
            return ResonatorSweep((sa.stream(index) for sa in self.stream_arrays), state=self.state)
        else:
            raise ValueError("Invalid index: {}".format(index))


class SweepStreamArray(core.Measurement):

    def __init__(self, sweep_array, stream_array, state, analyze=False, description='SweepStreamArray'):
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


def make_stream_array(tone_index=np.arange(16), mean=np.zeros(16), rms=np.ones(16), length=1, t0=0, roach_state=None):
    variables = {'state': {}}
    if roach_state is None:
        roach_array, roach_other = temp_state.fake_baseband()
    else:
        roach_array, roach_other = roach_state
    variables['tone_bin'] = roach_array['tone_bin']
    variables['amplitude'] = roach_array['amplitude']
    variables['phase'] = roach_array['phase']
    variables['tone_index'] = tone_index
    variables['fft_bin'] = roach_array['fft_bin'][tone_index]
    variables['state']['roach'] = roach_other
    num_samples = length * calculate.audio_sample_rate(roach_other)
    variables['s21'] = mean[:, np.newaxis] + rms[:, np.newaxis] * (np.random.randn(tone_index.size, num_samples) +
                                                                   1j * np.random.randn(tone_index.size, num_samples))
    variables['epoch'] = np.linspace(t0, t0 + length, num_samples)
    return core.instantiate(full_class_name=__name__ + '.' + 'StreamArray', variables=variables, extras=False)
