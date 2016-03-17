"""
This module has classes that contain single-channel measurements.
"""
from __future__ import division

import numpy as np
import pandas as pd
from matplotlib.pyplot import mlab  # TODO: replace with a scipy PSD estimator

from kid_readout.analysis.resonator import resonator
from kid_readout.analysis.timedomain.despike import deglitch_window
from kid_readout.measurement import core
from kid_readout.roach import calculate


class Stream(core.Measurement):
    """
    This class contains time-ordered data from a single channel.
    """

    dimensions = {'tone_bin': ('tone_bin',),
                  'tone_amplitude': ('tone_bin',),
                  'tone_phase': ('tone_bin',),
                  's21_raw': ('sample_time',)}

    def __init__(self, tone_bin, tone_amplitude, tone_phase, tone_index, filterbank_bin, epoch, s21_raw, roach_state,
                 state=None, analyze=False, description='Stream'):
        """
        Return a new Stream instance. The integer tone_index is the common index of tone_bin, tone_amplitude,
        and tone_phase for
        the single tone used to produce the time-ordered s21_raw data.

        :param tone_bin: an array of integers representing the frequencies of the tones played during the measurement.
        :param tone_amplitude: an array of floats representing the amplitudes of the tones played during the
          measurement.
        :param tone_phase: an array of floats representing the radian phases of the tones played during the measurement.
        :param tone_index: an int for which tone_bin[tone_index] corresponds to the frequency used to produce s21_raw.
        :param filterbank_bin: an int that is the filter bank bin in which the tone lies.
        :param epoch: float, unix timestamp of first sample of the time stream.
        :param s21_raw: an 1-D array of complex floats containing the data, demodulated or not.
        :param roach_state: a dict containing state information for the roach.
        :param state: a dict containing all non-roach state information.
        :param analyze: if True, call the analyze() method at the end of instantiation.
        :param description: a string describing this measurement.
        :return: a new Stream instance.
        """
        self.tone_bin = tone_bin
        self.tone_amplitude = tone_amplitude
        self.tone_phase = tone_phase
        self.tone_index = tone_index
        self.filterbank_bin = filterbank_bin
        self.epoch = epoch
        self.s21_raw = s21_raw
        self.roach_state = core.to_state_dict(roach_state)
        self._frequency = None
        self._sample_time = None
        self._baseband_frequency = None
        self._stream_sample_rate = None
        self._s21_raw_mean = None
        self._s21_raw_mean_error = None
        super(Stream, self).__init__(state=state, analyze=analyze, description=description)

    def analyze(self):
        self.baseband_frequency
        self.frequency
        self.stream_sample_rate
        self.s21_raw_mean
        self.s21_raw_mean_error

    @property
    def sample_time(self):
        if self._sample_time is None:
            self._sample_time = (np.arange(self.s21_raw.shape[0], dtype='float') /
                                 self.stream_sample_rate)
        return self._sample_time

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
    def s21_point(self):
        return self.s21_raw_mean

    @property
    def s21_point_error(self):
        return self.s21_raw_mean_error

    @property
    def s21_raw_mean(self):
        if self._s21_raw_mean is None:
            self._s21_raw_mean = self.s21_raw.mean()
        return self._s21_raw_mean

    @property
    def s21_raw_mean_error(self):
        if self._s21_raw_mean_error is None:
            self._s21_raw_mean_error = ((self.s21_raw.real.std() +1j * self.s21_raw.imag.std()) /
                                        self.s21_raw.size ** (1 / 2))
        return self._s21_raw_mean_error

    def __getitem__(self, key):
        """
        Return a Stream containing only the data corresponding to the times given in the slice. If no start (stop) time
        is given, the value is taken to be -inf (+inf). The returned Stream has the same state.

        The indexing follows the Python convention that the first value is inclusive and the second is exclusive:
        start <= epoch < stop
        Thus, the two slices stream[t0:t1] and stream[t1:t2] will contain all the data occurring at or after t0 and
        before t2, with no duplication.  This means that
        streamarray[streamarray.epoch.min():streamarray.epoch.max()]
        will include all but the last sample.

        Passing a slice with a step size is not implemented and will raise a ValueError.
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
            start_index = np.searchsorted(self.sample_time, (start,), side='left')
            stop_index = np.searchsorted(self.sample_time, (stop,), side='right')  # This index is not included
            return Stream(tone_bin=self.tone_bin, tone_amplitude=self.tone_amplitude, tone_phase=self.tone_phase,
                          tone_index=self.tone_index, filterbank_bin=self.filterbank_bin,
                          epoch=self.sample_time[start_index:stop_index], s21_raw=self.s21_raw[:, start_index:stop_index],
                          roach_state=self.state, description=self.description)
        else:
            raise ValueError("Invalid slice: {}".format(key))


class Sweep(core.Measurement):
    """
    This class represents a group of streams with different frequencies.
    """

    def __init__(self, streams, state=None, analyze=False, description='Sweep'):
        # Don't sort by frequency so that non-monotonic order can be preserved if needed, but note that this will fail
        # for a ResonatorSweep because the Resonator class requires a monotonic frequency array.
        self.streams = core.MeasurementTuple(streams)
        for stream in self.streams:
            stream._parent = self
        self._frequency = None
        self._s21_points = None
        self._s21_points_error = None
        self._s21_raw_stack = None
        super(Sweep, self).__init__(state=state, analyze=analyze, description=description)

    @property
    def frequency(self):
        if self._frequency is None:
            self._frequency = np.array([stream.frequency for stream in self.streams])
        return self._frequency

    @property
    def s21_points(self):
        if self._s21_points is None:
            self._s21_points = np.array([stream.s21_point for stream in self.streams])
        return self._s21_points

    @property
    def s21_points_error(self):
        if self._s21_points_error is None:
            self._s21_points_error = np.array([stream.s21_point_error for stream in self.streams])
        return self._s21_points_error

    @property
    def s21_raw_stack(self):
        if self._s21_raw_stack is None:
            self._s21_raw_stack = np.vstack([stream.s21_raw for stream in self.streams])
        return self._s21_raw_stack


class ResonatorSweep(Sweep):
    def __init__(self, streams, state=None, analyze=False, description='ResonatorSweep'):
        self._resonator = None
        self._s21_normalized = None
        self._s21_normalized_error = None
        super(ResonatorSweep, self).__init__(streams=streams, state=state, analyze=analyze, description=description)

    def analyze(self):
        self.resonator

    @property
    def s21_normalized(self):
        if self._s21_normalized is None:
            self._set_s21_normalized()
        return self._s21_normalized

    def _set_s21_normalized(self):
        self._s21_normalized = np.array([self.resonator.normalize(f, s21)
                                         for f, s21 in zip(self.frequency, self.s21_points)])

    @property
    def s21_normalized_error(self):
        if self._s21_normalized_error is None:
            self._set_s21_normalized_error()
        return self._s21_normalized_error

    def _set_s21_normalized_error(self):
        self._s21_normalized_error = np.array([self.resonator.normalize(f, s21_error)
                                               for f, s21_error in zip(self.frequency, self.s21_points_error)])

    @property
    def resonator(self):
        if self._resonator is None:
            self.fit_resonator()
        return self._resonator

    def fit_resonator(self, delay_estimate=None, nonlinear_a_threshold=0.08):
        self._resonator = resonator.fit_best_resonator(self.frequency, self.s21_points, errors=self.s21_points_error,
                                                       delay_estimate=delay_estimate, min_a=nonlinear_a_threshold)


# Think of another name for this. This class is intended to fit the gain and delay off-resonance.
class ThroughSweep(Sweep):
    def __init__(self, streams, state=None, analyze=False, description='ThroughSweep'):
        self._through = None
        super(ThroughSweep, self).__init__(streams, state, analyze=analyze, description=description)

    @property
    def through(self):
        return None


class SweepStream(core.Measurement):
    def __init__(self, sweep, stream, state=None, analyze=False, description='SweepStream'):
        self.sweep = sweep
        self.sweep._parent = self
        self.stream = stream
        self.stream._parent = self
        self._stream_s21_normalized = None
        self._stream_s21_normalized_deglitched = None
        self._q = None
        self._x = None
        self._S_frequency = None
        self._S_qq = None
        self._S_xx = None
        super(SweepStream, self).__init__(state=state, analyze=analyze, description=description)

    def analyze(self):
        self._set_stream_s21_normalized_deglitched()
        self._set_q_and_x()
        self._set_S_qq_and_S_xx()

    @property
    def stream_s21_normalized(self):
        if self._stream_s21_normalized is None:
            self._stream_s21_normalized = self.sweep.resonator.normalize(self.stream.frequency, self.stream.s21)
        return self._stream_s21_normalized

    @property
    def stream_s21_normalized_deglitched(self):
        if self._stream_s21_normalized_deglitched is None:
            self._set_stream_s21_normalized_deglitched()
        return self._stream_s21_normalized_deglitched

    def _set_stream_s21_normalized_deglitched(self, window_in_seconds=1, deglitch_threshold=5):
        window = int(2 ** np.ceil(np.log2(window_in_seconds * self.stream.sample_frequency)))
        self._stream_s21_normalized_deglitched = deglitch_window(self.stream_s21_normalized, window,
                                                                 thresh=deglitch_threshold)
    
    @property
    def q(self):
        """
        Return the inverse internal quality factor q = 1 / Q_i calculated by inverting the resonator model.
        :return: an array of q values from self.stream corresponding to self.stream.epoch.
        """
        if self._q is None:
            self._set_q_and_x()
        return self._q

    @property
    def y(self):
        """
        Return half the inverse internal quality factor y = q / 2 calculated by inverting the resonator model. The
        purpose of this is that S_yy = S_xx when amplifier-noise dominated.
        :return: an array of y values from self.stream corresponding to self.stream.epoch.
        """
        return self.q / 2

    @property
    def x(self):
        """
        Return the fractional frequency shift x = f / f_r - 1 calculated by inverting the resonator model.
        :return: an array of x values from self.stream corresponding to self.stream.epoch.
        """
        if self._x is None:
            self._set_q_and_x()
        return self._x

    def _set_q_and_x(self, deglitch=True):
        if deglitch:
            s21 = self.stream_s21_normalized_deglitched
        else:
            s21 = self.stream_s21_normalized
        c = 1 / self.sweep.resonator.Q_e  # c is the inverse of the complex couping quality factor.
        z = c / (1 - s21)
        self._q = z.real - c.real
        self._x = z.imag / 2  # This factor of two means S_xx = S_qq / 4 when amplifier noise dominated.

    @property
    def S_frequency(self):
        """
        Return the frequencies used in calculating the single-sided spectral densities.
        :return: an array of frequencies ranging from 0 through the Nyquist frequency.
        """
        if self._S_frequency is None:
            self._set_S_qq_and_S_xx()
        return self._S_frequency

    @property
    def S_qq(self):
        """
        The single-sided spectral density of q(t), S_qq(f), where f is self.S_frequency.
        :return: an array of complex values representing the spectral density of q(t)
        """
        if self._S_qq is None:
            self._set_S_qq_and_S_xx()
        return self._S_qq

    @property
    def S_yy(self):
        """
        The single-sided spectral density of y(t), S_yy(f), where f is self.S_frequency.
        :return: an array of complex values representing the spectral density of y(t)
        """
        return self.S_qq / 4

    @property
    def S_xx(self):
        """
        The single-sided spectral density of x(t), S_xx(f), where f is self.S_frequency.
        :return: an array of complex values representing the spectral density of x(t)
        """
        if self._S_xx is None:
            self._set_S_qq_and_S_xx()
        return self._S_xx

    # TODO: calculate errors in PSDs
    def _set_S_qq_and_S_xx(self, NFFT=None, window=mlab.window_none, **kwargs):
        # Use the same length calculation as SweepNoiseMeasurement
        if NFFT is None:
            NFFT = int(2**(np.floor(np.log2(self.stream.s21.size)) - 3))
        S_qq, f = mlab.psd(self.q, Fs=self.stream.sample_frequency, NFFT=NFFT, window=window, **kwargs)
        S_xx, f = mlab.psd(self.x, Fs=self.stream.sample_frequency, NFFT=NFFT, window=window, **kwargs)
        self._S_frequency = f
        self._S_qq = S_qq
        self._S_xx = S_xx

    def to_dataframe(self):
        data = {}
        try:
            for thermometer, temperature in self.state['temperature'].items():
                data['temperature_{}'.format(thermometer)] = temperature
        except KeyError:
            pass
        try:
            # TODO: need to flatten sub-dicts, if any.
            for key, value in self.stream.roach_state.items():
                data['roach_{}'.format(key)] = value
        except KeyError:
            pass
        for param in self.sweep.resonator.result.params.values():
            data['resonator_{}'.format(param.name)] = param.value
            data['resonator_{}_error'.format(param.name)] = param.stderr
        data['resonator_redchi'] = self.sweep.resonator.result.redchi
        dataframe = pd.DataFrame(data, index=[0])
        self.add_origin(dataframe)
        return dataframe
