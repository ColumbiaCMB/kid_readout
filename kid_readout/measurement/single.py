"""
This module has classes that contain single-channel measurements.
"""
from __future__ import division
from collections import OrderedDict
import numpy as np
from matplotlib.pyplot import mlab  # TODO: replace with a scipy PSD estimator
import pandas as pd
from kid_readout.roach import calculate, temp_state
from kid_readout.analysis.resonator import resonator
from kid_readout.analysis.timedomain.despike import deglitch_window
from kid_readout.measurement import core


class Stream(core.Measurement):
    """
    This class contains time-ordered data from a single channel.
    """

    dimensions = OrderedDict([('tone_bin', ('tone_bin',)),
                              ('amplitude', ('tone_bin',)),
                              ('phase', ('tone_bin',)),
                              ('epoch', ('epoch',)),
                              ('s21', ('epoch',))])

    def __init__(self, tone_bin, amplitude, phase, tone_index, fft_bin, epoch, s21, state, analyze=False,
                 description='Stream'):
        """
        Return a new Stream instance. The integer tone_index is the common index of tone_bin, amplitude, and phase for
        the single tone used to produce the time-ordered s21 data.

        :param tone_bin: an array of integers representing the frequencies of the tones played during the measurement.
        :param amplitude: an array of floats representing the amplitudes of the tones played during the measurement.
        :param phase: an array of floats representing the radian phases of the tones played during the measurement.
        :param tone_index: an int for which tone_bin[tone_index] corresponds to the frequency used to produce s21.
        :param fft_bin: an int that is the fft bin in which the tone lies.
        :param epoch: an array of floats representing the unix timestamp when the data was recorded.
        :param s21: an 1-D array of complex floats containing the demodulated data.
        :param state: a dict containing state information for the roach and other hardware.
        :param analyze: if True, call the analyze() method at the end of instantiation.
        :param description: a string describing this measurement.
        :return: a new Stream instance.
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
        super(Stream, self).__init__(state=state, analyze=analyze, description=description)

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
            self._baseband_frequency = calculate.baseband_frequency(self.state.roach,
                                                                          self.tone_bin[self.tone_index])
        return self._baseband_frequency

    @property
    def baseband_frequency_MHz(self):
        return 1e-6 * self.baseband_frequency

    @property
    def output_sample_rate(self):
        if self._output_sample_rate is None:
            self._output_sample_rate = calculate.output_sample_rate(self.state.roach)
        return self._output_sample_rate

    @property
    def s21_mean(self):
        if self._s21_mean is None:
            self._s21_mean = self.s21.mean()
        return self._s21_mean

    @property
    def s21_mean_error(self):
        if self._s21_mean_error is None:
            self._s21_mean_error = (self.s21.real.std() + 1j * self.s21.imag.std()) / self.s21.size ** (1 / 2)
        return self._s21_mean_error

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
            start_index = np.searchsorted(self.epoch, (start,), side='left')
            stop_index = np.searchsorted(self.epoch, (stop,), side='right')  # This index is not included
            return Stream(tone_bin=self.tone_bin, amplitude=self.amplitude, phase=self.phase,
                          tone_index=self.tone_index, fft_bin=self.fft_bin,
                          epoch=self.epoch[start_index:stop_index], s21=self.s21[:, start_index:stop_index],
                          state=self.state, description=self.description)
        else:
            raise ValueError("Invalid slice: {}".format(key))


class Sweep(core.Measurement):
    """
    This class represents a group of streams with different frequencies.
    """

    def __init__(self, streams, state, analyze=False, description='Sweep'):
        # Don't sort by frequency so that non-monotonic order can be preserved if needed, but note that this will fail
        # for a ResonatorSweep because the Resonator class requires a monotonic frequency array.
        self.streams = core.MeasurementTuple(streams)
        for stream in self.streams:
            stream._parent = self
        self._frequency = None
        self._s21 = None
        self._s21_error = None
        self._s21_raw = None
        super(Sweep, self).__init__(state=state, analyze=analyze, description=description)

    @property
    def frequency(self):
        if self._frequency is None:
            self._frequency = np.array([stream.frequency for stream in self.streams])
        return self._frequency

    @property
    def s21(self):
        if self._s21 is None:
            self._s21 = np.array([stream.s21_mean for stream in self.streams])
        return self._s21

    @property
    def s21_error(self):
        if self._s21_error is None:
            self._s21_error = np.array([stream.s21_mean_error for stream in self.streams])
        return self._s21_error

    @property
    def s21_raw(self):
        if self._s21_raw is None:
            self._s21_raw = np.vstack([stream.s21 for stream in self.streams])
        return self._s21_raw

    # TODO: add s21 with delay removal
    # TODO: add psd


class ResonatorSweep(Sweep):
    def __init__(self, streams, state, analyze=False, description='ResonatorSweep'):
        self._resonator = None
        super(ResonatorSweep, self).__init__(streams=streams, state=state, analyze=analyze, description=description)

    def analyze(self):
        self.resonator

    def fit_resonator(self, delay_estimate=None, nonlinear_a_threshold=0.08):
        self._resonator = resonator.fit_best_resonator(self.frequency, self.s21, errors=self.s21_error,
                                                       delay_estimate=delay_estimate, min_a=nonlinear_a_threshold)

    @property
    def resonator(self):
        if self._resonator is None:
            self.fit_resonator()
        return self._resonator


# Think of another name for this. This class is intended to fit the gain and delay off-resonance.
class ThroughSweep(Sweep):
    def __init__(self, streams, state, analyze=False, description='ThroughSweep'):
        self._through = None
        super(ThroughSweep, self).__init__(streams, state, analyze=analyze, description=description)

    @property
    def through(self):
        return None


class SweepStream(core.Measurement):
    def __init__(self, sweep, stream, state, analyze=False, description='SweepStream'):
        self.sweep = sweep
        self.sweep._parent = self
        self.stream = stream
        self.stream._parent = self
        self._sweep_s21_normalized = None
        self._stream_s21_normalized = None
        self._stream_s21_normalized_deglitched = None
        self._q = None
        self._x = None
        self._S_frequency = None
        self._S_qq = None
        self._S_xx = None
        super(SweepStream, self).__init__(state=state, analyze=analyze, description=description)

    def analyze(self):
        self._set_sweep_s21_normalized()
        self._set_stream_s21_normalized_deglitched()
        self._set_q_and_x()
        self._set_S_qq_and_S_xx()

    @property
    def sweep_s21_normalized(self):
        if self._sweep_s21_normalized is None:
            self._set_sweep_s21_normalized()
        return self._sweep_s21_normalized

    def _set_sweep_s21_normalized(self):
        self._sweep_s21_normalized = np.array([self.sweep.resonator.normalize(f, s21)
                                               for f, s21 in zip(self.sweep.frequency, self.sweep.s21)])

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
        self._x = z.imag / 2  # This factor of two means S_xx = S_qq / 4 when amplifier-noise dominated.

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

    # TODO: move this forward to a usable version.
    def to_dataframe(self):
        data = {}
        try:
            for thermometer, temperature in self.state['temperature'].items():
                data['temperature_{}'.format(thermometer)] = temperature
        except KeyError:
            pass
        try:
            for key, value in self.state['roach'].items():
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


# Functions for generating fake measurements.


def make_stream(tone_index=0, mean=0, rms=1, length=1, t0=0, roach_state=None):
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
    variables['s21'] = mean + rms * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    variables['epoch'] = np.linspace(t0, t0 + length, num_samples)
    return core.instantiate(full_class_name=__name__ + '.' + 'Stream', variables=variables, extras=False)
