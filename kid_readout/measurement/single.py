"""
This module contains classes that represent single-channel measurements.
"""
from __future__ import division
from collections import OrderedDict
import numpy as np
from matplotlib.pyplot import mlab  # TODO: replace with a scipy PSD estimator
import pandas as pd
from kid_readout.analysis.resonator import resonator
from kid_readout.analysis.timedomain.despike import deglitch_window
from kid_readout.measurement.core import Measurement, MeasurementTuple


class Stream(Measurement):
    """
    This class represents time-ordered data from a single channel.
    """

    dimensions = OrderedDict([('epoch', ('epoch',)),
                              ('s21', ('epoch',))])

    def __init__(self, frequency, epoch, s21, state=None, analyze=False, description='Stream'):
        self.frequency = frequency
        self.epoch = epoch
        self.s21 = s21
        self._s21_mean = None
        self._s21_mean_error = None
        super(Stream, self).__init__(state, analyze, description)

    def analyze(self):
        self.s21_mean
        self.s21_mean_error

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
            start_index = np.searchsorted(self.epoch, (start,))
            stop_index = np.searchsorted(self.epoch, (stop,), side='right')
            mask = (start <= self.epoch) & (self.epoch < stop)
            return Stream(self.frequency, self.epoch[start_index:stop_index], self.s21[:, start_index:stop_index],
                          self.state)
        else:
            raise ValueError("Invalid slice: {}".format(key))


class Sweep(Measurement):
    """
    This class represents a group of streams with different frequencies.
    """

    def __init__(self, streams=(), state=None, analyze=False, description='Sweep'):
        # Don't sort by frequency so that non-monotonic order can be preserved if needed, but note that this will fail
        # for a ResonatorSweep because the Resonator class requires a monotonic frequency array.
        self.streams = MeasurementTuple(streams)
        for stream in self.streams:
            stream._parent = self
        self._frequency = None
        self._s21 = None
        self._s21_error = None
        self._s21_raw = None
        super(Sweep, self).__init__(state, analyze, description)

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
    def __init__(self, streams=(), state=None, analyze=False, description='ResonatorSweep'):
        self._resonator = None
        super(ResonatorSweep, self).__init__(streams, state, analyze, description)

    def analyze(self):
        self.resonator

    def fit_resonator(self,delay_estimate=None,nonlinear_a_threshold=0.08):
        self._resonator = resonator.fit_best_resonator(self.frequency, self.s21, errors=self.s21_error,
                                                       delay_estimate=delay_estimate, min_a=nonlinear_a_threshold)

    @property
    def resonator(self):
        if self._resonator is None:
            self.fit_resonator()
        return self._resonator


# Think of another name for this. This class is intended to fit the gain and delay off-resonance.
class ThroughSweep(Sweep):
    def __init__(self, streams=(), state=None, analyze=False, description='ThroughSweep'):
        self._through = None
        super(ThroughSweep, self).__init__(streams, state, analyze, description)

    @property
    def through(self):
        return None


class SweepStream(Measurement):
    def __init__(self, sweep=None, stream=None, state=None, analyze=False, description='SweepStream'):
        self.sweep = sweep
        if self.sweep is not None:
            self.sweep._parent = self
        self.stream = stream
        if self.stream is not None:
            self.stream._parent = self
        self._sweep_s21_normalized = None
        self._stream_s21_normalized = None
        self._stream_s21_normalized_deglitched = None
        self._i = None
        self._x = None
        self._psd_frequency = None
        self._psd_ii = None
        self._psd_xx = None
        super(SweepStream, self).__init__(state, analyze, description)

    def analyze(self):
        self._set_sweep_s21_normalized()
        self._set_stream_s21_normalized_deglitched()
        self._set_i_and_x()
        self._set_psd_i_and_x()

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
    def i(self):
        if self._i is None:
            self._set_i_and_x()
        return self._i

    @property
    def x(self):
        if self._x is None:
            self._set_i_and_x()
        return self._x

    def _set_i_and_x(self, deglitch=True):
        if deglitch:
            s21 = self.stream_s21_normalized_deglitched
        else:
            s21 = self.stream_s21_normalized
        iQ_e = 1 / self.sweep.resonator.Q_e
        z = iQ_e / (1 - s21)
        self._i = 0.5 * (z.real - iQ_e.real)
        self._x = 1 / 2 * z.imag

    @property
    def psd_frequency(self):
        if self._psd_frequency is None:
            self._set_psd_i_and_x()
        return self._psd_frequency

    @property
    def psd_ii(self):
        if self._psd_ii is None:
            self._set_psd_i_and_x()
        return self._psd_ii

    @property
    def psd_xx(self):
        if self._psd_xx is None:
            self._set_psd_i_and_x()
        return self._psd_xx

    # TODO: calculate errors in PSDs
    def _set_psd_i_and_x(self, NFFT=None, window=mlab.window_none, **kwargs):
        # Use the same length calculation as SweepNoiseMeasurement
        if NFFT is None:
            NFFT = int(2**(np.floor(np.log2(self.stream.s21.size)) - 3))
        psd_ii, f = mlab.psd(self.i, Fs=self.stream.sample_frequency, NFFT=NFFT, window=window, **kwargs)
        psd_xx, f = mlab.psd(self.x, Fs=self.stream.sample_frequency, NFFT=NFFT, window=window, **kwargs)
        self._psd_frequency = f
        self._psd_ii = psd_ii
        self._psd_xx = psd_xx

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
