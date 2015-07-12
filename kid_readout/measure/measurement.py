from __future__ import division
from collections import OrderedDict
import numpy as np
# TODO: replace with a scipy PSD estimator
import matplotlib.pyplot as plt
import pandas as pd
from kid_readout.measure.nc import Measurement
from kid_readout.measure import noise
from kid_readout.analysis import resonator, fitter
from kid_readout.analysis import noise_fit
from kid_readout.utils.despike import deglitch_window


class Stream(Measurement):
    """
    This class represents time-ordered data from a single channel.
    """

    netcdf_dimensions = OrderedDict(s21=None)

    netcdf_variables = OrderedDict(frequency=Measurement.Variable(netcdf_name='frequency',
                                                                  data_type=np.float64,
                                                                  dimensions=()),
                                   s21=Measurement.Variable(netcdf_name='s21',
                                                            data_type=np.complex64,
                                                            dimensions=('s21',)),
                                   start_epoch=Measurement.Variable(netcdf_name='start_epoch',
                                                                    data_type=np.float64,
                                                                    dimensions=()),
                                   end_epoch=Measurement.Variable(netcdf_name='end_epoch',
                                                                  data_type=np.float64,
                                                                  dimensions=()))

    def __init__(self, frequency=0, s21=np.array([1]), start_epoch=0, end_epoch=1, state=None, analyze=False):
        self.frequency = float(frequency)
        self.s21 = np.array(s21)
        self.start_epoch = float(start_epoch)
        self.end_epoch = float(end_epoch)
        self._s21_mean = None
        self._s21_mean_error = None
        self._epoch = None
        self._sample_frequency = None
        super(Stream, self).__init__(state, analyze)

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

    @property
    def epoch(self):
        """
        Return an array with the same size as the data containing the epochs, assuming that the data sample rate is
        constant.
        """
        if self._epoch is None:
            self._epoch = np.linspace(self.start_epoch, self.end_epoch, self.s21.size)
        return self._epoch

    @property
    def sample_frequency(self):
        if self._sample_frequency is None:
            self._sample_frequency = self.s21.size / (self.end_epoch - self.start_epoch)
        return self._sample_frequency


class FrequencySweep(Measurement):
    """
    This class represents a set of streams.
    """

    # If we save no derived data, then we can't use these in the file.
    """
    netcdf_dimensions = OrderedDict(frequency=None)
    netcdf_variables = {'frequency': Measurement.Variable(netcdf_name='frequency',
                                                          data_type=np.float64,
                                                          dimensions=('frequency',)),
                        's21': Measurement.Variable(netcdf_name='s21',
                                                    data_type=np.complex64,
                                                    dimensions=('frequency',))}
    """

    def __init__(self, streams=(), state=None, analyze=False):
        # Don't sort by frequency so that non-monotonic order can be preserved if needed.
        # self.streams = tuple(sorted(streams, key=lambda s: s.frequency))
        self.streams = tuple(streams)
        for stream in self.streams:
            stream._parent = self
        self._frequency = None
        self._s21 = None
        self._s21_error = None
        super(FrequencySweep, self).__init__(state, analyze)

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

    def to_group(self, group, numpy_to_netcdf):
        super(FrequencySweep, self).to_group(group, numpy_to_netcdf)
        streams_group = group.createGroup('streams')
        for n, stream in enumerate(self.streams):
            stream_group = streams_group.createGroup("Stream_{}".format(n))
            stream.to_group(stream_group, numpy_to_netcdf)

    def from_group(self, group):
        super(FrequencySweep, self).from_group(group)
        stream_list = []
        for stream_group in group.groups['streams'].groups.values():
            stream = Stream().from_group(stream_group)
            stream._parent = self
            stream_list.append(stream)
        self.streams = tuple(stream_list)
        return self


class ResonatorSweep(FrequencySweep):
    def __init__(self, streams=(), state=None, analyze=False):
        self._resonator = None
        super(ResonatorSweep, self).__init__(streams, state, analyze)

    def analyze(self):
        self.resonator

    @property
    def resonator(self):
        if self._resonator is None:
            self._resonator = resonator.fit_best_resonator(self.frequency, self.s21, errors=self.s21_error)
        return self._resonator


# Think of another name for this. This class is intended to fit the gain and delay off-resonance.
class ThroughSweep(FrequencySweep):
    def __init__(self, streams=(), state=None, analyze=False):
        self._through = None
        super(ThroughSweep, self).__init__(streams, state, analyze)

    @property
    def through(self):
        return None


class SweepStream(Measurement):
    def __init__(self, sweep=None, stream=None, state=None, analyze=False):
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
        self._noise = None
        super(SweepStream, self).__init__(state, analyze)

    def analyze(self):
        self._set_sweep_s21_normalized()
        self._set_stream_s21_normalized_deglitched()
        self._set_i_and_x()
        self._set_psd_i_and_x()
        self._fit_noise_i_and_x()

    def to_group(self, group, numpy_to_netcdf):
        super(SweepStream, self).to_group(group, numpy_to_netcdf)
        sweep_group = group.createGroup('sweep')
        self.sweep.to_group(sweep_group, numpy_to_netcdf)
        stream_group = group.createGroup('stream')
        self.stream.to_group(stream_group, numpy_to_netcdf)

    def from_group(self, group):
        super(SweepStream, self).from_group(group)
        self.sweep = ResonatorSweep().from_group(group.groups['sweep'])
        self.sweep._parent = self
        self.stream = Stream().from_group(group.groups['stream'])
        self.sweep._parent = self
        return self

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

    def _set_i_and_x(self):
        iQ_e = 1 / self.sweep.resonator.Q_e
        z = iQ_e / (1 - self.stream_s21_normalized_deglitched)
        self._i = z.real - iQ_e.real
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

    def _set_psd_i_and_x(self, NFFT=None, window=plt.mlab.window_none, **kwargs):
        # Use the same length calculation as SweepNoiseMeasurement
        if NFFT is None:
            NFFT = int(2**(np.floor(np.log2(self.stream.s21.size)) - 3))
        psd_ii, f = plt.mlab.psd(self.i, Fs=self.stream.sample_frequency, NFFT=NFFT, window=window, **kwargs)
        psd_xx, f = plt.mlab.psd(self.x, Fs=self.stream.sample_frequency, NFFT=NFFT, window=window, **kwargs)
        self._psd_frequency = f
        self._psd_ii = psd_ii
        self._psd_xx = psd_xx

    def _fit_noise_i_and_x(self):
        self.noise_ii = noise_fit.fit_single_pole_noise(self.psd_frequency[1:], self.psd_ii[1:],
                                                        errors=1e-18*np.ones(self.psd_ii.size-1),
                                                        max_num_masked=8)
        self.noise_xx = noise_fit.fit_single_pole_noise(self.psd_frequency[1:], self.psd_xx[1:],
                                                        errors=1e-18*np.ones(self.psd_xx.size-1),
                                                        max_num_masked=8)

    # This isn't working yet.
    def _fit_full_noise_i_and_x(self):
        f_r = (1e6 * self.sweep.resonator.f_0) / (2 * self.sweep.resonator.Q)
        f_qp = 10 * self.psd_frequency.max()

        guess_ii = noise.bandwidth_limited_guess(self.psd_frequency[1:], self.psd_ii[1:], f_r)
        guess_ii['f_r'].vary = False
        guess_ii['f_qp'].vary = False
        guess_ii['S_TLS'].value = 0
        guess_ii['S_TLS'].vary = False
        self.full_noise_ii = fitter.Fitter(self.psd_frequency[1:], self.psd_ii[1:], model=noise.model,
                                           guess=lambda f, S: guess_ii,
                                           errors=1e-18*np.ones(self.psd_frequency.size - 1),
                                           xtol=1e-12)

        guess_xx = noise.bandwidth_limited_guess(self.psd_frequency[1:], self.psd_xx[1:], f_r)
        guess_xx['f_r'].vary = False
        guess_xx['f_qp'].vary = False
        self.full_noise_xx = fitter.Fitter(self.psd_frequency[1:], self.psd_xx[1:], model=noise.model,
                                           guess=lambda f, S: guess_xx,
                                           errors=1e-18*np.ones(self.psd_frequency.size - 1),
                                           xtol=1e-12)

    # TODO: move this forward to a usable version.
    def to_dataframe(self):
        data = {}
        for param in self.sweep.resonator.result.params.values():
            data['resonator_{}'.format(param.name)] = [param.value]
            data['resonator_{}_error'.format(param.name)] = [param.stderr]
        data['resonator_redchi'] = self.sweep.resonator.result.redchi
        for param in self.noise_ii.result.params.values():
            data['noise_ii_{}'.format(param.name)] = [param.value]
            data['noise_ii_{}_error'.format(param.name)] = [param.stderr]
        data['noise_ii_redchi'] = self.noise_ii.result.redchi
        for param in self.noise_xx.result.params.values():
            data['noise_xx_{}'.format(param.name)] = [param.value]
            data['noise_xx_{}_error'.format(param.name)] = [param.stderr]
        data['noise_xx_redchi'] = self.noise_xx.result.redchi
        # temperatures
        # roach state
        return pd.DataFrame(data, index=[0])

