"""
This module contains basic measurement classes for data acquired with the ROACH.
"""
from __future__ import division
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
from matplotlib.pyplot import mlab
from memoized_property import memoized_property

from kid_readout.measurement import core
from kid_readout.analysis.resonator import lmfit_resonator
from kid_readout.analysis.timedomain.despike import deglitch_window
from kid_readout.analysis.timedomain import iqnoise
from kid_readout.roach import calculate


class RoachStream(core.Measurement):

    _version = 1

    def __init__(self, tone_bin, tone_amplitude, tone_phase, tone_index, filterbank_bin, epoch, sequence_start_number,
                 s21_raw, data_demodulated, roach_state, state=None, description=''):
        """
        Return a new RoachStream instance. This class has no dimensions and is intended to be subclassed.

        Parameters
        ----------
        tone_bin : numpy.ndarray(int)
            An array of integers representing the frequencies of the tones played during the measurement.
        tone_amplitude : numpy.ndarray(float)
            An array of floats representing the amplitudes of the tones played during the measurement.
        tone_phase : numpy.ndarray(float)
            An array of floats representing the radian phases of the tones played during the measurement.
        tone_index : int or numpy.ndarray(int)
            tone_bin[tone_index] corresponds to the frequency used to produce s21_raw.
        filterbank_bin : int or numpy.ndarray(int)
            The filter bank bin(s) containing the tone(s).
        epoch : float
            The unix timestamp of the first sample of the time-ordered data.
        sequence_start_number : int
            The ROACH sequence number for the first sample of the time-ordered data.
        s21_raw : numpy.ndarray(complex)
            The data, demodulated or not.
        data_demodulated : bool
            True if the data is demodulated.
        roach_state : dict
            State information for the roach; the result of roach.state.
        state : dict
            All non-roach state information.
        description : str
            A human-readable description of this measurement.
        """
        self.tone_bin = tone_bin
        self.tone_amplitude = tone_amplitude
        self.tone_phase = tone_phase
        self.tone_index = tone_index
        self.filterbank_bin = filterbank_bin
        self.epoch = epoch
        self.sequence_start_number = sequence_start_number
        self.s21_raw = s21_raw
        self.data_demodulated = data_demodulated
        self.roach_state = core.StateDict(roach_state)
        super(RoachStream, self).__init__(state=state, description=description)

    @memoized_property
    def sample_time(self):
        """numpy.ndarray(float): The time of each sample relative to the first sample in the stream."""
        return (np.arange(self.s21_raw.shape[-1], dtype='float') /
                self.stream_sample_rate)

    @memoized_property
    def frequency(self):
        return calculate.frequency(self.roach_state, self.tone_bin[self.tone_index])

    @property
    def frequency_MHz(self):
        return 1e-6 * self.frequency

    @memoized_property
    def baseband_frequency(self):
        return calculate.baseband_frequency(self.roach_state, self.tone_bin[self.tone_index])

    @property
    def baseband_frequency_MHz(self):
        return 1e-6 * self.baseband_frequency

    @property
    def stream_sample_rate(self):
        return calculate.stream_sample_rate(self.roach_state)

    @memoized_property
    def s21_raw_mean(self):
        """
        Return the mean of s21_raw for each channel. NaN samples are excluded from the calculation.

        Raises
        ------
        RuntimeWarning
            If all of the samples for a channel are NaN; in this case the return value for that channel will be NaN.

        Returns
        -------
        numpy.ndarray(complex)
           The mean of s21_raw for each channel.
        """
        return np.nanmean(self.s21_raw, axis=-1)

    @memoized_property
    def s21_raw_mean_error(self):
        """
        Estimate the error in s21_raw_mean for each channel. NaN samples are excluded from the calculation.

        The method assumes that the samples are independent, which is probably not true.

        Raises
        ------
        RuntimeWarning
            If all of the samples for a channel are NaN; in this case the return value for that channel will be NaN.

        Returns
        -------
        numpy.ndarray(complex)
            An estimate of the complex standard error of the mean of s21_raw.
        """
        #  If
        # The float cast allows conversion to NaN.
        num_good_samples = np.sum(~np.isnan(self.s21_raw), axis=-1).astype(np.float)
        if isinstance(num_good_samples, np.ndarray):
            # Avoid a ZeroDivisionError if some of the channels have no good samples.
            num_good_samples[num_good_samples == 0] = np.nan
        elif num_good_samples == 0:  # num_good_samples is a scalar; avoid a ZeroDivisionError.
            num_good_samples = np.nan
        return ((np.nanstd(self.s21_raw.real, axis=-1) + 1j * np.nanstd(self.s21_raw.imag, axis=-1)) /
                np.sqrt(num_good_samples))

    @property
    def s21_point(self):
        """
        Return one s21 point per stream, calculated using the best available method.

        The method used to calculate this point may change.
        """
        return self.s21_raw_mean

    @property
    def s21_point_error(self):
        """
        Return an estimate of the standard error of the mean for s21_point, calculated using the best available method.

        The method used to calculate this error may change.
        """
        return self.s21_raw_mean_error

    def folded_shape(self, array, period_samples=None):
        if period_samples is None:
            period_samples = calculate.modulation_period_samples(self.roach_state)
        if period_samples == 0:
            raise ValueError("Cannot fold unmodulated data or with period=0")
        shape = list(array.shape)
        shape[-1] = -1
        shape.append(period_samples)
        return tuple(shape)

    def fold(self, array, period_samples=None, reduce=np.mean):
        reshaped = array.reshape(self.folded_shape(array, period_samples=period_samples))
        if reduce:
            return reduce(reshaped, axis=reshaped.ndim-2)
        else:
            return reshaped

    def epochs(self, start=-np.inf, stop=np.inf):
        """
        Return a StreamArray containing only the data between the given start and stop epochs.

        The returned StreamArray has the same state.

        The indexing follows the Python convention that the first value is inclusive and the second is exclusive:
        start <= epoch < stop
        Thus, the two slices stream_array[t0:t1] and stream_array[t1:t2] will contain all the data occurring at or after
        t0 and before t2, with no duplication.
        """
        start_index = np.searchsorted(self.epoch + self.sample_time, (start,), side='left')
        stop_index = np.searchsorted(self.epoch + self.sample_time, (stop,), side='right')  # This index is not included
        return self.__class__(tone_bin=self.tone_bin, tone_amplitude=self.tone_amplitude,
                              tone_phase=self.tone_phase, tone_index=self.tone_index,
                              filterbank_bin=self.filterbank_bin, epoch=self.epoch + self.sample_time[start_index],
                              sequence_start_number=np.nan,  # This may no longer be valid for the sliced data.
                              s21_raw=self.s21_raw[..., start_index:stop_index],
                              data_demodulated=self.data_demodulated, roach_state=self.roach_state,
                              state=self.state, description=self.description)


class RoachStream0(core.Measurement):
    """
    This class is a factory for producing RoachStream version 1 instances from version 0 data.

    Version 0 did not save the sequence_start_number integer.
    """
    def __new__(cls, *args, **kwargs):
        kwargs['sequence_start_number'] = np.nan
        return RoachStream(*args, **kwargs)


class StreamArray(RoachStream):
    """
    This class represents simultaneously-sampled data from multiple channels.
    """

    _version = 1

    dimensions = OrderedDict([('tone_bin', ('tone_bin',)),
                              ('tone_amplitude', ('tone_bin',)),
                              ('tone_phase', ('tone_bin',)),
                              ('tone_index', ('tone_index',)),
                              ('filterbank_bin', ('tone_index',)),
                              ('s21_raw', ('tone_index', 'sample_time'))])

    def __init__(self, tone_bin, tone_amplitude, tone_phase, tone_index, filterbank_bin, epoch, sequence_start_number,
                 s21_raw, data_demodulated, roach_state, state=None, description=''):
        """
        Return a new StreamArray instance. The integer array tone_index contains the indices of tone_bin,
        tone_amplitude, and tone_phase for the tones demodulated to produce the time-ordered s21_raw data.
        The tone_bin, tone_amplitude, tone_phase, tone_index, and filterbank_bin arrays are 1-D, while s21_raw is 2-D:
        s21_raw.shape == (tone_index.size, sample_time.size)

        Parameters
        ----------
        tone_bin : numpy.ndarray(int)
            An array of integers representing the frequencies of the tones played during the measurement.
        tone_amplitude : numpy.ndarray(float)
            An array of floats representing the amplitudes of the tones played during the measurement.
        tone_phase : numpy.ndarray(float)
            An array of floats representing the radian phases of the tones played during the measurement.
        tone_index : numpy.ndarray(int)
            tone_bin[tone_index] corresponds to the frequency used to produce s21_raw.
        filterbank_bin : numpy.ndarray(int)
            The filter bank bins in which the tones lie.
        epoch : float
            The unix timestamp of the first sample of the time-ordered data.
        sequence_start_number : int
            The ROACH sequence number for the first sample of the time-ordered data.
        s21_raw : numpy.ndarray(complex)
            The data, demodulated or not.
        data_demodulated : bool
            True if the data is demodulated.
        roach_state : dict
            State information for the roach; the result of roach.state.
        state : dict
            All non-roach state information.
        description : str
            A human-readable description of this measurement.
        """
        super(StreamArray, self).__init__(tone_bin=tone_bin, tone_amplitude=tone_amplitude, tone_phase=tone_phase,
                                          tone_index=tone_index, filterbank_bin=filterbank_bin, epoch=epoch,
                                          sequence_start_number=sequence_start_number, s21_raw=s21_raw,
                                          data_demodulated=data_demodulated, roach_state=roach_state, state=state,
                                          description=description)

    def __getitem__(self, number):
        """
        Return a SingleStream object containing the data at the frequency corresponding to the given integer number.
        """
        if isinstance(number, int):
            return SingleStream(tone_bin=self.tone_bin, tone_amplitude=self.tone_amplitude, tone_phase=self.tone_phase,
                                tone_index=self.tone_index[number], filterbank_bin=self.filterbank_bin[number],
                                epoch=self.epoch, sequence_start_number=self.sequence_start_number,
                                s21_raw=self.s21_raw[number, :], data_demodulated=self.data_demodulated,
                                roach_state=self.roach_state, number=number, state=self.state,
                                description=self.description)
        else:
            raise ValueError("Invalid number: {}".format(number))

    def stream(self, number):
        """Deprecated: use __getitem__."""
        return self[number]


class StreamArray0(RoachStream0):
    """
    This class is a factory for producing StreamArray version 1 instances from version 0 data.

    Version 0 did not save the sequence_start_number integer.
    """
    def __new__(cls, *args, **kwargs):
        kwargs['sequence_start_number'] = np.nan
        return StreamArray(*args, **kwargs)


class SingleStream(RoachStream):
    """
    This class contains time-ordered data from a single channel.
    """

    _version = 1

    dimensions = OrderedDict([('tone_bin', ('tone_bin',)),
                              ('tone_amplitude', ('tone_bin',)),
                              ('tone_phase', ('tone_bin',)),
                              ('s21_raw', ('sample_time',))])

    def __init__(self, tone_bin, tone_amplitude, tone_phase, tone_index, filterbank_bin, epoch, sequence_start_number,
                 s21_raw, data_demodulated, roach_state, number=0, state=None, description=''):
        """
        Return a new SingleStream instance. The single integer tone_index is the common index of tone_bin,
        tone_amplitude, and tone_phase for the tone used to produce the time-ordered s21_raw data.

        The tone_bin, tone_amplitude, tone_phase, tone_index, filterbank_bin, and s21_raw arrays are all 1-D.

        Parameters
        ----------
        tone_bin : numpy.ndarray(int)
            An array of integers representing the frequencies of the tones played during the measurement.
        tone_amplitude : numpy.ndarray(float)
            An array of floats representing the amplitudes of the tones played during the measurement.
        tone_phase : numpy.ndarray(float)
            An array of floats representing the radian phases of the tones played during the measurement.
        tone_index : int
            tone_bin[tone_index] corresponds to the frequency used to produce s21_raw.
        filterbank_bin : int
            An int that is the filter bank bin in which the tone lies.
        epoch : float
            The unix timestamp of the first sample of the time-ordered data.
        sequence_start_number : int
            The ROACH sequence number for the first sample of the time-ordered data.
        s21_raw : numpy.ndarray(complex)
            The data, demodulated or not.
        data_demodulated : bool
            True if the data is demodulated.
        roach_state : dict
            State information for the roach; the result of roach.state.
        number : int
            The number of this instance in some larger structure, such as a StreamArray.
        state : dict
            All non-roach state information.
        description : str
            A human-readable description of this measurement.
        """
        self.number = number
        super(SingleStream, self).__init__(tone_bin=tone_bin, tone_amplitude=tone_amplitude, tone_phase=tone_phase,
                                           tone_index=tone_index, filterbank_bin=filterbank_bin, epoch=epoch,
                                           sequence_start_number=sequence_start_number, s21_raw=s21_raw,
                                           data_demodulated=data_demodulated, roach_state=roach_state, state=state,
                                           description=description)


class SingleStream0(RoachStream0):
    """
    This class is a factory for producing SingleStream version 1 instances from version 0 data.

    Version 0 did not save the sequence_start_number integer.
    """
    def __new__(cls, *args, **kwargs):
        kwargs['sequence_start_number'] = np.nan
        return SingleStream(*args, **kwargs)


class SweepArray(core.Measurement):
    """
    This class contains list of streams.

    The properties return values in ascending frequency order.
    """

    _version = 0

    def __init__(self, stream_arrays, state=None, description=''):
        """
        Parameters
        ----------
        stream_arrays : MeasurementList(StreamArray)
            The streams that make up the sweep.
        state : dict
            All non-roach state information.
        description : str
            A human-readable description of this measurement.
        """
        self.stream_arrays = stream_arrays
        super(SweepArray, self).__init__(state=state, description=description)

    def __getitem__(self, number):
        if isinstance(number, int):
            return SingleSweep(streams=core.MeasurementList(sa.stream(number) for sa in self.stream_arrays),
                               number=number, state=self.state, description=self.description)
        else:
            raise ValueError("Invalid number: {}".format(number))

    def sweep(self, number):
        """Deprecated; use __getitem__."""
        return self[number]

    @property
    def num_channels(self):
        try:
            if np.any(np.diff([sa.tone_index.size for sa in self.stream_arrays])):
                raise ValueError("Channel numbers differ between stream arrays.")
            else:
                return self.stream_arrays[0].tone_index.size
        except IndexError:
            return 0

    @memoized_property
    def ascending_order(self):
        """numpy.ndarray[int]: Re-arranges values for this SweepArray in ascending frequency order."""
        return np.concatenate([sa.frequency for sa in self.stream_arrays]).argsort()

    @memoized_property
    def frequency(self):
        """numpy.ndarray[float]: The frequencies of all data points in ascending order."""
        return np.concatenate([sa.frequency for sa in self.stream_arrays])[self.ascending_order]

    @property
    def frequency_MHz(self):
        """numpy.ndarray[float]: The frequencies in MHz of all data points, in ascending order."""
        return 1e-6 * self.frequency

    @memoized_property
    def s21_point(self):
        """numpy.ndarray[complex]: The s21_point values of all data points, in ascending frequency order."""
        return np.concatenate([sa.s21_point for sa in self.stream_arrays])[self.ascending_order]

    @memoized_property
    def s21_point_error(self):
        """numpy.ndarray[complex]: The s21_point_error values of all data points, in ascending frequency order."""
        return np.concatenate([sa.s21_point_error for sa in self.stream_arrays])[self.ascending_order]

    @memoized_property
    def s21_raw(self):
        """numpy.ndarray[complex]: The raw s21 streams of all data points, in ascending frequency order."""
        return np.vstack([stream_array.s21_raw for stream_array in self.stream_arrays])[self.ascending_order, :]

    def to_dataframe(self, add_origin=True):
        dataframes = []
        for number in range(self.num_channels):
            dataframes.append(self.sweep(number).to_dataframe(add_origin=add_origin))
        return pd.concat(dataframes, ignore_index=True)


class SingleSweep(core.Measurement):
    """
    This class contains list of single streams with different frequencies.

    The properties return values in ascending frequency order.
    """

    _version = 0

    def __init__(self, streams, number=0, state=None, description=''):
        """
        Parameters
        ----------
        streams: MeasurementList(SingleStream)
            The streams that make up the sweep.
        number : int
            The number of this instance in the SweepArray from which it was created.
        state : dict
            All non-roach state information.
        description : str
            A human-readable description of this measurement.
        """
        self.streams = streams
        self.number = number
        super(SingleSweep, self).__init__(state=state, description=description)

    @memoized_property
    def ascending_order(self):
        """numpy.ndarray[int]: Re-arranges values for this SingleSweep in ascending frequency order."""
        return np.array([s.frequency for s in self.streams]).argsort()

    @memoized_property
    def frequency(self):
        """numpy.ndarray[float]: The frequencies of all data points in ascending order."""
        return np.array([s.frequency for s in self.streams])[self.ascending_order]

    @property
    def frequency_MHz(self):
        """numpy.ndarray[float]: The frequencies in MHz of all data points, in ascending order."""
        return 1e-6 * self.frequency

    @memoized_property
    def s21_point(self):
        """numpy.ndarray[complex]: The s21_point values of all data points, in ascending frequency order."""
        return np.array([s.s21_point for s in self.streams])[self.ascending_order]

    @memoized_property
    def s21_point_error(self):
        """numpy.ndarray[complex]: The s21_point_error values of all data points, in ascending frequency order."""
        return np.array([s.s21_point_error for s in self.streams])[self.ascending_order]

    @memoized_property
    def s21_raw(self):
        """numpy.ndarray[complex]: The raw s21 streams of all data points, in ascending frequency order."""
        return np.vstack([s.s21_raw for s in self.streams])[self.ascending_order, :]

    @memoized_property
    def s21_normalized(self):
        return np.array([self.resonator.remove_background(f, s21) for f, s21 in zip(self.frequency, self.s21_point)])

    @memoized_property
    def s21_normalized_error(self):
        return np.array([self.resonator.remove_background(f, s21_error)
                         for f, s21_error in zip(self.frequency, self.s21_point_error)])

    @memoized_property
    def resonator(self):
        """BaseResonator: the result of the last """
        return self.fit_resonator()

    # TODO: add arguments to specify model, etc.
    def fit_resonator(self, model=lmfit_resonator.LinearResonatorWithCable):
        # Reset the memoized properties that depend on the resonator fit.
        for attr in ('_s21_normalized', '_s21_normalized_error'):
            if hasattr(self, attr):
                delattr(self, attr)
        self._resonator = model(frequency=self.frequency, s21=self.s21_point, errors=self.s21_point_error)
        self._resonator.fit()
        return self._resonator

    def to_dataframe(self, add_origin=True):
        data = {'number': self.number, 'analysis_epoch': time.time(), 'start_epoch': self.start_epoch()}
        try:
            for thermometer, temperature in self.state['temperature'].items():
                data['temperature_{}'.format(thermometer)] = temperature
        except KeyError:
            pass
        try:
            for key, value in self.streams[0].roach_state.items():
                data['roach_{}'.format(key)] = value
        except KeyError:
            pass

        flat_state = self.state.flatten(wrap_lists=True)
        data.update(flat_state)

        for param in self.resonator.current_result.params.values():
            data['res_{}'.format(param.name)] = param.value
            data['res_{}_error'.format(param.name)] = param.stderr
        data['res_redchi'] = self.resonator.current_result.redchi
        data['res_Q_i'] = self.resonator.Q_i
        data['res_Q_e'] = self.resonator.Q_e

        data['res_s21_data'] = [self.resonator.data]
        data['res_frequency_data'] = [self.resonator.frequency]
        data['res_s21_errors'] = [self.resonator.errors]
        modelf = np.linspace(self.resonator.frequency.min(), self.resonator.frequency.max(), 1000)
        models21 = self.resonator.model.eval(params=self.resonator.current_params, f=modelf)
        data['res_model_frequency'] = [modelf]
        data['res_model_s21'] = [models21]

        dataframe = pd.DataFrame(data, index=[0])
        if add_origin:
            self.add_origin(dataframe)
        return dataframe


class SweepStreamArray(core.Measurement):

    _version = 0

    def __init__(self, sweep_array, stream_array, state=None, description=''):
        if sweep_array.num_channels != stream_array.tone_index.size:
            raise core.MeasurementError("The number of SweepArray channels does not match the StreamArray number.")
        self.sweep_array = sweep_array
        self.stream_array = stream_array
        super(SweepStreamArray, self).__init__(state=state, description=description)

    @property
    def num_channels(self):
        return self.sweep_array.num_channels

    def __getitem__(self, number):
        """
        Return a SweepStream object containing the data at the frequency corresponding to the given integer number.
        """
        if isinstance(number, int):
            return SingleSweepStream(sweep=self.sweep_array.sweep(number), stream=self.stream_array.stream(number),
                                     number=number, state=self.state, description=self.description)
        else:
            raise ValueError("Invalid number: {}".format(number))

    def sweep_stream(self, number):
        """Deprecated: use __getitem__"""
        return self[number]

    def to_dataframe(self, deglitch=True):
        dataframes = []
        for number in range(self.num_channels):
            dataframes.append(self.sweep_stream(number).to_dataframe(deglitch=deglitch))
        return pd.concat(dataframes, ignore_index=True)


class SingleSweepStream(core.Measurement):

    _version = 0

    def __init__(self, sweep, stream, number=0, state=None, description=''):
        self.sweep = sweep
        self.stream = stream
        self.number = number
        super(SingleSweepStream, self).__init__(state=state, description=description)

    @property
    def resonator(self):
        return self.sweep.resonator

    @memoized_property
    def stream_s21_normalized(self):
        return self.sweep.resonator.remove_background(self.stream.frequency, self.stream.s21_raw)

    @memoized_property
    def stream_s21_normalized_deglitched(self):
        return self._set_stream_s21_normalized_deglitched()

    def _set_stream_s21_normalized_deglitched(self, window_in_seconds=1, deglitch_threshold=5):
        window = int(2 ** np.ceil(np.log2(window_in_seconds * self.stream.stream_sample_rate)))
        self._stream_s21_normalized_deglitched = deglitch_window(self.stream_s21_normalized, window,
                                                                 thresh=deglitch_threshold)
        return self._stream_s21_normalized_deglitched

    @property
    def q(self):
        """
        Return the inverse internal quality factor q = 1 / Q_i calculated by inverting the resonator model.
        :return: an array of q values from self.stream corresponding to self.stream.epoch.
        """
        if not hasattr(self, '_q'):
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
        if not hasattr(self, '_x'):
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
        if not hasattr(self, '_S_frequency'):
            self._set_S()
        return self._S_frequency

    @property
    def S_qq(self):
        """
        The single-sided spectral density of q(t), S_qq(f), where f is self.S_frequency.
        :return: an array of complex values representing the spectral density of q(t)
        """
        if not hasattr(self, '_S_qq'):
            self._set_S()
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
        if not hasattr(self, '_S_xx'):
            self._set_S()
        return self._S_xx

    # TODO: calculate errors in PSDs
    def _set_S(self, NFFT=None, window=mlab.window_none, binned=True, **kwargs):
        # Use the same length calculation as SweepNoiseMeasurement
        if NFFT is None:
            NFFT = int(2**(np.floor(np.log2(self.stream.s21_raw.size)) - 3))
        S_qq, f = mlab.psd(self.q, Fs=self.stream.stream_sample_rate, NFFT=NFFT, window=window, **kwargs)
        S_xx, f = mlab.psd(self.x, Fs=self.stream.stream_sample_rate, NFFT=NFFT, window=window, **kwargs)
        if binned:
            f, (S_xx, S_qq) = iqnoise.log_bin(f,[S_xx, S_qq])
        self._S_frequency = f
        self._S_qq = S_qq
        self._S_xx = S_xx

    def to_dataframe(self, deglitch=True, add_origin=True):
        if not deglitch:
            self._set_q_and_x(deglitch=False)
        data = {'number': self.number, 'analysis_epoch': time.time(), 'start_epoch': self.start_epoch()}
        try:
            for thermometer, temperature in self.state['temperature'].items():
                data['temperature_{}'.format(thermometer)] = temperature
        except KeyError:
            pass
        try:
            for key, value in self.stream.roach_state.items():
                data['roach_{}'.format(key)] = value
        except KeyError:
            pass

        flat_state = self.stream.state.flatten(wrap_lists=True)
        data.update(flat_state)

        for param in self.sweep.resonator.current_result.params.values():
            data['res_{}'.format(param.name)] = param.value
            data['res_{}_error'.format(param.name)] = param.stderr
        data['res_redchi'] = self.sweep.resonator.current_result.redchi
        data['res_Q_i'] = self.sweep.resonator.Q_i
        data['res_Q_e'] = self.sweep.resonator.Q_e

        data['S_xx'] = [self.S_xx]
        data['S_yy'] = [self.S_yy]
        data['S_qq'] = [self.S_qq]
        data['S_frequency'] = [self.S_frequency]

        data['res_s21_data'] = [self.sweep.resonator.data]
        data['res_frequency_data'] = [self.sweep.resonator.frequency]
        data['res_s21_errors'] = [self.sweep.resonator.errors]
        modelf = np.linspace(self.sweep.resonator.frequency.min(), self.sweep.resonator.frequency.max(), 1000)
        models21 = self.sweep.resonator.model.eval(params=self.sweep.resonator.current_params, f=modelf)
        data['res_model_frequency'] = [modelf]
        data['res_model_s21'] = [models21]

        try:
            data['folded_x'] = [self.stream.fold(self.x)]
            data['folded_q'] = [self.stream.fold(self.q)]
            data['folded_normalized_s21'] = [self.stream.fold(self.stream_s21_normalized)]
        except ValueError:
            pass

        dataframe = pd.DataFrame(data, index=[0])
        if add_origin:
            self.add_origin(dataframe)
        return dataframe

    # TODO: think about how to avoid code reuse here; monkey-patching in __init__ causes read/write problems.
    def folded_shape(self, array, period_samples=None):
        if period_samples is None:
            period_samples = calculate.modulation_period_samples(self.stream.roach_state)
        if period_samples == 0:
            raise ValueError("Cannot fold unmodulated data or with period=0")
        shape = list(array.shape)
        shape[-1] = -1
        shape.append(period_samples)
        return tuple(shape)

    def fold(self, array, period_samples=None, reduce=np.mean):
        reshaped = array.reshape(self.folded_shape(array, period_samples=period_samples))
        if reduce:
            return reduce(reshaped, axis=reshaped.ndim - 2)
        else:
            return reshaped


class SweepStreamList(core.Measurement):

    _version = 0

    def __init__(self, sweep, stream_list, state=None, description=''):
        self.sweep = sweep
        self.stream_list = stream_list
        super(SweepStreamList, self).__init__(state=state, description=description)

    def __getitem__(self, number):
        return SingleSweepStreamList(self.sweep[number],
                                     core.MeasurementList(sa[number] for sa in self.stream_list),
                                     number=number, state=self.state, description=self.description)

    def single_sweep_stream_list(self, number):
        """Deprecated: use __getitem__"""
        return self[number]


class SingleSweepStreamList(core.Measurement):

    _version = 0

    def __init__(self, single_sweep, stream_list, number=0, state=None, description=''):
        self.sweep = single_sweep
        self.stream_list = stream_list
        self.number = number
        super(SingleSweepStreamList, self).__init__(state=state, description=description)

    def state_vector(self, *keys):
        vector = []
        for stream in self.stream_list:
            state = stream.state
            for key in keys:
                if state is np.nan:
                    break
                state = state.get(key, np.nan)
            vector.append(state)
        return np.array(vector)
