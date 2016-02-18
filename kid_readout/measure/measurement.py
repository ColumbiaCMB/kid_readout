from __future__ import division
import numpy as np
# TODO: replace with a scipy PSD estimator
import matplotlib.pyplot as plt
import pandas as pd
from kid_readout.analysis import resonator
from kid_readout.utils.despike import deglitch_window


CLASS_NAME = '_class_name'  # This is the string used by writer objects to save class names.
RESERVED_NAMES = [CLASS_NAME]


# TODO: add plugin functionality for classes
def get_class(class_name):
    return globals()[class_name]


def is_sequence(class_):
    return issubclass(class_, MeasurementSequence)


class Measurement(object):
    """
    This is an abstract class that represents a measurement for a single channel.

    Measurements are hierarchical: a Measurement can contain other Measurements.

    Each Measurement should be self-contained, meaning that it should contain all data and metadata necessary to
    analyze and understand it. Can this include temperature data?

    Caching: all raw data attributes are public and all special or processed data attributes are private.
    """

    def __init__(self, state=None, analyze=False):
        self._parent = None
        if state is None:
            self.state = {}
        else:
            self.state = state
        if analyze:
            self.analyze()

    def analyze(self):
        """
        Analyze the raw data and create all data products.

        :return: None
        """
        pass

    def to_dataframe(self):
        """
        :return: a DataFrame containing all of the instance attributes.
        """
        pass

    def write(self, writer, location, name):
        """
        Write this measurement to disk using the given writer object. The abstraction used here is that a location is
        a container for hierarchically-organized data.

        :param writer: an object that implements the writer interface.
        :param location: the existing root location into which this object will be written.
        :param name: the name of the location containing this object.
        :return: None

        For example, when called with parameters (writer, 'root', 'measurement0'), the location "root" must already
        exist and the data from this measurement will be stored within the location root/measurement0
        """
        self_location = writer.new(location, name)
        writer.write(self.__class__.__name__, self_location, CLASS_NAME)
        for name, thing in self.__dict__.items():
            if not name.startswith('_'):
                if isinstance(thing, Measurement):
                    thing.write(writer, self_location, name)
                elif isinstance(thing, MeasurementSequence):
                    sequence_location = writer.new(self_location, name)
                    writer.write(thing.__class__.__name__, sequence_location, CLASS_NAME)
                    for index, meas in enumerate(thing):
                        meas.write(writer, sequence_location, str(index))
                else:
                    writer.write(thing, self_location, name)
        return self_location


class MeasurementSequence(object):
    """
    This is a dummy class that exists so that Measurements can contain sequences of other Measurements.
    """
    pass


class MeasurementTuple(tuple, MeasurementSequence):
    """
    Measurements containing tuples of Measurements should use instances of this class so that loading and saving are
    handled automatically.
    """
    pass


class MeasurementList(list, MeasurementSequence):
    """
    Measurements containing lists of Measurements should use instances of this class so that loading and saving are
    handled automatically.
    """
    pass



class Stream(Measurement):
    """
    This class represents time-ordered data from a single channel.
    """

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

    def __init__(self, streams=(), state=None, analyze=False):
        # Don't sort by frequency so that non-monotonic order can be preserved if needed.
        self.streams = MeasurementTuple(streams)
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
        super(SweepStream, self).__init__(state, analyze)

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

    # TODO: calculate errors in PSDs
    def _set_psd_i_and_x(self, NFFT=None, window=plt.mlab.window_none, **kwargs):
        # Use the same length calculation as SweepNoiseMeasurement
        if NFFT is None:
            NFFT = int(2**(np.floor(np.log2(self.stream.s21.size)) - 3))
        psd_ii, f = plt.mlab.psd(self.i, Fs=self.stream.sample_frequency, NFFT=NFFT, window=window, **kwargs)
        psd_xx, f = plt.mlab.psd(self.x, Fs=self.stream.sample_frequency, NFFT=NFFT, window=window, **kwargs)
        self._psd_frequency = f
        self._psd_ii = psd_ii
        self._psd_xx = psd_xx

    # TODO: move this forward to a usable version.
    def to_dataframe(self):
        data = {}
        for param in self.sweep.resonator.result.params.values():
            data['resonator_{}'.format(param.name)] = [param.value]
            data['resonator_{}_error'.format(param.name)] = [param.stderr]
        data['resonator_redchi'] = self.sweep.resonator.result.redchi
        # temperatures
        # roach state
        return pd.DataFrame(data, index=[0])
