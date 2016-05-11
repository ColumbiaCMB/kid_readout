import numpy as np
import warnings

from kid_readout.measurement.test import utilities


def test_s21_raw_mean():
    num_tones = 4
    num_samples = 128
    s21 = np.empty((num_tones, num_samples), dtype=np.complex)
    s21[0, :] = np.nan * (1 + 1j)
    s21[1, :num_samples/2] = 1 + 2j
    s21[1, num_samples/2:] = np.nan * (1 + 1j)
    s21[2, :] = np.linspace(-1, 1, num_samples) + 1j * np.linspace(-2, 0, num_samples)
    s21[3, :] = np.linspace(-100, 0, num_samples) + 1j * np.linspace(-100, 100, num_samples)
    sa = utilities.fake_stream_array(num_tones=num_tones)
    sa.s21_raw = s21
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        s21_raw_mean = sa.s21_raw_mean
        assert len(w) == 1
        assert issubclass(w[0].category, RuntimeWarning)
    assert np.isnan(s21_raw_mean[0])
    np.testing.assert_allclose(s21_raw_mean[1], 1 + 2j)
    np.testing.assert_allclose(s21_raw_mean[2], 0 - 1j)
    np.testing.assert_allclose(s21_raw_mean[3], -50 + 0j)


def test_s21_raw_mean_error():
    num_tones = 4
    num_samples = 128
    s21 = np.empty((num_tones, num_samples), dtype=np.complex)
    # Using np.nan * (1 + 1j) is crucial here because the error calculation takes real and imag parts of s21_raw.
    s21[0, :] = np.nan * (1 + 1j)
    s21[1, ::2] = 1 + 2j
    s21[1, 1::2] = np.nan * (1 + 1j)
    s21[2, ::2] = 1 + 0j
    s21[2, 1::2] = 0 + 1j
    s21[3, ::4] = -1 + 0j
    s21[3, 2::4] = 0 - 1j
    s21[3, 1::2] = np.nan * (1 + 1j)
    sa = utilities.fake_stream_array(num_tones=num_tones)
    sa.s21_raw = s21
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        s21_raw_mean_error = sa.s21_raw_mean_error
        print(w)
        assert len(w) == 2
        assert issubclass(w[0].category, RuntimeWarning)
        assert issubclass(w[1].category, RuntimeWarning)
    assert np.isnan(s21_raw_mean_error[0])
    np.testing.assert_allclose(s21_raw_mean_error[1], 0)
    np.testing.assert_allclose(s21_raw_mean_error[2], (0.5 + 0.5j) / np.sqrt(num_samples))
    np.testing.assert_allclose(s21_raw_mean_error[3], (0.5 + 0.5j) / np.sqrt(num_samples/ 2))


class TestSweepArray(object):

    @classmethod
    def setup(cls):
        cls.sweep_array = utilities.fake_sweep_array()

    def test_frequency(self):
        assert np.all(np.diff(self.sweep_array.frequency))  # Note that this will fail if there are duplicate tones.

    def test_s21_raw(self):
        random_stream_index = np.random.randint(len(self.sweep_array.stream_arrays))
        random_stream_array = self.sweep_array.stream_arrays[random_stream_index]
        random_channel_number = np.random.randint(random_stream_array.tone_index.size)
        random_single_stream = random_stream_array[random_channel_number]
        random_frequency = random_single_stream.frequency
        random_s21_raw = random_single_stream.s21_raw
        sorted_frequencies = np.sort(np.concatenate([s.frequency for s in self.sweep_array.stream_arrays]))
        sorted_index = np.searchsorted(sorted_frequencies, random_frequency)
        assert random_frequency == self.sweep_array.frequency[sorted_index]
        assert np.all(random_s21_raw == self.sweep_array.s21_raw[sorted_index])


    #def test_s21_point(self):



