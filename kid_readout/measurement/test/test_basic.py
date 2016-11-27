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
    s21[0, ::2] = 1 + 2j
    s21[0, 1::2] = np.nan * (1 + 1j)
    correct_s21_error_0 = 0
    s21[1, ::2] = 1 + 0j
    s21[1, 1::2] = 0 + 1j
    correct_s21_error_1 = (0.5 + 0.5j) / np.sqrt(num_samples)
    s21[2, ::4] = -1 + 0j
    s21[2, 2::4] = 0 - 1j
    s21[2, 1::2] = np.nan * (1 + 1j)
    correct_s21_error_2 = (0.5 + 0.5j) / np.sqrt(num_samples/ 2)  # Half the samples are NaN.
    # Using np.nan * (1 + 1j) is crucial here because the error calculation takes real and imag parts of s21_raw.
    # The correct error for index 3 is NaN
    s21[3, :] = np.nan * (1 + 1j)
    # Test StreamArray code path
    sa = utilities.fake_stream_array(num_tones=num_tones)
    sa.s21_raw = s21
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        s21_raw_mean_error = sa.s21_raw_mean_error
        assert len(w) == 2
        assert issubclass(w[0].category, RuntimeWarning)  # From real
        assert issubclass(w[1].category, RuntimeWarning)  # From imag
    np.testing.assert_allclose(s21_raw_mean_error[0], correct_s21_error_0)
    np.testing.assert_allclose(s21_raw_mean_error[1], correct_s21_error_1)
    np.testing.assert_allclose(s21_raw_mean_error[2], correct_s21_error_2)
    assert np.isnan(s21_raw_mean_error[3])
    # Test SingleStream code path.
    assert sa[0].s21_raw_mean_error == correct_s21_error_0
    assert sa[1].s21_raw_mean_error == correct_s21_error_1
    assert sa[2].s21_raw_mean_error == correct_s21_error_2
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert np.isnan(sa[3].s21_raw_mean_error)
        assert len(w) == 2
        assert issubclass(w[0].category, RuntimeWarning)  # From real
        assert issubclass(w[1].category, RuntimeWarning)  # From imag


class TestSweepArray(object):

    @classmethod
    def setup(cls):
        cls.sa = utilities.fake_sweep_array()

    def test_delete_memoized_property_caches(self):
        memoized = ['ascending_order', 'frequency', 's21_point', 's21_point_error', 's21_raw']
        for attr in memoized:
            assert not hasattr(self.sa, '_' + attr), "Cache present: {}".format(attr)
        _ = self.sa.frequency
        _ = self.sa.s21_point
        _ = self.sa.s21_point_error
        _ = self.sa.s21_raw
        for attr in memoized:
            assert hasattr(self.sa, '_' + attr), "Cache missing: {}".format(attr)
        self.sa._delete_memoized_property_caches()
        for attr in memoized:
            assert not hasattr(self.sa, '_' + attr), "Cache present: {}".format(attr)

    def test_frequency(self):
        assert np.all(np.diff(self.sa.frequency))  # Note that this will fail if there are duplicate tones.

    def test_s21_raw(self):
        random_stream_index = np.random.randint(len(self.sa.stream_arrays))
        random_stream_array = self.sa.stream_arrays[random_stream_index]
        random_channel_number = np.random.randint(random_stream_array.tone_index.size)
        random_single_stream = random_stream_array[random_channel_number]
        random_frequency = random_single_stream.frequency
        random_s21_raw = random_single_stream.s21_raw
        sorted_frequencies = np.sort(np.concatenate([s.frequency for s in self.sa.stream_arrays]))
        sorted_index = np.searchsorted(sorted_frequencies, random_frequency)
        assert random_frequency == self.sa.frequency[sorted_index]
        assert np.all(random_s21_raw == self.sa.s21_raw[sorted_index])

    def test_start_epoch(self):
        assert self.sa.start_epoch() == self.sa.stream_arrays[0].epoch


class TestSingleSweep(object):

    @classmethod
    def setup(cls):
        cls.ss = utilities.fake_single_sweep()

    def test_delete_memoized_property_caches(self):
        memoized = ['ascending_order', 'frequency', 's21_point', 's21_point_error', 's21_raw',
                    's21_normalized', 's21_normalized_error', 'resonator']  # These depend on the resonator
        for attr in memoized:
            assert not hasattr(self.ss, '_' + attr), "Cache present: {}".format(attr)
        # Note that we have to fit the resonator first because fit_resonator clears all caches.
        _ = self.ss.resonator
        _ = self.ss.frequency
        _ = self.ss.s21_point
        _ = self.ss.s21_point_error
        _ = self.ss.s21_raw
        _ = self.ss.s21_normalized
        _ = self.ss.s21_normalized_error
        for attr in memoized:
            assert hasattr(self.ss, '_' + attr), "Cache missing: {}".format(attr)
        self.ss._delete_memoized_property_caches()
        for attr in memoized:
            assert not hasattr(self.ss, '_' + attr), "Cache present: {}".format(attr)

    def test_frequency(self):
        assert np.all(np.diff(self.ss.frequency))  # Note that this will fail if there are duplicate tones.

    def test_s21_raw(self):
        random_stream_index = np.random.randint(len(self.ss.streams))
        random_single_stream = self.ss.streams[random_stream_index]
        random_frequency = random_single_stream.frequency
        random_s21_raw = random_single_stream.s21_raw
        sorted_frequencies = np.sort([s.frequency for s in self.ss.streams])
        sorted_index = np.searchsorted(sorted_frequencies, random_frequency)
        assert random_frequency == self.ss.frequency[sorted_index]
        assert np.all(random_s21_raw == self.ss.s21_raw[sorted_index])

    def test_start_epoch(self):
        assert self.ss.start_epoch() == self.ss.streams[0].epoch


class TestSingleSweepStream(object):

    @classmethod
    def setup(cls):
        cls.sss = utilities.fake_single_sweep_stream()

    def test_delete_memoized_property_caches(self):
        memoized = ['stream_s21_normalized', 'stream_s21_normalized_deglitched', 'q', 'x',
                    'S_frequency', 'S_qq', 'S_xx', 'pca_S_frequency', 'pca_S_00', 'pca_S_11', 'pca_angles',
                    'S_xx_variance', 'S_qq_variance', 'S_counts']
        for attr in memoized:
            assert not hasattr(self.sss, '_' + attr), "Cache present: {}".format(attr)
        self.sss.set_S()
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            self.sss.set_pca()  # This raises 40 ComplexWarnings
            for w in ws:
                assert issubclass(w.category, np.ComplexWarning)
        for attr in memoized:
            assert hasattr(self.sss, '_' + attr), "Cache missing: {}".format(attr)
        _ = self.sss.S_yy_variance
        self.sss.sweep._delete_memoized_property_caches()
        for attr in memoized:
            assert not hasattr(self.sss, '_' + attr), "Cache present: {}".format(attr)

    def test_start_epoch(self):
        assert self.sss.start_epoch() == self.sss.sweep.streams[0].epoch

    def test_tone_offset_frequency(self):
        self.sss.stream.tone_offset_frequency()
        self.sss.stream.tone_offset_frequency(normalized_frequency=False)


