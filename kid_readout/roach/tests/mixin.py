"""
This module contains mix-in classes with methods designed to test RoachInterface instances.

These classes do not set up real or mock hardware, so they are not intended to be run directly. To use them, write a
class that inherits from one or more of them and has a setup() class method that creates a RoachInterface instance:
@classmethod
def setup(cls):
    cls.ri = set_up_roach_somehow()
using combination of real and mock hardware.

Currently, the setup() method is run before the tests in each mix-in class. I'm not sure how to get nose to stop doing
this, and in any case it's probably easier to write tests if each class can expects a fresh ROACH.
"""
import numpy as np

from kid_readout.measurement import core
from kid_readout.roach import calculate


class RoachMixin(object):
    """
    This class has methods that are tests applicable to all ROACH types and configurations.
    """

    def test_state(self):
        _ = self.ri.state
        _ = self.ri.get_state()
        _ = self.ri.state_arrays
        _ = self.ri.get_state_arrays()
        _ = self.ri.active_state_arrays
        _ = self.ri.get_active_state_arrays()

    # TODO: why was this test being run only for RoachBaseband and RoachHeterodyne?
    def test_precomputed_wavenorm(self):
        for k in range(9):
            self.ri.set_tone_baseband_freqs(np.linspace(100, 120, 2 ** k), nsamp=2 ** 16, preset_norm=False)
            actual_wavenorm = self.ri.wavenorm
            self.ri.set_tone_baseband_freqs(np.linspace(100, 120, 2 ** k), nsamp=2 ** 16, phases=self.ri.phases,
                                            preset_norm=True)
            assert self.ri.wavenorm >= 0.99 * actual_wavenorm  # guarantees the wave won't overflow

    def test_calculate_modulation_period(self):
        roach_state = core.StateDict(modulation_rate=7, modulation_output=2)
        assert calculate.modulation_period_samples(roach_state) == 256

    def test_get_measurement_blocks(self):
        num_tones = 32
        self.ri.set_tone_baseband_freqs(np.linspace(100, 120, num_tones), nsamp=2 ** 16)
        self.ri.select_fft_bins(range(num_tones))
        _ = self.ri.get_measurement_blocks(2)


class Roach1Mixin(object):
    """
    This class contains tests for the ROACH1 that can run using either real or mock hardware.
    """

    def test_get_current_bank(self):
        assert self.ri.get_current_bank() is not None


class Roach2Mixin(object):
    """
    This class contains tests for the ROACH2 that can run using either real or mock hardware.
    """
    pass


class BasebandSoftwareMixin(object):
    """
    This class contains tests for baseband software that can run using either real or mock hardware.
    """

    def test_is_not_heterodyne(self):
        assert not self.ri.heterodyne

    # TODO: why was this test being run only for RoachBaseband and RoachHeterodyne?
    def test_calc_fft_bins(self):
        for nsamp in 2 ** np.arange(10, 18):
            max_nsamp = nsamp / 2  # only positive bins are valid for baseband
            # Arguments give closed interval, and need to avoid max_nsamp edge
            tone_bins = np.random.random_integers(0, max_nsamp - 1, size=128)
            bins = self.ri.calc_fft_bins(tone_bins, nsamp)
            assert np.all(bins >= 0)
            assert np.all(bins < self.ri.nfft)


class HeterodyneSoftwareMixin(object):
    """
    This class contains tests for heterodyne software that can run using either real or mock hardware.
    """

    def test_is_heterodyne(self):
        assert self.ri.heterodyne

    def test_calc_fft_bins(self):
        for nsamp in 2 ** np.arange(10, 18):
            max_nsamp = nsamp  # Both positive and negative bins are valid for heterodyne
            # Arguments give closed interval, and need to avoid max_nsamp edge
            tone_bins = np.random.random_integers(0, max_nsamp - 1, size=128)
            bins = self.ri.calc_fft_bins(tone_bins, nsamp)
            assert np.all(bins >= 0)
            assert np.all(bins < self.ri.nfft)


class BasebandHardwareMixin(object):
    """
    This class contains tests for baseband hardware that can run in loopback mode.
    """
    pass


class HeterodyneHardwareMixin(object):
    """
    This class contains tests for heterodyne hardware that can run in loopback mode.
    """
    def test_fft_bin_selection(self):
        test_cases = [#np.array([[16368, 16370, 16372, 16374, 16376, 16379, 16381, 16383, 1,
            #3,     5,     8,    10,    12,    14,    16]]),  #this special case doesn't quite work because of the
            # readout order
                      np.array([[16368, 16370, 16372, 16374, 16376, 16379, 16381, 16383,
                                 8,    10,    12,    14,    16, 18, 20, 22]]),
                      np.array([[7333,7335]]),
                      np.array([[ 9328, 10269, 11210, 12150, 13091, 14032, 14973, 15914,   470,
         1411,  2352,  3293,  4234,  5174,  6115,  7056]]),
                      np.array([[7040, 7042, 7044, 7046, 7048, 7051, 7053, 7055, 7057, 7059, 7061,
        7064, 7066, 7068, 7070, 7072]]),
                      np.array([[8193, 8195, 8197, 8199, 8201, 8203, 8206, 8208, 8210, 8212, 8214,
        8216, 8218, 8220, 8222, 8224, 8160, 8162, 8164, 8166, 8168, 8171,
        8173, 8175, 8177, 8179, 8181, 8183, 8185, 8187, 8189, 8191]])]
        for bin_array in test_cases:
            yield self.check_fft_bin_selection, bin_array

    def check_fft_bin_selection(self,bin_array):
        self.ri.set_debug(True)
        self.ri.fft_bins = bin_array
        self.ri.select_fft_bins(range(self.ri.fft_bins.shape[1]))
        #time.sleep(0.1)
        data,sequence = self.ri.get_data(demod=False)
        assert(np.all(data[0,:].imag.astype('int') == self.ri.fpga_fft_readout_indexes))

    def test_peak_fft_bin(self):
        # Because the PFB channels currently have rather poor isolation up to several channels away, this test will
        # only work for widely spaced channeels
        test_cases = [np.array([[1000, 1201, 1457, 8096]]),
                      np.array([[8193, 11111, 12345, 16383, 1000, 1201, 1457, 7096]])]
        for bin_array in test_cases:
            yield self.check_peak_fft_bin, bin_array
    def check_peak_fft_bin(self,bin_array):
        baseband_freqs = bin_array[0,:]/2.**14
        baseband_freqs[baseband_freqs>0.5] = baseband_freqs[baseband_freqs>0.5]-1
        baseband_freqs = self.ri.fs*baseband_freqs
        self.ri.set_tone_baseband_freqs(baseband_freqs,nsamp=2**16)
        self.ri.set_debug(False)
        self.ri.set_loopback(True)
        self.ri.set_fft_gain(0)
        base_fft_bins = self.ri.fft_bins.copy()
        values = []
        bin_offsets = [-1,0,1]
        for bin_offset in bin_offsets:
            self.ri.fft_bins = base_fft_bins + bin_offset
            self.ri.fft_bins = np.mod(self.ri.fft_bins,2**14)
            self.ri.select_fft_bins(range(self.ri.fft_bins.shape[1]))
            data,sequence = self.ri.get_data(demod=False)
            values.append((np.abs(data)**2).mean(0))
        values = np.array(values)
        for bin_index in range(values.shape[1]):
            assert(bin_offsets[values[:,bin_index].argmax()]==0)


class BasebandAnalogMixin(object):
    """
    This class contains tests for heterodyne hardware that require an analog board.
    """
    pass


class HeterodyneAnalogMixin(object):
    """
    This class contains tests for heterodyne hardware that require an analog board.
    """
    pass


class MockMixin(object):
    """
    This class contains tests specifically for the MockRoach and MockValon classes.
    """
    pass
