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
    This class contains tests for the ROACH1 that can run using either real or mock hardware.
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
    pass


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
