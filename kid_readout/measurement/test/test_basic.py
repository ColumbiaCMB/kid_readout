import numpy as np
import warnings

from kid_readout.measurement import basic
from kid_readout.measurement.test import utilities
from kid_readout.roach.tests.mock_roach import MockRoach
from kid_readout.roach.tests.mock_valon import MockValon
from kid_readout.roach.baseband import RoachBaseband
from kid_readout.measurement.acquire import acquire


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
    """Using np.nan * (1 + 1j) is crucial here because the error calculation takes real and imag parts of s21_raw."""
    num_tones = 4
    num_samples = 128
    s21 = np.empty((num_tones, num_samples), dtype=np.complex)
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


class TestStack(object):

    @classmethod
    def setup(cls):
        num_tones = 16
        num_waveforms = 2**5
        num_tone_samples = 2**10
        length_seconds = 0.1
        ri = RoachBaseband(roach=MockRoach('roach'), initialize=False, adc_valon=MockValon())
        center_frequencies = np.linspace(100, 200, num_tones)
        offsets = np.linspace(-20e-3, 20e-3, num_waveforms)
        tone_banks = [center_frequencies + offset for offset in offsets]
        state = {'something': 'something state'}
        cls.sweep_array = acquire.run_sweep(ri=ri, tone_banks=tone_banks, num_tone_samples=num_tone_samples,
                                            length_seconds=length_seconds, state=state, description="description")


    def test_tone_bin_stack(self):
        assert self.sweep_array.tone_bin_stack.shape == (np.sum([stream_array.tone_bin.size
                                                                 for stream_array
                                                                 in self.sweep_array.stream_arrays]),)

    def test_tone_amplitude_stack(self):
        assert self.sweep_array.tone_amplitude_stack.shape == (np.sum([stream_array.tone_amplitude.size
                                                                       for stream_array
                                                                       in self.sweep_array.stream_arrays]),)

    def test_tone_phase_stack(self):
        assert self.sweep_array.tone_phase_stack.shape == (np.sum([stream_array.tone_phase.size
                                                                   for stream_array
                                                                   in self.sweep_array.stream_arrays]),)

    def test_s21_raw_stack(self):
        assert self.sweep_array.s21_raw_stack.shape[0] == np.sum([stream_array.s21_raw.shape[0]
                                                                  for stream_array
                                                                  in self.sweep_array.stream_arrays])
        assert all([self.sweep_array.s21_raw_stack.shape[1] == stream_array.s21_raw.shape[1]
                    for stream_array in self.sweep_array.stream_arrays])
