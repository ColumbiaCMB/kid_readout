import numpy as np

from kid_readout.roach.tests.mock_roach import MockRoach
from kid_readout.roach.tests.mock_valon import MockValon
from kid_readout.roach.baseband import RoachBaseband
from kid_readout.measurement.acquire import acquire


class Test(object):

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
