import numpy as np
from kid_readout.measurement.acquire import acquire
from kid_readout.roach.baseband import RoachBaseband
from kid_readout.roach.tests.mock_roach import MockRoach
from kid_readout.roach.tests.mock_valon import MockValon


def test_baseband_sweep():
    num_tones = 16
    num_waveforms = 2**5
    num_tone_samples = 2**10
    ri = RoachBaseband(roach=MockRoach('roach'), initialize=False, adc_valon=MockValon())
    center_frequencies = np.linspace(100, 200, num_tones)
    offsets = np.linspace(-20e-3, 20e-3, num_waveforms)
    tone_banks = [center_frequencies + offset for offset in offsets]
    get_state = lambda: {'something': 'something state'}
    # preload = False
    sweep = acquire.sweep(ri=ri, tone_banks=tone_banks, num_tone_samples=num_tone_samples, length_seconds=0.1,
                          get_state=get_state, description="description")
    assert len(sweep.stream_arrays) == num_waveforms
    assert all([stream_array.s21_raw.shape[0] == num_tones for stream_array in sweep.stream_arrays])
    # preload = True
    sweep = acquire.sweep(ri=ri, tone_banks=tone_banks, num_tone_samples=num_tone_samples, length_seconds=1,
                          preload=True, get_state=get_state, description="description")
    assert len(sweep.stream_arrays) == num_waveforms
    assert all([stream_array.s21_raw.shape[0] == num_tones for stream_array in sweep.stream_arrays])
