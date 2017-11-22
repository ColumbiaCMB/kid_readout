"""
For each resonator, sweep the tone and filterbank frequencies across the resonance.

At each LO frequency, take one sweep and one streams at different places across the filterbank bin, moving the baseband
frequencies and LO frequency in opposite directions.
"""
import time

import numpy as np

from kid_readout.roach import analog, hardware_tools
from kid_readout.measurement import acquire, basic
from kid_readout.equipment import hardware

acquire.show_settings()
acquire.show_git_status()
logger = acquire.get_script_logger(__file__)

# Parameters
suffix = 'r1h11_lo_sweep'
wait = 5
fft_gain = 2
num_tones_sweep = 255
f_minimum = 10e6  # Keep the tones away from the LO by at least this frequency.
df_lo_MHz = 2.5e-3  # Allegedly the minimum resolution
all_f0_MHz = np.array([2254.837, 2326.842, 2483.490, 3313.270, 3378.300, 3503.600, 3524.435])
f0 = all_f0_MHz[3:4]
attenuations = [40]
# The minimum for sweep resolution: gives 2 ** 5 tones per bin
tone_sample_exponent = 16
# This should sweep across one entire bin, with 32 / 4 = 8 tones in the bin.
n_lo_offset = np.arange(-32, 33, 4)
length_seconds_sweep = 0.1
length_seconds_stream = 10

# Hardware
conditioner = analog.HeterodyneMarkII()
magnet = hardware.Thing(name='magnet_array', state={'orientation': 'up',
                                                    'distance_from_base_mm': 276})
hw = hardware.Hardware(conditioner, magnet)
ri = hardware_tools.r1h11_with_mk2(initialize=True, use_config=False)
ri.adc_valon.set_ref_select(0)  # internal
ri.lo_valon.set_ref_select(1)  # external
ri.set_fft_gain(fft_gain)

# Calculate LO and baseband frequencies
roach_state = ri.state
df_tone = roach_state.adc_sample_rate / 2 ** tone_sample_exponent
df_filterbank = roach_state.adc_sample_rate / roach_state.num_filterbank_channels
minimum_integer_sweep = int(f_minimum / df_tone)
n_baseband_sweep = minimum_integer_sweep + np.arange(num_tones_sweep)
f_baseband_MHz_sweep = 1e-6 * df_tone * n_baseband_sweep
f_baseband_MHz_sweep_actual = f_baseband_MHz_sweep[:, np.newaxis]
f_lo_MHz_centers = df_lo_MHz * np.round((f0 - f_baseband_MHz_sweep.mean()) / df_lo_MHz)
f_lo_MHz_offsets = 1e-6 * df_tone * n_lo_offset
logger.info("Frequency spacing is {:.1f} kHz".format(1e3 * (f_baseband_MHz_sweep[1] -
                                                            f_baseband_MHz_sweep[0])))
logger.info("Sweep span is {:.1f} MHz".format(f_baseband_MHz_sweep.ptp()))

# Run
npd = acquire.new_npy_directory(suffix=suffix)
tic = time.time()
try:
    ri.set_tone_baseband_freqs(f_baseband_MHz_sweep_actual, nsamp=2 ** tone_sample_exponent)
    for resonator_index, f_lo_MHz_center in enumerate(f_lo_MHz_centers):
        for attenuation_index, attenuation in enumerate(attenuations):
            ri.set_dac_attenuator(attenuation)
            for offset_index, f_lo_MHz_offset in enumerate(f_lo_MHz_offsets):
                f_lo_MHz = df_lo_MHz * np.round((f_lo_MHz_center + f_lo_MHz_offset) / df_lo_MHz)
                ri.set_lo(lomhz=f_lo_MHz, chan_spacing=df_lo_MHz)
                logger.info("Set LO to {:.3f} MHz".format(f_lo_MHz))
                assert np.all(ri.adc_valon.get_phase_locks())
                assert np.all(ri.lo_valon.get_phase_locks())
                state = hw.state()
                state['resonator_index'] = resonator_index
                state['lo_valon'] = {'frequency_a': 1e6 * ri.lo_valon.get_frequency_a(),
                                     'frequency_b': 1e6 * ri.lo_valon.get_frequency_b()}
                sweep = acquire.run_loaded_sweep(ri, length_seconds=length_seconds_sweep,
                                                 tone_bank_indices=np.arange(num_tones_sweep), state=state)[0]
                f0_MHz_fit = 1e-6 * sweep.resonator.f_0
                logger.info("Fit resonance frequency is {:.3f} MHz".format(f0_MHz_fit))
                is_not_first_loop = (resonator_index > 0) or (attenuation_index > 0) or (offset_index > 0)
                f_stream_MHz = ri.add_tone_freqs(np.array([f0_MHz_fit]), overwrite_last=is_not_first_loop)
                ri.select_bank(num_tones_sweep)
                ri.select_fft_bins(np.arange(f_stream_MHz.size))
                time.sleep(wait)
                logger.info("Recording {:.1f} s stream at {:.3f} MHz".format(length_seconds_stream, f_stream_MHz[0]))
                stream = ri.get_measurement(num_seconds=length_seconds_stream, state=state)[0]
                sss = basic.SingleSweepStream(sweep=sweep, stream=stream, state=state)
                npd.write(sss)
                npd.write(ri.get_adc_measurement())
finally:
    npd.close()
    print("Wrote {}".format(npd.root_path))
    print("Elapsed time {:.0f} minutes.".format((time.time() - tic) / 60))
