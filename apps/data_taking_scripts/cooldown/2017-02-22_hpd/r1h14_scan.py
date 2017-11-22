"""
Measure resonators, one at a time, with the readout tone centered in the filterbank bin.
"""
from __future__ import division
import time

import numpy as np

from kid_readout.roach import analog, hardware_tools, tools
from kid_readout.measurement import acquire
from kid_readout.equipment import hardware

acquire.show_settings()
acquire.show_git_status()
logger = acquire.get_script_logger(__file__)


# Parameters
suffix = 'led_on'
attenuation = 20
fft_gain = 4
df_baseband_target = 15e3
f_start = 2.4e9
f_stop = 3.1e9
overlap_fraction = 0.5
f_baseband_minimum = 10e6  # Keep the tones away from the LO by at least this frequency
f_baseband_maximum = 200e6  # Keep the tones below this frequency
length_seconds = 0  # Take the minimum amount of data, in this case one block
filterbank_bin_separation = 2  # The minimum number of PFB bins that separate tones
df_lo = 2.5e3  # The minimum
num_tones_maximum = 128  # Imposed by the data streaming rate

# Hardware
conditioner = analog.HeterodyneMarkII()
shield = hardware.Thing(name='magnetic_shield_pocket', state={'orientation': 'horizontal'})
led = hardware.Thing(name='led', state={'bias_current_mA': 10})
hw = hardware.Hardware(conditioner, shield)
ri = hardware_tools.r1h14_with_mk2(initialize=True, use_config=False)
ri.adc_valon.set_ref_select(0)
ri.lo_valon.set_ref_select(1)
ri.set_dac_attenuator(attenuation)
ri.set_fft_gain(fft_gain)

# Calculate sweep parameters, LO and baseband sweep frequencies
ri_state = ri.state
tone_sample_exponent = int(np.round(np.log2(ri_state.adc_sample_rate / df_baseband_target)))
df_baseband = ri_state.adc_sample_rate / 2 ** tone_sample_exponent
logger.info("Baseband resolution is {:.0f} Hz using 2^{:d} samples".format(df_baseband, tone_sample_exponent))
num_sweep_tones = np.min([int((f_baseband_maximum - f_baseband_minimum) / df_baseband),
                          ri.max_num_waveforms(2 ** tone_sample_exponent)])
df_sweep = df_baseband * num_sweep_tones
logger.info("Using {:d} tones spanning {:.1f} MHz".format(num_sweep_tones, 1e-6 * df_sweep))
df_sweep_start = (1 - overlap_fraction) * df_sweep
num_sweeps = np.int(np.ceil((f_stop - f_start) / df_sweep_start))
logger.info("Dividing {:.0f} MHz span into {:d} sweeps with {:.2f} overlap fraction".format(
        1e-6 * (f_stop - f_start), num_sweeps, overlap_fraction))
df_filterbank = ri_state.adc_sample_rate / ri_state.num_filterbank_channels
df_tone_minimum = df_filterbank * filterbank_bin_separation
logger.info("Separating tones by {:d} filterbank bins requires minimum separation of {:.3f} MHz".format(
        filterbank_bin_separation, 1e-6 * df_tone_minimum))
num_tones_per_step = np.min([num_tones_maximum, np.int(df_sweep / df_tone_minimum)])
num_steps = np.int(num_sweep_tones / num_tones_per_step)
logger.info("Dividing each sweep into {:d} steps of {:d} tones each".format(num_steps, num_tones_per_step))
f_baseband_all = f_baseband_minimum + df_baseband * (num_steps * np.arange(num_tones_per_step)[np.newaxis, :] +
                                                     np.arange(num_steps)[:, np.newaxis])
ri.set_tone_baseband_freqs(freqs=1e-6 * f_baseband_all, nsamp=2 ** tone_sample_exponent)
f_lo_all = f_start - f_baseband_minimum + df_sweep_start * np.arange(num_sweeps)

# Run
npd = acquire.new_npy_directory(suffix=suffix)
tic = time.time()
try:
    for lo_index, f_lo in enumerate(f_lo_all):
        assert np.all(ri.adc_valon.get_phase_locks())
        tools.set_and_attempt_external_phase_lock(ri=ri, f_lo = 1e-6 * f_lo, f_lo_spacing=1e-6 * df_lo)
        npd.write(acquire.run_loaded_sweep(ri=ri, length_seconds=length_seconds, state=hw.state()))
        npd.write(ri.get_adc_measurement())
finally:
    npd.close()
    print("Wrote {}".format(npd.root_path))
    print("Elapsed time {:.0f} minutes.".format((time.time() - tic) / 60))
