"""
Measure several resonators per LO frequency and record SweepStreamArrays.
"""
import time

import numpy as np

from kid_readout.roach import hardware_tools, analog
from kid_readout.measurement import acquire, basic
from kid_readout.equipment import hardware

logger = acquire.get_script_logger(__file__)

# Parameters
suffix = 'interactive'
low_f0_MHz = np.array([2254.837, 2326.842, 2483.490, 2580])
high_f0_MHz = np.array([3313.270, 3378.300, 3503.600, 3524.435])
f0_MHz = high_f0_MHz[0]
f_minimum = 10e6  # Keep the tones away from the LO by at least this frequency.
f_stream_offset_MHz = 10  # Set a second tone away from the resonance by this amount
df_lo_MHz = 0.1
sweep_interval = 6
dac_attenuation = 33
fft_gain = 0
tone_sample_exponent = 18
sweep_length_seconds = 0.1
num_sweep_tones = 255

# Hardware
conditioner = analog.HeterodyneMarkII()
magnet = hardware.Thing(name='magnet_array', state={'orientation': 'up',
                                                    'distance_from_base_mm': 276})
hw = hardware.Hardware(conditioner, magnet)
ri = hardware_tools.r1h11_with_mk2(initialize=True, use_config=False)
ri.set_dac_attenuator(dac_attenuation)
ri.set_fft_gain(fft_gain)

# Calculate LO and baseband frequencies
f_resolution = ri.state.adc_sample_rate / 2**tone_sample_exponent
minimum_integer = int(f_minimum / f_resolution)
offset_integers = minimum_integer + sweep_interval * np.arange(num_sweep_tones)
offset_frequencies_MHz = 1e-6 * f_resolution * offset_integers
f_lo_MHz = df_lo_MHz * np.round((f0_MHz - offset_frequencies_MHz.mean()) / df_lo_MHz)
logger.info("Frequency spacing is {:.1f} kHz".format(1e-3 * sweep_interval * f_resolution))
logger.info("Sweep span is {:.1f} MHz".format(offset_frequencies_MHz.ptp()))
ri.set_lo(lomhz=f_lo_MHz, chan_spacing=df_lo_MHz)
logger.info("Set LO to {:.3f} MHz".format(f_lo_MHz))
ri.set_tone_baseband_freqs(offset_frequencies_MHz[:, np.newaxis], nsamp=2 ** tone_sample_exponent)


sweep_array = acquire.run_loaded_sweep(ri, length_seconds=sweep_length_seconds,
                                       tone_bank_indices=np.arange(num_sweep_tones))
fit_f0_MHz = 1e-6 * sweep_array[0].resonator.f_0
logger.info("Fit resonance frequency in MHz is {}".format(fit_f0_MHz))
f_stream_MHz = ri.add_tone_freqs(freqs=np.array([fit_f0_MHz]))
ri.select_bank(num_sweep_tones)
ri.select_fft_bins(np.array([0]))
