from __future__ import division

import time

import numpy as np

try:
    from tqdm import tqdm
    make_iterable = tqdm
except ImportError:
    make_iterable = list

from kid_readout.roach import calculate, hardware_tools
from kid_readout.measurement import core, basic
from kid_readout.measurement import acquire

tic = time.time()

# Input parameters
f_start = 2000e6
f_stop = 2020e6
stream_seconds = 0.01
f_lo_resolution = 10e3  # 5 kHz is the limit of the LO in divide-by-1 mode.
num_tones_maximum = 2 ** 7  # This currently has to be a power of two.
f_minimum = 5e6  # The minimum spacing from the LO frequency.
f_maximum = 200e6  # The filter turns on not too far above this frequency.
num_filterbanks_per_tone = 16  # The minimum spacing between filter banks read out simultaneously.

# Hardware
ri = hardware_tools.r1_with_mk2()
ri.set_dac_atten(40)
ri.set_fft_gain(4)
ri.set_modulation_output('high')
roach_state = ri.state  # Cache this for speed; its values are in Hz.

# Calculated parameters
f_tone = num_filterbanks_per_tone * calculate.stream_sample_rate(roach_state)  # The minimum spacing between tones
print("Minimum tone spacing: {:.0f} kHz.".format(1e-3 * f_tone))
tone_sample_exponent = np.ceil(np.log2(roach_state.adc_sample_rate / f_lo_resolution))
num_tone_samples = 2 ** tone_sample_exponent
f_roach_resolution = roach_state.adc_sample_rate / num_tone_samples
print("LO resolution: {:.2f} kHz.".format(1e-3 * f_lo_resolution))
print("ROACH resolution: {:.2f} kHz.".format(1e-3 * f_roach_resolution))
num_tones = num_tones_maximum
while num_tones * f_tone > f_maximum - f_minimum:
    num_tones /= 2
f_block = num_tones * f_tone  # The spacing between blocks
print("Using {} simultaneous tones spanning {:.1f} MHz.".format(num_tones, 1e-6 * f_block))
f_center_MHz = 1e-6 * np.arange(f_start, f_stop, f_block)
print("Block center frequencies in MHz: {}".format(', '.join(['{:.1f}'.format(f) for f in f_center_MHz])))
f_lo_offset_MHz = 1e-6 * (np.arange(-f_tone / 2, f_tone / 2, f_lo_resolution) - f_minimum)
# This order covers the entire band then comes back to fill in the gaps.
f_lo_MHz = (f_center_MHz[np.newaxis, :] + f_lo_offset_MHz[:, np.newaxis]).flatten()
n_minimum = np.ceil(f_minimum / f_roach_resolution)
step = np.floor(f_tone / f_roach_resolution)
n_baseband = n_minimum + step * np.arange(num_tones)
f_baseband_MHz = 1e-6 * n_baseband * f_roach_resolution

# State
state = {'cryostat': 'hpd',
         'canceling_magnet': {'orientation': 'up',
                              'distance_from_base_mm': 25}}

# Acquire
ri.set_tone_baseband_freqs(freqs=f_baseband_MHz, nsamp=num_tone_samples)
ri.select_fft_bins(np.arange(f_baseband_MHz.size))
sweep = basic.SweepArray(core.IOList(), state=state, description=acquire.script_code())
ncf = acquire.new_nc_file(suffix='lo_scan_test')
ncf.write(sweep)
try:
    for f_lo in make_iterable(f_lo_MHz):
        ri.set_lo(lomhz=f_lo, chan_spacing=1e-6 * f_lo_resolution)
        sweep.stream_arrays.append(ri.get_measurement(num_seconds=stream_seconds, state=state))
finally:
    ncf.close()
    print("Wrote {}".format(ncf.root_path))
    print("Elapsed time {:.0f} minutes.".format((time.time() - tic) / 60))
