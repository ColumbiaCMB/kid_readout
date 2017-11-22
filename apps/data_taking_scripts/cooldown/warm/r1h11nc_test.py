"""

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
suffix = 'r1h11a'
attenuation = 10
f_lo_MHz = 3000
df_lo_MHz = 10
num_tones = 4
f_baseband_MHz = np.linspace(10, 200, num_tones)
tone_sample_exponent = 15
length_seconds = 10

# Hardware
conditioner = analog.HeterodyneMarkII()
hw = hardware.Hardware(conditioner)
ri = hardware_tools.r1h11nc_with_mk2(initialize=True, use_config=False)
ri.adc_valon.set_ref_select(0)
ri.lo_valon.set_ref_select(1)
ri.set_dac_attenuator(attenuation)
ri.set_lo(f_lo_MHz, chan_spacing=df_lo_MHz)
ri.set_tone_baseband_freqs(f_baseband_MHz, nsamp=2 ** tone_sample_exponent)

# Run
npd = acquire.new_npy_directory(suffix=suffix)
tic = time.time()
try:
    tools.optimize_fft_gain(ri)
    npd.write(ri.get_measurement(num_seconds=length_seconds, state=hw.state()))
    npd.write(ri.get_adc_measurement())
finally:
    npd.close()
    print("Wrote {}".format(npd.root_path))
    print("Elapsed time {:.0f} minutes.".format((time.time() - tic) / 60))
