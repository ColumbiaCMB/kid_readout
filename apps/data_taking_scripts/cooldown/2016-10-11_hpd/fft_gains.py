"""
Measure signal levels
"""
import time

import numpy as np

from kid_readout.roach import hardware_tools, analog
from kid_readout.measurement import acquire, basic
from kid_readout.equipment import hardware

acquire.show_settings()
acquire.show_git_status()
logger = acquire.get_script_logger(__file__)

# Parameters
suffix = 'signal_level'
fft_gains = range(10)
lo_MHz = 3000
lo_round_to_MHz = 0.1
offsets_MHz = np.array([100])
dac_attenuation = 0
tone_sample_exponent = 18
stream_length_blocks = 1
wait = 1

# Hardware
conditioner = analog.HeterodyneMarkII()
magnet = hardware.Thing(name='magnet_array', state={'orientation': 'up',
                                                    'distance_from_base_mm': 276})
hw = hardware.Hardware(conditioner, magnet)
ri = hardware_tools.r1_with_mk2()
ri.initialize(use_config=False)
# External is 1 and internal is 0
ri.adc_valon.set_ref_select(0)
ri.lo_valon.set_ref_select(1)

# Acquire
npd = acquire.new_npy_directory(suffix=suffix)
tic = time.time()
try:
    ri.set_lo(lomhz=lo_MHz, chan_spacing=lo_round_to_MHz)
    ri.set_tone_baseband_freqs(freqs=offsets_MHz, nsamp=2**tone_sample_exponent)
    ri.select_bank(0)
    ri.select_fft_bins(np.arange(offsets_MHz.size))
    ri.set_dac_attenuator(dac_attenuation)
    for fft_gain in fft_gains:
        ri.set_fft_gain(fft_gain)
        time.sleep(wait)
        npd.write(ri.get_measurement_blocks(num_blocks=stream_length_blocks, demod=False, state=hw.state()))
        npd.write(ri.get_adc_measurement())
finally:
    npd.close()
    print("Wrote {}".format(npd.root_path))
    print("Elapsed time {:.0f} minutes.".format((time.time() - tic) / 60))
