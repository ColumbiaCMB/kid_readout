import time

import numpy as np

from kid_readout.equipment import hardware
from kid_readout.roach import analog, hardware_tools
from kid_readout.measurement import acquire

acquire.show_settings()
acquire.show_git_status()
logger = acquire.get_script_logger(__file__)

# Parameters
suffix = 'data_rate'
length_seconds = 1
num_tone_samples = 2**15
dac_attenuation = 62  # The maximum
f_lo_MHz = 2200
df_lo_MHz = 2
f_min_MHz = 10
f_max_MHz = 250
num_tones = [1, 2, 4, 8, 16, 32, 64]

# Hardware
conditioner = analog.HeterodyneMarkII()
magnet = hardware.Thing(name='magnet_array', state={'orientation': 'up',
                                                    'distance_from_base_mm': 276})
hw = hardware.Hardware(conditioner, magnet)
ri = hardware_tools.r1h11_with_mk2(initialize=True, use_config=False)
ri.adc_valon.set_ref_select(0)  # internal
ri.lo_valon.set_ref_select(1)  # external
ri.set_lo(lomhz=f_lo_MHz, chan_spacing=df_lo_MHz)
logger.info("Set LO to {:.3f} MHz".format(f_lo_MHz))
assert np.all(ri.adc_valon.get_phase_locks())
assert np.all(ri.lo_valon.get_phase_locks())

# Run
npd = acquire.new_npy_directory(suffix=suffix)
tic = time.time()
try:
    for n in num_tones:
        f_baseband = np.linspace(f_min_MHz, f_max_MHz, n)
        ri.set_tone_baseband_freqs(freqs=f_baseband, nsamp=num_tone_samples)
        logger.info("Reading {:d} tones simultaneously.".format(n))
        npd.write(ri.get_measurement(num_seconds=length_seconds, state=hw.state()))
finally:
    npd.close()
    print("Wrote {}".format(npd.root_path))
    print("Elapsed time {:.0f} minutes.".format((time.time() - tic) / 60))
