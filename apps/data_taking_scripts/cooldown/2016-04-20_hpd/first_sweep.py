import time

import numpy as np
try:
    from tqdm import tqdm as progress
except ImportError:
    progress = list

from kid_readout.roach import hardware_tools, analog
from kid_readout.measurement import acquire
from kid_readout.equipment import hardware
from kid_readout.settings import CRYOSTAT
if CRYOSTAT.lower() == 'hpd':
    from kid_readout.equipment import hpd_temps as temps
elif CRYOSTAT.lower() == 'starcryo':
    from kid_readout.equipment import starcryo_temps as temps
else:
    raise ValueError("Unknown cryostat: {}".format(repr(CRYOSTAT)))

# Parameters
f0_MHz = np.array([2201.8, 2378.8, 2548.9, 2731.5, 2905.1, 3416.0])
num_offsets = 100
span_MHz = 20
minimum_MHz = 10
round_to_MHz = 2
lo_MHz = round_to_MHz * np.round((f0_MHz - span_MHz / 2 - minimum_MHz) / round_to_MHz)
offsets_MHz = np.linspace(minimum_MHz, minimum_MHz + span_MHz, num_offsets)
num_tone_samples = 2**19
length_seconds = 0.1

# Hardware
conditioner = analog.HeterodyneMarkII()
magnet = hardware.Thing('canceling_magnet',
                        {'orientation': 'up',
                         'distance_from_base_mm': 25})
hw = hardware.Hardware(conditioner, magnet)
ri = hardware_tools.r2_with_mk2()
ri.set_dac_atten(40)
ri.set_fft_gain(4)
ri.set_modulation_output('high')

# Run
ncf = acquire.new_nc_file(suffix='sweep')
tic = time.time()
try:
    for lo in progress(lo_MHz):
        state = hw.state()
        state['temperature'] = {'package': temps.get_temperature_at(time.time())}
        tone_banks = np.array([np.array([f]) for f in lo + offsets_MHz])
        ri.set_lo(lomhz=lo, chan_spacing=round_to_MHz)
        sweep = acquire.run_sweep(ri, tone_banks=tone_banks, num_tone_samples=num_tone_samples,
                                  length_seconds=length_seconds, state=state)
        ncf.write(sweep)
finally:
        ncf.close()
        print("Wrote {}".format(ncf.root_path))
        print("Elapsed time {:.0f} minutes.".format((time.time() - tic) / 60))
