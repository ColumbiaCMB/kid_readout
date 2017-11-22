from time import sleep

import numpy as np

from kid_readout.settings import CRYOSTAT
from kid_readout.roach import hardware_tools
from kid_readout.measurement import core, basic, acquire

# Parameters
wait = 60
length_seconds = 0.1
offset_frequency = np.linspace(10, 200, 16)
num_tone_samples = 2**15
lo_frequency = 3000

# Hardware
ri = hardware_tools.r2_with_mk2()
ri.set_dac_atten(50)
ri.set_fft_gain(4)
ri.set_lo(lomhz=lo_frequency)
ri.set_tone_freqs(freqs=lo_frequency + offset_frequency, nsamp=num_tone_samples)
ri.select_fft_bins(np.arange(offset_frequency.size))

# State
state = {'cryostat': CRYOSTAT,
         'canceling_magnet': {'orientation': 'up',
                              'distance_from_base_mm': 25}}

# Acquire
sweep = basic.SweepArray(core.IOList(), state=state, description="T_c measurement")
name = 'sweep'
npd = acquire.new_npy_directory(suffix='Tc')
npd.write(sweep, name)
try:
    while True:
        sweep.stream_arrays.append(ri.get_measurement(num_seconds=length_seconds, state=state))
        print("Wrote stream {}".format(len(sweep.stream_arrays)))
        sleep(wait)
finally:
    npd.close()
    print("Wrote {}".format(npd.root_path))
