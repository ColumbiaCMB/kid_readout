from time import sleep

import numpy as np

from kid_readout.equipment import hardware
from kid_readout.roach import analog, hardware_tools
from kid_readout.measurement import core, basic, acquire

acquire.show_settings()
acquire.show_git_status()
logger = acquire.get_script_logger(__file__)

# Parameters
wait = 3
length_seconds = 0.1
offset_frequency = np.linspace(10, 200, 16)
num_tone_samples = 2**15
dac_attenuation = 62  # The maximum
lo_frequency = 3000

# Hardware
conditioner = analog.HeterodyneMarkII()
hw = hardware.Hardware(conditioner)
ri = hardware_tools.r1_with_mk2()
ri.initialize(use_config=False)
ri.set_dac_attenuator(dac_attenuation)
ri.set_lo(lomhz=lo_frequency)
ri.set_tone_freqs(freqs=lo_frequency + offset_frequency, nsamp=num_tone_samples)
ri.select_fft_bins(np.arange(offset_frequency.size))

# Acquire
sweep = basic.SweepArray(core.IOList(), description="T_c measurement")
name = 'sweep'
ncf = acquire.new_nc_file(suffix='Tc')
ncf.write(sweep, name)
try:
    while True:
        sweep.stream_arrays.append(ri.get_measurement(num_seconds=length_seconds, state=hw.state()))
        sleep(wait)
finally:
    ncf.close()
    print("Wrote {}".format(ncf.root_path))
