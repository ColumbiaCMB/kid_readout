"""
Record data at different readout powers at regular intervals.
"""
from __future__ import division
import time

import numpy as np

from kid_readout.roach import r2baseband, analog
from kid_readout.measurement import acquire, core, basic
from kid_readout.equipment import hardware
from kid_readout.settings import ROACH2_IP, ROACH2_VALON

acquire.show_settings()
logger = acquire.get_script_logger(__file__)

# Parameters
dummy_MHz = 170
f0_MHz = np.array([78.350, 116.164, 160.815, dummy_MHz])  # The frequencies used for this cooldown
frequency = np.array([79, 117, 161, 170])  # Round up to avoid the resonances as they shift down
attenuations = [41, 47, 53, 59]
fft_gains = [6, 7, 8, 9]
tone_sample_exponent = 15
length_seconds = 0.1
wait = 10

# Hardware
conditioner = analog.Baseband()
shield = hardware.Thing(name='magnetic_shield_bucket', state={})
hw = hardware.Hardware(conditioner, shield)
ri = r2baseband.Roach2Baseband(roachip=ROACH2_IP, adc_valon=ROACH2_VALON)
ri.set_modulation_output('high')
ri.set_tone_freqs(freqs=frequency, nsamp=2**tone_sample_exponent)

# Run
sweeps = [basic.SweepArray(core.IOList(), description="T_c {:.1f} dB".format(attenuation))
          for attenuation in attenuations]
ncf = acquire.new_nc_file(suffix='Tc')
for sweep in sweeps:
    ncf.write(sweep)
try:
    while True:
        for sweep, attenuation, fft_gain in zip(sweeps, attenuations, fft_gains):
            ri.set_fft_gain(fft_gain)
            ri.set_dac_attenuator(attenuation)
            sweep.stream_arrays.append(ri.get_measurement(num_seconds=length_seconds, state=hw.state()))
            time.sleep(wait)
finally:
    ncf.close()
    print("Wrote {}".format(ncf.root_path))
