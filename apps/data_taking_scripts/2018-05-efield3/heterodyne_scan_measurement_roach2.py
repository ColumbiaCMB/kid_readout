import time

import numpy as np

from kid_readout.interactive import *
from kid_readout.measurement import acquire
from kid_readout.roach import r2heterodyne, attenuator, hardware_tools


logger.setLevel(logging.DEBUG)

setup = hardware.Hardware()

ri = hardware_tools.r2h14_with_mk2(initialize=True, use_config=False)
ri.iq_delay=-1

dac_atten = 20
ri.set_dac_atten(dac_atten)
ri.set_fft_gain(6)

nsamp = 2**16
step = 1
nstep = 64
offset_bins = np.arange(-(nstep), (nstep)) * step

offsets = offset_bins * 512.0 / nsamp

ri.set_modulation_output('high')

#ri.set_lo(3200.)
ri.set_lo(2370.)

state = dict(magnetic_shield = 'on', cryostat='starcryo')
state.update(other=setup.state())

tic = time.time()

#for lo in 2200.+190*np.arange(0,2):

for lo in 2200. + 190 * np.arange(0, 2):
    logger.info("Measuring at LO %.1f" % lo)
    df = acquire.new_nc_file(suffix='scan_lo_%.1f_MHz_atten_%.1f_dB' % (lo, dac_atten))
    ri.set_lo(lo)
    state.update(other=setup.state(fast=True))
    swa = acquire.run_sweep(ri, (np.arange(1, 257)[None, :] * 7 / 8. + ri.lo_frequency + offsets[:, None]),
                            num_tone_samples=nsamp, length_seconds=0.2, state=state, verbose=True)
    df.write(swa)
    df.close()
    print "elapsed:", (time.time()-tic)/60.0,'minutes'