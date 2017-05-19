import time

import numpy as np

from kid_readout.interactive import *
from kid_readout.measurement import acquire
from kid_readout.roach import r2heterodyne, attenuator, hardware_tools

from equipment.custom import mmwave_source
from equipment.hittite import signal_generator
from equipment.srs import lockin

hittite = signal_generator.Hittite(ipaddr='192.168.0.200')
hittite.set_power(0)
hittite.on()
hittite.set_freq(148e9/12.)

lockin = lockin.Lockin(LOCKIN_SERIAL_PORT)
tic = time.time()
print lockin.identification
print time.time()-tic
tic = time.time()
print lockin.fast_state
print time.time()-tic
source = mmwave_source.MMWaveSource()
source.set_attenuator_turns(6.0,6.0)
source.multiplier_input = 'hittite'
source.waveguide_twist_angle = 0
source.ttl_modulation_source = 'roach'

setup = hardware.Hardware(hittite, source,lockin)

logger.setLevel(logging.DEBUG)


# lockin = lockin.Lockin(LOCKIN_SERIAL_PORT)
# tic = time.time()
# print lockin.identification
# print time.time()-tic
# tic = time.time()
# print lockin.state(measurement_only=True)
# print time.time()-tic


ri = hardware_tools.r2_with_mk1()
ri.iq_delay=-1

ri.set_dac_atten(0)
ri.set_fft_gain(6)

nsamp = 2**15
step = 1
nstep = 32
#f0binned = np.round(f0s * nsamp / 512.0) * 512.0 / nsamp
offset_bins = np.arange(-(nstep), (nstep)) * step

offsets = offset_bins * 512.0 / nsamp

ri.set_modulation_output(7)

ri.set_lo(1250.)

#legacy.load_heterodyne_sweep_tones(ri,(np.arange(1,129)[None,:]*7/4.+ri.lo_frequency + offsets[:,None]),
#                                    num_tone_samples=nsamp)

state = dict(magnetic_shield = 'on', cryostat='starcryo')
state.update(other=setup.state())

tic = time.time()
for lo in 790.+190*np.arange(0,5):
    logger.info("Measuring at LO %.1f" % lo)
    df = acquire.new_nc_file(suffix='scan_lo_%.1f_MHz' % lo)
    ri.set_lo(lo)
    state.update(other=setup.state(fast=True))
    swa = acquire.run_sweep(ri, (np.arange(1, 257)[None, :] * 7 / 8. + ri.lo_frequency + offsets[:, None]),
                            num_tone_samples=nsamp, length_seconds=0.2, state=state, verbose=True)
    df.write(swa)
    df.close()
    print "elapsed:", (time.time()-tic)/60.0,'minutes'

