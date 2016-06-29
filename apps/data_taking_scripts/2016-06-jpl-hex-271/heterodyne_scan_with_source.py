import time

import numpy as np

from kid_readout.interactive import *
from kid_readout.measurement import acquire
from kid_readout.roach import r2heterodyne, attenuator, hardware_tools

from equipment.custom import mmwave_source
from equipment.hittite import signal_generator
from equipment.srs import lockin

logger.setLevel(logging.DEBUG)

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
source.set_attenuator_turns(3.0,3.0)
source.multiplier_input = 'hittite'
source.waveguide_twist_angle = 45
source.ttl_modulation_source = 'roach'

setup = hardware.Hardware(hittite, source,lockin)

ri = hardware_tools.r2_with_mk1(1000.)
ri.iq_delay=-1

ri.set_dac_atten(20)
ri.set_fft_gain(6)

nsamp = 2**15
step = 1
nstep = 32
#f0binned = np.round(f0s * nsamp / 512.0) * 512.0 / nsamp
offset_bins = np.arange(-(nstep), (nstep)) * step

offsets = offset_bins * 512.0 / nsamp

ri.set_modulation_output('low')

ri.set_lo(1250.)

#legacy.load_heterodyne_sweep_tones(ri,(np.arange(1,129)[None,:]*7/4.+ri.lo_frequency + offsets[:,None]),
#                                    num_tone_samples=nsamp)

state = dict(field_canceling_magnet=False,magnetic_shield=True,cryostat='starcryo')
state.update(**setup.state())

for hittite_power in np.arange(-3.0,1,.4):
    logger.info("Measuring at %.1f dBm" % hittite_power)
    hittite.set_power(hittite_power)
    tic = time.time()
    for lo in 830.+190*np.arange(0,4):
        logger.info("Measuring at LO %.1f" % lo)
        ri.set_lo(lo)
        df = acquire.new_nc_file(suffix='scan_lo_%.1f_MHz' % lo)
        ri.set_modulation_output(7)
        logger.info("autogain lockin")
        time.sleep(1)
        lockin.auto_gain(wait_until_done=True)
        time.sleep(3)
        logger.info("new sensitivity: %d values %s" % (lockin.sensitivity,str(lockin.fast_state)))
        state.update(**setup.state())
        ri.set_modulation_output('low')
        swa = acquire.run_sweep(ri, (np.arange(1, 257)[None, :] * 7 / 8. + ri.lo_frequency + offsets[:, None]),
                                num_tone_samples=nsamp, length_seconds=0.1, state=state, verbose=True)
        df.write(swa)
        df.close()
        print "elapsed:", (time.time()-tic)/60.0,'minutes'
    #time.sleep(60.)
    # while time.time() - tic < 5*60:
    #     print "waiting... %.1f min remaining" % ((5*60 - (time.time() - tic))/60)
    #     time.sleep(60)


