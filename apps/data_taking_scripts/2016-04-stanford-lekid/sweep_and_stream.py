import time
import sys

import numpy as np

from kid_readout.roach import analog, hardware_tools
from kid_readout.interactive import *
from kid_readout.measurement import acquire
from kid_readout.measurement.legacy import legacy_acquire
from kid_readout.equipment import hardware
from kid_readout.measurement import mmw_source_sweep, core, basic

from equipment.keithley.sourcemeter import SourceMeter

logger.setLevel(logging.DEBUG)
# fg = FunctionGenerator()

sourcemeter = SourceMeter(serial_device='/dev/ttyUSB2')
ifboard = analog.Baseband()

setup = hardware.Hardware(ifboard,sourcemeter)


ri = baseband.RoachBaseband()
ri.initialize()
ri.set_fft_gain(6)

f0s = np.array([104.293,
                110.995,
                141.123,
                145.652,
                170.508,
                193.611,
                195.439,
                201.111])

current_f0s = f0s
nsamp = 2**20
step = 1
nstep = 48
offset_bins = legacy_acquire.offset_integers[20]
offsets = offset_bins * 512.0 / nsamp

tic = time.time()
measured_frequencies = acquire.load_baseband_sweep_tones(ri,np.add.outer(offsets,current_f0s),num_tone_samples=nsamp)
logger.info("waveforms loaded in %.1f minutes", (time.time()-tic)/60.)

logger.info(sourcemeter.identify())
sourcemeter.set_current_source()
sourcemeter.set_current_amplitude(0.0)
sourcemeter.enable_output()
ri.set_modulation_output('low')

#ri.set_dac_atten(36)
first = True
#while True: #for dac_atten in [40,36,32,28,24,20]:
for led_current in np.array([0.0,0.1,1,2,4,8])*1e-3:
    sourcemeter.set_current_amplitude(led_current)
    dac_atten = 28
    ri.set_dac_atten(dac_atten)
    ncf = new_nc_file(suffix='dark_%d_dB_dac_%.1f_mA_led' % (dac_atten,led_current*1e3))
    ri.set_modulation_output('high')

    swpa = acquire.run_loaded_sweep(ri,length_seconds=0,state=setup.state(),description='light sweep')
    logger.info("resonance sweep done %.1f min", (time.time()-tic)/60.)
    ri.set_modulation_output('low')
    ncf.write(swpa)
    #print "sweep written", (time.time()-tic)/60.
    if first:
        current_f0s = []
        for sidx in range(8):
            swp = swpa.sweep(sidx)
            res = lmfit_resonator.LinearResonatorWithCable(swp.frequency, swp.s21_point, swp.s21_point_error)
            print res.f_0, res.Q, res.current_result.redchi, (f0s[sidx]*1e6-res.f_0)
            if False: #sidx not in [15,17] and np.abs(res.f_0 - f0s[sidx]*1e6) > 200e3:
                current_f0s.append(f0s[sidx]*1e6)
                print "using original frequency for ",f0s[sidx]
            else:
                current_f0s.append(res.f_0)
        logger.info("fits complete %.1f min", (time.time()-tic)/60.)
        current_f0s = np.array(current_f0s)/1e6
        current_f0s.sort()
        if np.any(np.diff(current_f0s)<0.1):
            logger.error("problematic resonator collision: %s",current_f0s)
            logger.info("deltas: %s", np.diff(current_f0s))
        ri.add_tone_freqs(current_f0s,overwrite_last=True)
        first = False
    ri.select_bank(ri.tone_bins.shape[0]-1)
    ri.select_fft_bins(range(8))
    ri.set_modulation_output('low')
    state = setup.state()
    logger.info("Taking source off stream")
    meas = ri.get_measurement(num_seconds=30., state=state,description='source off stream')
    ncf.write(meas)

    if True:
        ri.set_modulation_output('high')
        state = setup.state()
        logger.info("Taking source on stream")
        meas = ri.get_measurement(num_seconds=30., state=state,description='source on stream')
        ncf.write(meas)

        ri.set_modulation_output(7)
        state = setup.state()
        logger.info("Taking modulated stream")
        meas = ri.get_measurement(num_seconds=4., state=state,description='source modulated stream')
        ncf.write(meas)
        ri.set_modulation_output('low')

    logger.info("led_current %f done in %.1f minutes" % (led_current, (time.time()-tic)/60.))
    ncf.close()
#    7/0
#    logger.info("waiting 5 minutes")
#    time.sleep(300)