import time
import sys

import numpy as np

from kid_readout.roach import analog, hardware_tools
from kid_readout import *
from kid_readout.measurement.acquire import acquire,legacy_acquire
from kid_readout.measurement.acquire import hardware
from kid_readout.measurement import mmw_source_sweep, core, basic

import logging
logger = logging.getLogger('kid_readout')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s: %(asctime)s - %(pathname)s.%(funcName)s:%(lineno)d  %('
                                       'message)s'))
logger.addHandler(handler)
handler.setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)
# fg = FunctionGenerator()

ifboard = analog.Baseband()

setup = hardware.Hardware(ifboard)


ri = baseband.RoachBaseband(adc_valon=hardware_tools.roach1_valon)

f0s = np.array([104.293,
                110.995,
                141.123,
                145.652,
                170.508,
                193.611,
                195.439,
                201.111])

nsamp = 2**20
step = 1
nstep = 48
offset_bins = legacy_acquire.offset_integers[20]
offsets = offset_bins * 512.0 / nsamp

ri.set_dac_atten(36)
tic = time.time()
measured_frequencies = acquire.load_baseband_sweep_tones(ri,np.add.outer(offsets,f0s),num_tone_samples=nsamp)
logger.info("waveforms loaded in %.1f minutes", (time.time()-tic)/60.)
for dac_atten in [40,36,32,28]:
    ncf = new_nc_file(suffix='dark_%d_dB_dac' % dac_atten)
    ri.set_modulation_output('high')
    swpa = acquire.run_loaded_sweep(ri,length_seconds=0,state=setup.state(),description='dark sweep')
    logger.info("resonance sweep done %.1f min", (time.time()-tic)/60.)
    ncf.write(swpa)
    #print "sweep written", (time.time()-tic)/60.
    current_f0s = []
    for sidx in range(8):
        swp = swpa.sweep(sidx)
        res = lmfit_resonator.LinearResonatorWithCable(swp.frequency,swp.s21_points,swp.s21_points_error)
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
    ri.select_bank(ri.tone_bins.shape[0]-1)
    ri.select_fft_bins(range(32))
    meas = ri.get_measurement(num_seconds=30., state=setup.state(),description='source off stream')
    ncf.write(meas)

    logger.info("dac_atten %f done in %.1f minutes" % (dac_atten, (time.time()-tic)/60.))
    ncf.close()