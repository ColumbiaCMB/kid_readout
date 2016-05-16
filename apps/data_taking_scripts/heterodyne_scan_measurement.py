import time

import numpy as np

import kid_readout.equipment.hittite_controller
import kid_readout.roach.heterodyne
from kid_readout.measurement import acquire

hc = kid_readout.equipment.hittite_controller.hittiteController(addr='192.168.0.200')

hc.set_freq(12.95833e9)
hc.on()
hc.set_power(0)




ri = kid_readout.roach.heterodyne.RoachHeterodyne(adc_valon='/dev/ttyUSB0')

ri.initialize()#use_config=False)
ri.iq_delay = 0

ri.set_dac_atten(30)
ri.set_fft_gain(4)

nsamp = 2**16
step = 1
nstep = 128
#f0binned = np.round(f0s * nsamp / 512.0) * 512.0 / nsamp
offset_bins = np.arange(-(nstep), (nstep)) * step

offsets = offset_bins * 512.0 / nsamp

ri.set_modulation_output(7)

ri.set_lo(1250.)

acquire.load_heterodyne_sweep_tones(ri, (np.arange(1, 129)[None, :] * 7 / 4. + ri.lo_frequency + offsets[:, None]),
                                    num_tone_samples=nsamp)

state = dict(mmw_atten_turns = (7.,7.))

tic = time.time()
for lo in 1010+190*np.arange(0,6):
    print "lo:",lo
    df = acquire.new_nc_file(suffix='scan_lo_%.1f_MHz_mmw_modulated_7_7_turns' % lo)
    ri.set_lo(lo)
    swa = acquire.run_multipart_sweep(ri, length_seconds=1.0, state=state, verbose=True)
    df.write(swa)
    df.close()
    print "elapsed:", (time.time()-tic)/60.0,'minutes'

