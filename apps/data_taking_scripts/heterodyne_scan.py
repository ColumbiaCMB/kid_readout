import numpy as np
import time
import kid_readout.roach.heterodyne
import kid_readout.utils.sweeps

import kid_readout.utils.data_file

import kid_readout.equipment.hittite_controller

#hc = kid_readout.equipment.hittite_controller.hittiteController(addr='192.168.0.200')

#hc.set_freq(12.95833e9)
#hc.on()
#hc.set_power(0)




ri = kid_readout.roach.heterodyne.RoachHeterodyne(adc_valon='/dev/ttyUSB0')

ri.initialize()#use_config=False)

ri.set_dac_atten(40)
ri.set_fft_gain(4)

nsamp = 2**16
step = 1
nstep = 128
#f0binned = np.round(f0s * nsamp / 512.0) * 512.0 / nsamp
offset_bins = np.arange(-(nstep), (nstep)) * step

offsets = offset_bins * 512.0 / nsamp

ri.set_modulation_output(7)

ri.set_lo(1250.)
frq = kid_readout.utils.sweeps.prepare_sweep(ri,np.arange(1,129)*7/4.+ri.lo_frequency,offsets,nsamp=nsamp)

tic = time.time()
for lo in 1010+190*np.arange(0,6):
    print "lo:",lo
    df = kid_readout.utils.data_file.DataFile(suffix='scan_lo_%.1f_MHz_mmw_on' % lo)
    df.nc.mmw_atten_turns = (4.,4.)
    ri.set_lo(lo)
    df.log_hw_state(ri)
    df.log_adc_snap(ri)
    swp = kid_readout.utils.sweeps.do_prepared_sweep(ri,nchan_per_step=32,reads_per_step=2)
    df.add_sweep(swp)
    df.close()
    print "elapsed:", (time.time()-tic)/60.0,'minutes'

