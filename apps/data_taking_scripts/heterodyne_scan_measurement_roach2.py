import numpy as np
import time
from kid_readout.roach import r2heterodyne, attenuator, hardware_tools
from kid_readout.measurement.acquire import acquire
from kid_readout.measurement.io.helpers import new_nc_file

from kid_readout.measurement.acquire import hardware

#setup = hardware.Hardware()


ri = r2heterodyne.Roach2Heterodyne(adc_valon=hardware_tools.roach2_valon, lo_valon=hardware_tools.mark2_valon,
                                   attenuator=attenuator.Attenuator())

ri.set_dac_atten(40)
ri.set_fft_gain(4)

nsamp = 2**15
step = 1
nstep = 32
#f0binned = np.round(f0s * nsamp / 512.0) * 512.0 / nsamp
offset_bins = np.arange(-(nstep), (nstep)) * step

offsets = offset_bins * 512.0 / nsamp

ri.set_modulation_output('high')

ri.set_lo(1250.)

#acquire.load_heterodyne_sweep_tones(ri,(np.arange(1,129)[None,:]*7/4.+ri.lo_frequency + offsets[:,None]),
#                                    num_tone_samples=nsamp)

state = dict(field_canceling_magnet=True,cryostat='hpd')

tic = time.time()
for lo in 2010.+190*np.arange(0,10):
    print "lo:",lo
    df = new_nc_file(suffix='scan_lo_%.1f_MHz' % lo)
    ri.set_lo(lo)
    swa = acquire.run_sweep(ri,(np.arange(1,257)[None,:]*7/8.+ri.lo_frequency + offsets[:,None]),
                                    num_tone_samples=nsamp,length_seconds=0.2,state=state)
    df.write(swa)
    df.close()
    print "elapsed:", (time.time()-tic)/60.0,'minutes'

