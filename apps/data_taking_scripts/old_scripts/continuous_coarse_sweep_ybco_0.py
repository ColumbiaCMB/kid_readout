import matplotlib

from kid_readout.roach import baseband

matplotlib.use('agg')
import numpy as np
import time
from kid_readout.utils import data_file,sweeps

ri = baseband.RoachBaseband()
ri.initialize()
#ri.set_fft_gain(6)
f0s = np.array([130.0, 193.])
f0s.sort()
#f0s = f0s*(0.9995)

suffix = "continous_cooling"

nf = len(f0s)
atonce = 2
if nf % atonce > 0:
    print "extending list of resonators to make a multiple of ",atonce
    f0s = np.concatenate((f0s,np.arange(1,1+atonce-(nf%atonce))+f0s.max()))


nsamp = 2**16
step = 1
nstep = 120*2
f0binned = np.round(f0s*nsamp/512.0)*512.0/nsamp
offset_bins = np.arange(-(nstep+1),(nstep+1))*step

offsets = offset_bins*512.0/nsamp
offsets = np.concatenate(([offsets.min()-20e-3,],offsets,[offsets.max()+20e-3]))

print f0s
print f0s+(offsets).max()
print f0s+(offsets).min()
print len(f0s)


if False:
    from kid_readout.equipment.parse_srs import get_all_temperature_data
    while True:
        temp = get_all_temperature_data()[1][-1]
        print "mk stage at", temp
        if temp > 0.348:
            break
        time.sleep(300)
    time.sleep(600)
start = time.time()

use_fmin = True
attenlist = [63]
#attenlist = [44.0]
#attenlist = attenlist[:4]
df = data_file.DataFile(suffix=suffix)
measured_freqs = sweeps.prepare_sweep(ri,f0binned,offsets,nsamp=nsamp)
print "loaded waveforms in", (time.time()-start),"seconds"

nscans = 0
while True:
    for atten in attenlist:
        print "setting attenuator to",atten
        ri.set_dac_attenuator(atten)
        tic = time.time()
        sweep_data = sweeps.do_prepared_sweep(ri, nchan_per_step=atonce, reads_per_step=1)
        df.log_hw_state(ri)

        df.add_sweep(sweep_data)

        df.sync()
        nscans +=1
        print "%d scan done in %.1f per scan" % (nscans,(time.time()-tic))
    
df.nc.close()
    
print "completed in",((time.time()-start)/60.0),"minutes"
