import matplotlib

from kid_readout.roach import baseband

matplotlib.use('agg')
import numpy as np
import time
import sys
from kid_readout.utils import data_file,sweeps

ri = baseband.RoachBaseband()
#ri.initialize(use_config=False)
#f0s = np.load('/home/gjones/workspace/apps/f8_fit_resonances.npy')
#f0s = np.load('/home/gjones/workspace/apps/first_pass_sc3x3_0813f9.npy')
#f0s = np.load('/home/gjones/workspace/apps/sc5x4_0813f10_first_pass.npy')#[:4]
#f0s = np.load('/home/gjones/workspace/readout/apps/sc3x3_0813f9_2014-02-11.npy')
#f0s = np.load('/home/gjones/workspace/readout/apps/sc3x3_0813f5_2014-02-27.npy')
f0s = np.array([157.315, 201.49])
f0s.sort()
#f0s = f0s*(1-4e-5)

nf = len(f0s)
atonce = 2
if nf % atonce > 0:
    print "extending list of resonators to make a multiple of ",atonce
    f0s = np.concatenate((f0s,np.arange(1,1+atonce-(nf%atonce))+f0s.max()))

offsets = np.linspace(-4882.8125,4638.671875,20)#[5:15]
#offsets = np.concatenate(([-40e3,-20e3],offsets,[20e3,40e3]))/1e6
offsets = np.concatenate(([-40e3],offsets,[40e3]))/1e6
offsets = offsets*4

print f0s
print len(f0s)
start = time.time()

nsamp = 2**18
step = 1
nstep = 100
offset_bins = np.arange(-(nstep+1),(nstep+1))*step

offsets = offset_bins*512.0/nsamp
f0binned = np.round(f0s*nsamp/512.0)*512.0/nsamp

measured_freqs = sweeps.prepare_sweep(ri,f0binned,offsets,nsamp=2**18)
print "loaded waveforms in", (time.time()-start),"seconds"

sys.stdout.flush()
time.sleep(1)
n =0
atten_list = [36]#np.linspace(27,34,8)#[30]#[35.5,33.5,46.5,43.5,40.5,37.5]
for atten in atten_list:
    df = data_file.DataFile()
    ri.set_dac_attenuator(atten)
    while True:
        sweep_data = sweeps.do_prepared_sweep(ri, nchan_per_step=atonce, reads_per_step=2)
        df.add_sweep(sweep_data)
        df.sync()

        df.log_hw_state(ri)
        df.nc.sync()
        n = n + 1
        print "done measurement", n
        time.sleep(60)
    df.nc.close()
    
print "completed in",((time.time()-start)/60.0),"minutes"
