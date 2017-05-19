import time

import numpy as np

from kid_readout.roach import baseband
from kid_readout.utils import data_file, data_block


ri = baseband.RoachBaseband()
#ri.initialize()
ri.set_fft_gain(4)
df = data_file.DataFile()

f0s = np.load('/home/gjones/workspace/apps/first_pass_sc3x3_0813f9.npy')[:4]
f0s.sort()
#ri.set_tone_freqs(center_freqs, nsamp=2**20)
ri._sync()

ri.set_tone_freqs(f0s,nsamp=2**22)
ri.set_dac_attenuator(50)

df.log_hw_state(ri)
df.log_adc_snap(ri)
ri._sync()
time.sleep(0.2)

tsg = None
t0 = time.time()
dmod,addr = ri.get_data(128)
chids = ri.fpga_fft_readout_indexes+1
tones = ri.tone_bins[ri.readout_selection]
nsamp = ri.tone_nsamp
print "saving data"
for m in range(len(chids)):
    block = data_block.DataBlock(data = dmod[:,m], tone=tones[m], fftbin = chids[m], 
             nsamp = nsamp, nfft = ri.nfft, wavenorm = ri.wavenorm, t0 = t0, fs = ri.fs)
    tsg = df.add_block_to_timestream(block,tsg=tsg)

ri.set_tone_freqs(f0s+100.0/1e6,nsamp=2**22)
ri._sync()
time.sleep(0.2)
tsg = None
dmod,addr = ri.get_data(128)
chids = ri.fpga_fft_readout_indexes+1
tones = ri.tone_bins[ri.readout_selection]
nsamp = ri.tone_nsamp
print "saving data"
for m in range(len(chids)):
    block = data_block.DataBlock(data = dmod[:,m], tone=tones[m], fftbin = chids[m], 
             nsamp = nsamp, nfft = ri.nfft, wavenorm = ri.wavenorm, t0 = time.time(), fs = ri.fs)
    tsg = df.add_block_to_timestream(block,tsg=tsg)

df.close()
    
