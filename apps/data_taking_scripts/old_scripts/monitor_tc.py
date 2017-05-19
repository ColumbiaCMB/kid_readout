import time

import numpy as np

from kid_readout.roach import baseband
from kid_readout.utils import data_file, data_block


ri = baseband.RoachBaseband()
#ri.initialize()
df = data_file.DataFile()
print(df.filename)

#f0s = np.load('/home/gjones/workspace/apps/first_pass_sc3x3_0813f9.npy')
#f0s = np.load('/home/gjones/workspace/readout/apps/sc3x3_0813f9_2014-02-11.npy')
#f0s.sort()
#ri.set_tone_freqs(center_freqs, nsamp=2**20)

f0s = np.load('/home/data2/resonances/2014-11-17_0813f8_140825-3_second_pass.npy')
ri._sync()

ri.set_tone_freqs(np.array([f0s[0]]),nsamp=2**20)
ri.set_dac_attenuator(63)

df.log_hw_state(ri)
df.log_adc_snap(ri)


tsg = None
try:
    while True:
        dmod,addr = ri.get_data(2)
        chids = ri.fpga_fft_readout_indexes+1
        tones = ri.tone_bins[ri.readout_selection]
        nsamp = ri.tone_nsamp
        m = 0
        block = data_block.DataBlock(data = dmod[:,m], tone=tones[m], fftbin = chids[m], 
                 nsamp = nsamp, nfft = ri.nfft, wavenorm = ri.wavenorm, t0 = time.time(), fs = ri.fs)
        tsg = df.add_block_to_timestream(block,tsg=tsg)
        df.nc.sync()
        time.sleep(60)
finally:
    df.close()
