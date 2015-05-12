import time

import numpy as np

from kid_readout.roach import baseband
from kid_readout.utils import data_block, data_file
from sim900 import sim900Client


ri = baseband.RoachBaseband()
df = data_file.DataFile()
sc = sim900Client.sim900Client()

ri.set_adc_attenuator(31)
ri.set_dac_attenuator(26)

ri.set_tone_freqs(np.array([67.001,150.001]), nsamp=2**18)
ri._sync()

df.log_hw_state(ri)
df.log_adc_snap(ri)

tsg = None
while True:
    try:
        dmod,addr = ri.get_data(64)
        chids = ri.fpga_fft_readout_indexes+1
        tones = ri.tone_bins[ri.readout_selection]
        nsamp = ri.tone_nsamp
        for m in range(len(chids)):
            block = data_block.DataBlock(data = dmod[:,m], tone=tones[m], fftbin = chids[m], 
                     nsamp = nsamp, nfft = ri.nfft, t0 = time.time(), fs = ri.fs)
            tsg = df.add_block_to_timestream(block, tsg=tsg)
        sc.fetchDict()
        df.add_cryo_data(sc.data)
        df.nc.sync()
        time.sleep(120.)
    except KeyboardInterrupt:
        df.nc.close()
        break