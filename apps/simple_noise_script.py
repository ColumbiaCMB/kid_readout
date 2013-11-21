import numpy as np
import time

from kid_readout.utils import data_block,roach_interface,data_file,sweeps
#from sim900 import sim900Client

ri = roach_interface.RoachBasebandWide()
ri.initialize()
df = data_file.DataFile()
#sc = sim900Client.sim900Client()

ri.set_dac_attenuator(32)  #42 db atten is ~ -50dBm/tone for two tones

f0s = np.load('/home/gjones/workspace/apps/f0s.npy')

#f0s = np.array([  77.923828,   79.814453,  81.123,  82.548828  ])
# ,  111.611328,     117.845703,  143.736328,  152.423828])


for k in range(len(f0s)):
    offsets = np.linspace(-5e3,5e3,20)
    offsets = np.concatenate(([-40e3,-30e3,-20e3,-10e3],offsets,[10e3,20e3,30e3,40e3]))/1e6
    sweep_data = data_block.SweepData(sweep_id=k)
    def callback(block):
        sweep_data.add_block(block)
    for offs in offsets:
        print f0s[k]+offs
        sweeps.coarse_sweep(ri, freqs=np.array([70.001,f0s[k]])+offs, nsamp=2**21, nchan_per_step=2, reads_per_step=8, callback=callback)
    df.add_sweep(sweep_data)
    
    fr,s21 = sweep_data.select_index(1)
    fmin = fr[np.abs(s21).argmin()]
    print "using fmin", fmin
    ri.set_tone_freqs(np.array([70.001,fmin]),nsamp=2**21)
    ri._sync()
    time.sleep(0.2)
        
    tsg = None
    dmod,addr = ri.get_data(2048*16,demod=True)
    chids = ri.fpga_fft_readout_indexes+1
    tones = ri.tone_bins[ri.readout_selection]
    nsamp = ri.tone_nsamp
    print "saving data"
    for m in range(len(chids)):
        print m
        block = data_block.DataBlock(data = dmod[:,m], tone=tones[m], fftbin = chids[m], 
                 nsamp = nsamp, nfft = ri.nfft, wavenorm = ri.wavenorm, t0 = time.time(), fs = ri.fs)
        tsg = df.add_block_to_timestream(block, tsg=tsg)

    
    #sc.fetchDict()
    #df.add_cryo_data(sc.data)
    df.nc.sync()

df.nc.close()
