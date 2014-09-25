import matplotlib
matplotlib.use('agg')
import numpy as np
import time

from kid_readout.utils import data_block,roach_interface,data_file,sweeps
from kid_readout.analysis.resonator import Resonator
#from sim900 import sim900Client

ri = roach_interface.RoachBasebandWide10()
#ri.initialize()
ri.set_fft_gain(4)
df = data_file.DataFile()
#sc = sim900Client.sim900Client()

ri.set_dac_attenuator(25.5)  

#f0s = np.load('/home/gjones/workspace/apps/f8_fit_resonances.npy')
#f0s = np.load('/home/gjones/workspace/apps/first_pass_sc3x3_0813f9.npy')
f0s = np.load('/home/gjones/workspace/apps/sc5x4_0813f10_first_pass.npy')
#f0s = [f0s[0]+400/1e6]
#f0s = np.array([  77.923828,   79.814453,  81.123,  82.548828  ])
# ,  111.611328,     117.845703,  143.736328,  152.423828])

#f0s = f0s*(1-0.001)
atonce = 4
for k in range(len(f0s)/atonce):
    cfs = f0s[k::len(f0s)/atonce]
    offsets = np.linspace(-5e3,5e3,25)
    offsets = np.concatenate(([-40e3,-30e3,-20e3,-10e3],offsets,[10e3,20e3,30e3,40e3]))/1e6
    offsets = offsets
    sweep_data = data_block.SweepData(sweep_id=k)
    def callback(block):
        sweep_data.add_block(block)
    for offs in offsets:
        print cfs+offs
        sweeps.coarse_sweep(ri, freqs=cfs+offs, nsamp=2**21, nchan_per_step=atonce, reads_per_step=8, callback=callback)
    df.add_sweep(sweep_data)
    
#    fr,s21 = sweep_data.select_index(0)
#    res = Resonator(fr,s21)
#    fmin = fr[np.abs(s21).argmin()]
#    print "s21 fmin", fmin, "fit fmin",f0s[k],"thisfit", res.f_0
    ri.set_tone_freqs(cfs,nsamp=2**21)
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

    
    df.log_hw_state(ri)
    #sc.fetchDict()
    #df.add_cryo_data(sc.data)
    df.nc.sync()
    
#    raw_input("turn off pulse tube")
#
#    tsg = None
#    dmod,addr = ri.get_data(2048,demod=True)
#    chids = ri.fpga_fft_readout_indexes+1
#    tones = ri.tone_bins[ri.readout_selection]
#    nsamp = ri.tone_nsamp
#    print "saving data"
#    for m in range(len(chids)):
#        print m
#        block = data_block.DataBlock(data = dmod[:,m], tone=tones[m], fftbin = chids[m], 
#                 nsamp = nsamp, nfft = ri.nfft, wavenorm = ri.wavenorm, t0 = time.time(), fs = ri.fs)
#        tsg = df.add_block_to_timestream(block, tsg=tsg)
#
#    df.log_hw_state(ri)
#    df.nc.sync()
    
df.nc.close()
