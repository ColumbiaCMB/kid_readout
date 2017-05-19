import matplotlib

from kid_readout.roach import baseband

matplotlib.use('agg')
import numpy as np
import time

from kid_readout.utils import data_block, data_file
#from sim900 import sim900Client

ri = baseband.RoachBasebandWide()
ri.initialize()
ri.set_fft_gain(4)
#sc = sim900Client.sim900Client()

ri.set_dac_attenuator(19.5)  

#f0s = np.load('/home/gjones/workspace/apps/f8_fit_resonances.npy')
#f0s = np.load('/home/gjones/workspace/apps/first_pass_sc3x3_0813f9.npy')
#f0s = np.load('/home/gjones/workspace/apps/sc5x4_0813f10_first_pass.npy')
f0s = np.array([108.904834,
                125.508098,
                155.543446,
                145.978222,
                132.982668,
                82.965662,
                89.920389,
                114.491795])
f0s.sort()
#f0s = f0s*0.99

nf = len(f0s)
atonce = 8
if nf % atonce > 0:
    print "extending list of resonators to make a multiple of ",atonce
    f0s = np.concatenate((f0s,np.arange(1,1+atonce-(nf%atonce))+f0s.max()))

print f0s
print len(f0s)
atten = 13.5
df = data_file.DataFile()
ri.set_dac_attenuator(atten)

ri.set_tone_freqs(f0s,nsamp=2**21)
ri._sync()
time.sleep(0.5)
    
start = time.time()
while time.time() - start < 20*60:    
    tsg = None
    dmod,addr = ri.get_data(128,demod=True)
    chids = ri.fpga_fft_readout_indexes+1
    tones = ri.tone_bins[0,ri.readout_selection]
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
df.nc.close()
    
print "completed in",((time.time()-start)/60.0),"minutes"
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
    