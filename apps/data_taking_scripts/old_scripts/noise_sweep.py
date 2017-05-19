import matplotlib

from kid_readout.roach import baseband

matplotlib.use('agg')
import numpy as np
import time

from kid_readout.utils import data_block, data_file,sweeps
from kid_readout.analysis.resonator import Resonator
#from sim900 import sim900Client

ri = baseband.RoachBasebandWide10()
#ri.initialize()
ri.set_fft_gain(4)
#sc = sim900Client.sim900Client()

ri.set_dac_attenuator(25.5)  

#f0s = np.load('/home/gjones/workspace/apps/f8_fit_resonances.npy')
#f0s = np.load('/home/gjones/workspace/apps/first_pass_sc3x3_0813f9.npy')
f0s = np.load('/home/gjones/workspace/apps/sc5x4_0813f10_first_pass.npy')
f0s.sort()

nf = len(f0s)
if nf % 4 > 0:
    print "extending list of resonators to make a multiple of 4"
    f0s = np.concatenate((f0s,np.arange(1,1+4-(nf%4))+f0s.max()))

print f0s
print len(f0s)
atonce = 4
start = time.time()
atten_list = [19.5]
for atten in atten_list:
    df = data_file.DataFile()
    ri.set_dac_attenuator(atten)
    for k in range(len(f0s)/atonce):
        cfs = f0s[k::len(f0s)/atonce]
        offsets = np.linspace(-10e3,10e3,30)
        offsets = np.concatenate(([-40e3,-30e3,-20e3,-10e3],offsets,[10e3,20e3,30e3,40e3]))/1e6
        offsets = offsets
        sweep_data = data_block.SweepData(sweep_id=k)
        def callback(block):
            sweep_data.add_block(block)
        for offs in offsets:
            print cfs+offs
            sweeps.coarse_sweep(ri, freqs=cfs+offs, nsamp=2**21, nchan_per_step=atonce, reads_per_step=8, callback=callback)
        df.add_sweep(sweep_data)
        
        meas_cfs= []
        for m in range(atonce):
            fr,s21 = sweep_data.select_index(m)
            thiscf = cfs[np.abs(cfs-fr.mean()).argmin()]
            res = Resonator(fr,s21)
            fmin = fr[np.abs(s21).argmin()]
            print "s21 fmin", fmin, "original guess",thiscf,"this fit", res.f_0
            if abs(res.f_0 - thiscf) > 0.02:
                if abs(fmin - thiscf) > 0.02:
                    print "using original guess"
                    meas_cfs.append(thiscf)
                else:
                    print "using fmin"
                    meas_cfs.append(fmin)
            else:
                print "using this fit"
                meas_cfs.append(res.f_0)
        print meas_cfs
        ri.set_tone_freqs(np.array(meas_cfs),nsamp=2**21)
        ri._sync()
        time.sleep(0.5)
            
        tsg = None
        dmod,addr = ri.get_data(2048*2,demod=True)
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
    