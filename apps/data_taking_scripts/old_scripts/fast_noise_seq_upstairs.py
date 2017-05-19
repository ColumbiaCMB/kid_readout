import matplotlib
from kid_readout.measurement.io import data_block

from kid_readout.roach import baseband

matplotlib.use('agg')
import numpy as np
import time
import sys
from kid_readout.utils import data_file,sweeps
from kid_readout.resonator.resonator import Resonator
#from sim900 import sim900Client

ri = baseband.RoachBasebandWide()
#ri.initialize()
ri.set_fft_gain(4)
#sc = sim900Client.sim900Client()

ri.set_dac_attenuator(19.5)  

#f0s = np.load('/home/gjones/workspace/apps/f8_fit_resonances.npy')
#f0s = np.load('/home/gjones/workspace/apps/first_pass_sc3x3_0813f9.npy')
f0s = np.load('/home/gjones/workspace/apps/sc5x4_0813f10_first_pass.npy')[:4]
#f0s = np.load('/home/gjones/workspace/readout/apps/sc3x3_0813f9_2014-02-11.npy')
f0s.sort()
#f0s = f0s*0.99

nf = len(f0s)
atonce = 4
if nf % atonce > 0:
    print "extending list of resonators to make a multiple of ",atonce
    f0s = np.concatenate((f0s,np.arange(1,1+atonce-(nf%atonce))+f0s.max()))

offsets = np.linspace(-4882.8125,4638.671875,20)
offsets = np.concatenate(([-40e3,-20e3],offsets,[20e3,40e3]))/1e6
offsets = offsets

print f0s
print len(f0s)
start = time.time()

measured_freqs = sweeps.prepare_sweep(ri,f0s,offsets,nsamp=2**21)
print "loaded waveforms in", (time.time()-start),"seconds"

sys.stdout.flush()
time.sleep(1)

atten_list = [30]#[35.5,33.5,46.5,43.5,40.5,37.5]
for atten in atten_list:
    df = data_file.DataFile()
    ri.set_dac_attenuator(atten)
    sweep_data = sweeps.do_prepared_sweep(ri, nchan_per_step=atonce, reads_per_step=8)
    df.add_sweep(sweep_data)
    meas_cfs = []
    idxs = []
    for m in range(len(f0s)):
        fr,s21 = sweep_data.select_by_freq(f0s[m])
        thiscf = f0s[m]
        res = Resonator(fr,s21)
        fmin = fr[np.abs(s21).argmin()]
        print "s21 fmin", fmin, "original guess",thiscf,"this fit", res.f_0
        if abs(res.f_0 - thiscf) > 0.1:
            if abs(fmin - thiscf) > 0.1:
                print "using original guess"
                meas_cfs.append(thiscf)
            else:
                print "using fmin"
                meas_cfs.append(fmin)
        else:
            print "using this fit"
            meas_cfs.append(res.f_0)
        idx = np.unravel_index(abs(measured_freqs - meas_cfs[-1]).argmin(),measured_freqs.shape)
        idxs.append(idx)
    print meas_cfs
    ri.add_tone_freqs(np.array(meas_cfs))
    ri.select_bank(ri.tone_bins.shape[0]-1)
    ri._sync()
    time.sleep(0.5)
    
    nsets = len(meas_cfs)/atonce
    start_time = time.time()
    while time.time() - start_time < 20*60:
        tsg = None
        for iset in range(nsets):
            selection = range(len(meas_cfs))[iset::nsets]
            ri.select_fft_bins(selection)
            ri._sync()
            time.sleep(0.2)
            dmod,addr = ri.get_data(128,demod=True)
            chids = ri.fpga_fft_readout_indexes+1
            tones = ri.tone_bins[ri.bank,ri.readout_selection]
            nsamp = ri.tone_nsamp
            print "saving data"
            for m in range(len(chids)):
                print m
                block = data_block.DataBlock(data = dmod[:,m], tone=tones[m], fftbin = chids[m], 
                         nsamp = nsamp, nfft = ri.nfft, wavenorm = ri.wavenorm, t0 = time.time(), fs = ri.fs)
                tsg = df.add_block_to_timestream(block, tsg=tsg)
        df.sync()
        print "elapsed time:", ((time.time()-start_time)/60), "minutes"
        time.sleep(30)

    sweep_data = sweeps.do_prepared_sweep(ri, nchan_per_step=atonce, reads_per_step=8)
    df.add_sweep(sweep_data)
        
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
    
