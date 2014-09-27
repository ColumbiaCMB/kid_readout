import matplotlib
matplotlib.use('agg')
import numpy as np
import time
import sys
from kid_readout.utils import roach_interface,data_file,sweeps
from kid_readout.analysis.resonator import Resonator,fit_best_resonator

ri = roach_interface.RoachBasebandWide()
ri.initialize()
#f0s = np.load('/home/gjones/workspace/apps/f8_fit_resonances.npy')
#f0s = np.load('/home/gjones/workspace/apps/first_pass_sc3x3_0813f9.npy')
#f0s = np.load('/home/gjones/workspace/apps/sc5x4_0813f10_first_pass.npy')#[:4]
f0s = np.load('/home/gjones/workspace/apps/sc5x4_0813f12.npy')
f0s.sort()
#f0s = f0s*0.998

nf = len(f0s)
atonce = 4
if nf % atonce > 0:
    print "extending list of resonators to make a multiple of ",atonce
    f0s = np.concatenate((f0s,np.arange(1,1+atonce-(nf%atonce))+f0s.max()))

#offsets = np.linspace(-4882.8125,4638.671875,20)
#offsets = np.concatenate(([-40e3,-20e3],offsets,[20e3,40e3]))/1e6
#offsets = offsets*8

nsamp = 2**20
step = 1
f0binned = np.round(f0s*nsamp/512.0)*512.0/nsamp
offset_bins = np.arange(-21,21)*step

offsets = offset_bins*512.0/nsamp
offsets = np.concatenate(([-40e-3,-20e-3],offsets,[20e-3,40e-3]))


print f0s
print len(f0s)
print offsets*1e6
start = time.time()

measured_freqs = sweeps.prepare_sweep(ri,f0binned,offsets,nsamp=nsamp)
print "loaded waveforms in", (time.time()-start),"seconds"

sys.stdout.flush()
time.sleep(1)

atten_list = np.linspace(15,46,8)#[30]#[35.5,33.5,46.5,43.5,40.5,37.5]
#atten_list = [33.0]
for atten in atten_list:
    df = data_file.DataFile()
    ri.set_dac_attenuator(atten)
    sweep_data = sweeps.do_prepared_sweep(ri, nchan_per_step=atonce, reads_per_step=8)
    df.add_sweep(sweep_data)
    meas_cfs = []
    idxs = []
    for m in range(len(f0s)):
        fr,s21,errors = sweep_data.select_by_freq(f0s[m])
        thiscf = f0s[m]
        res = fit_best_resonator(fr[2:-2],s21[2:-2],errors=errors[2:-2])
        delay = res.delay
#        delays.append(delay)
        s21 = s21*np.exp(2j*np.pi*res.delay*fr)
        res = fit_best_resonator(fr,s21,errors=errors)
        fmin = fr[np.abs(s21).argmin()]
        print "s21 fmin", fmin, "original guess",thiscf,"this fit", res.f_0,"delay",delay,"resid",res.delay
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
    tsg = None
    for iset in range(nsets):
        selection = range(len(meas_cfs))[iset::nsets]
        ri.select_fft_bins(selection)
        ri._sync()
        time.sleep(0.2)
        dmod,addr = ri.get_data_seconds(30,demod=True)
        tsg = df.add_timestream_data(dmod, ri, tsg=tsg)
    df.sync()

    df.log_hw_state(ri)
    df.nc.sync()
    
    df.nc.close()

print "completed in",((time.time()-start)/60.0),"minutes"
