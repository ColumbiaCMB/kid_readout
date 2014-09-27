import matplotlib
matplotlib.use('agg')
import numpy as np
import time
import sys
from kid_readout.utils import roach_interface,data_file,sweeps
from kid_readout.analysis.resonator import Resonator
from kid_readout.analysis.resonator import fit_best_resonator
from kid_readout.equipment import hittite_controller
from kid_readout.equipment import lockin_controller
from kid_readout.equipment.agilent_33220 import FunctionGenerator

fg = FunctionGenerator()

hittite = hittite_controller.hittiteController()
lockin = lockin_controller.lockinController()
print lockin.get_idn()

ri = roach_interface.RoachBaseband(initialize=False)
ri.boffile = 'bb2xpfb14mcr16_2014_Sep_23_1157.bof'
ri.initialize()
f0s = np.load('/home/gjones/kid_readout/apps/sc5x4_0813f12.npy')
f0s.sort()
f0s = f0s[[0,1,2,3,4,5,6,7,8,9,10,13,14,15,16,17]]  # remove close packed resonators to enable reading out all simultaneously

suffix = "readoutpower"
mmw_source_modulation_freq = 25.0
mmw_atten_turns = (6.5,6.5)

def source_on():
    ri.r.write_int('gpiob',0x00)

def source_off():
    ri.r.write_int('gpiob',0x01)

def source_modulate(rate=2):
    ri.r.write_int('gpiob', 0x02 + rate)

nf = len(f0s)
atonce = 16
if nf % atonce > 0:
    print "extending list of resonators to make a multiple of ",atonce
    f0s = np.concatenate((f0s,np.arange(1,1+atonce-(nf%atonce))+f0s.max()))

nsamp = 2**18
step = 1
nstep = 80
f0binned = np.round(f0s*nsamp/512.0)*512.0/nsamp
offset_bins = np.arange(-(nstep+1),(nstep+1))*step

offsets = offset_bins*512.0/nsamp
offsets = np.concatenate(([offsets.min()-20e-3,],offsets,[offsets.max()+20e-3]))

print f0s
print offsets*1e6
print len(f0s)

if False:
    from kid_readout.utils.parse_srs import get_all_temperature_data
    while True:
        temp = get_all_temperature_data()[1][-1]
        print "mk stage at", temp
        if temp > 0.348:
            break
        time.sleep(300)
    time.sleep(600)
start = time.time()

source_off()

use_fmin = False
attenlist = [39]
if True:

    for atten in attenlist:
        nsamp = 2**18
        step = 1
        nstep = 80
        offset_bins = np.arange(-(nstep+1),(nstep+1))*step

        offsets = offset_bins*512.0/nsamp
        offsets = np.concatenate(([offsets.min()-20e-3,],offsets,[offsets.max()+20e-3]))

        print "setting attenuator to",atten
        ri.set_dac_attenuator(atten)
        measured_freqs = sweeps.prepare_sweep(ri,f0binned,offsets,nsamp=nsamp)
        print "loaded waveforms in", (time.time()-start),"seconds"

        sweep_data = sweeps.do_prepared_sweep(ri, nchan_per_step=atonce, reads_per_step=1)
        orig_sweep_data = sweep_data
        meas_cfs = []
        idxs = []
        delays = []
        for m in range(len(f0s)):
            fr,s21,errors = sweep_data.select_by_freq(f0s[m])
            thiscf = f0s[m]
            res = fit_best_resonator(fr[1:-1],s21[1:-1],errors=errors[1:-1]) #Resonator(fr,s21,errors=errors)
            delay = res.delay
            delays.append(delay)
            s21 = s21*np.exp(2j*np.pi*res.delay*fr)
            res = fit_best_resonator(fr,s21,errors=errors)
            fmin = fr[np.abs(s21).argmin()]
            print "s21 fmin", fmin, "original guess",thiscf,"this fit", res.f_0, "delay",delay,"resid delay",res.delay
            if use_fmin:
                meas_cfs.append(fmin)
            else:
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

        delay = np.median(delays)
        print "median delay is ",delay
        nsamp = 2**22
        step = 1

        offset_bins = np.array([-8,-4,-2,-1,0,1,2,4])
        offset_bins = np.concatenate(([-40,-20],offset_bins,[20,40]))
        offsets = offset_bins*512.0/nsamp

        meas_cfs = np.array(meas_cfs)
        f0binned_meas = np.round(meas_cfs*nsamp/512.0)*512.0/nsamp
        f0s = f0binned_meas
        measured_freqs = sweeps.prepare_sweep(ri,f0binned_meas,offsets,nsamp=nsamp)
        print "loaded updated waveforms in", (time.time()-start),"seconds"



        sys.stdout.flush()
        time.sleep(1)


        df = data_file.DataFile(suffix=suffix)
        df.nc.mmw_atten_turns=mmw_atten_turns
        df.log_hw_state(ri)
        sweep_data = sweeps.do_prepared_sweep(ri, nchan_per_step=atonce, reads_per_step=1, sweep_data=orig_sweep_data)
        df.add_sweep(sweep_data)
        meas_cfs = []
        idxs = []
        for m in range(len(f0s)):
            fr,s21,errors = sweep_data.select_by_freq(f0s[m])
            thiscf = f0s[m]
            s21 = s21*np.exp(2j*np.pi*delay*fr)
            res = fit_best_resonator(fr,s21,errors=errors) #Resonator(fr,s21,errors=errors)
            fmin = fr[np.abs(s21).argmin()]
            print "s21 fmin", fmin, "original guess",thiscf,"this fit", res.f_0
            if use_fmin:
                meas_cfs.append(fmin)
            else:
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


        #raw_input("turn on LED take data")

        for atten2 in np.arange(39,20,-1):
            ri.set_dac_attenuator(atten2)
            df.log_hw_state(ri)
            nsets = len(meas_cfs)/atonce
            tsg = None
            for iset in range(nsets):
                selection = range(len(meas_cfs))[iset::nsets]
                ri.select_fft_bins(selection)
                ri._sync()
                time.sleep(0.4)
                t0 = time.time()
                dmod,addr = ri.get_data_seconds(30*4)
                x,y,r,theta = lockin.get_data()

                tsg = df.add_timestream_data(dmod, ri, t0, tsg=tsg, mmw_source_freq=-1.0,
                                             mmw_source_modulation_freq=mmw_source_modulation_freq,
                                             zbd_voltage=x)
                df.sync()
                print "done with atten",atten2

        df.close()

print "completed in",((time.time()-start)/60.0),"minutes"
