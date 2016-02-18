from __builtin__ import enumerate

import matplotlib

from kid_readout.roach import baseband

import numpy as np
import time
import sys
from kid_readout.utils import data_file,sweeps
from kid_readout.analysis.resonator import fit_best_resonator

mmw_source_frequency = np.nan


ri = baseband.RoachBaseband(adc_valon='/dev/ttyUSB0')
f0s = np.load('/data/readout/resonances/2016-01-30-SC-2015-04-WSPEC-MT-first-pass.npy')
f0s.sort()

suffix = "dark"
mmw_source_modulation_freq = np.nan
mmw_atten_turns = (np.nan,np.nan)



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

start = time.time()

max_fit_error = 0.5
use_fmin = False
attenlist = [45,41,37,33,29,26,24,22,20]
while True:
    print "*"*40
    print "Hit enter to take a data set"
    mmw_atten_str = raw_input("start: ")
    if mmw_atten_str == 'exit':
        break

    nsamp = 2**18
    step = 1
    nstep = 80
    offset_bins = np.arange(-(nstep+1),(nstep+1))*step

    offsets = offset_bins*512.0/nsamp
    offsets = np.concatenate(([offsets.min()-20e-3,],offsets,[offsets.max()+20e-3]))

    print "setting attenuator to",attenlist[0]
    ri.set_dac_attenuator(attenlist[0])
    f0binned = np.round(f0s*nsamp/512.0)*512.0/nsamp
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
        res = fit_best_resonator(fr,s21,errors=errors,min_a=1.0)
        fmin = fr[np.abs(s21).argmin()]
        print "s21 fmin", fmin, "original guess",thiscf,"this fit", res.f_0, "delay",delay,"resid delay",res.delay
        if use_fmin:
            meas_cfs.append(fmin)
        else:
            if abs(res.f_0 - thiscf) > max_fit_error:
                if abs(fmin - thiscf) > max_fit_error:
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
    nsamp = 2**21
    step = 1

    offset_bins = np.concatenate([np.arange(-35, -10, 5),
                                  np.arange(-10, 11),
                                  np.arange(15, 40, 5)])[:-2]
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
    for k,atten in enumerate(attenlist):
        ri.set_dac_attenuator(atten)
        print "measuring at attenuation", atten
        df.log_hw_state(ri)
        if k != 0:
            orig_sweep_data = None
        sweep_data = sweeps.do_prepared_sweep(ri, nchan_per_step=atonce, reads_per_step=1, sweep_data=orig_sweep_data)
        df.add_sweep(sweep_data)
        meas_cfs = []
        idxs = []
        for m in range(len(f0s)):
            fr,s21,errors = sweep_data.select_by_freq(f0s[m])
            thiscf = f0s[m]
            s21 = s21*np.exp(2j*np.pi*delay*fr)
            res = fit_best_resonator(fr,s21,errors=errors,min_a=1.0) #Resonator(fr,s21,errors=errors)
            fmin = fr[np.abs(s21).argmin()]
            print "s21 fmin", fmin, "original guess",thiscf,"this fit", res.f_0
            if k != 0 or use_fmin:
                print "using fmin"
                meas_cfs.append(fmin)
            else:
                if abs(res.f_0 - thiscf) > max_fit_error:
                    if abs(fmin - thiscf) > max_fit_error:
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
        ri.add_tone_freqs(np.array(meas_cfs),overwrite_last=True)
        ri.select_bank(ri.tone_bins.shape[0]-1)
        ri._sync()
        time.sleep(0.5)



        df.log_hw_state(ri)
        nsets = len(meas_cfs)/atonce
        tsg = None
        for iset in range(nsets):
            selection = range(len(meas_cfs))[iset::nsets]
            ri.select_fft_bins(selection)
            ri._sync()
            time.sleep(0.4)
            t0 = time.time()
            dmod,addr = ri.get_data_seconds(30)
            x = np.nan

            tsg = df.add_timestream_data(dmod, ri, t0, tsg=tsg, mmw_source_freq=mmw_source_frequency,
                                         mmw_source_modulation_freq=mmw_source_modulation_freq,
                                         zbd_voltage=x)
            df.sync()
            print "done with sweep"


print "completed in",((time.time()-start)/60.0),"minutes"
