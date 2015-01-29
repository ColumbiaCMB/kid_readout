from __builtin__ import enumerate
import matplotlib
matplotlib.use('agg')
import numpy as np
import time
import sys
from kid_readout.utils import roach_interface,data_file,sweeps
from kid_readout.analysis.resonator import Resonator
from kid_readout.analysis.resonator import fit_best_resonator
import kid_readout.equipment.agilent_33220

fg = kid_readout.equipment.agilent_33220.FunctionGenerator(addr=('192.168.1.145',5025))

mmw_source_frequency = np.nan

source_on_freq_scale = 1.0  # nominally 1 if low-ish power

ri = roach_interface.RoachBaseband()
f0s = np.array([157.315, 201.49])

suffix = "led"
mmw_source_modulation_freq = 256e6/2**21
print mmw_source_modulation_freq
mmw_atten_turns = (np.nan,np.nan)


nf = len(f0s)
atonce = 2
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

max_fit_error = 0.5
use_fmin = False
attenlist = [33]#[45,43,41,39,37,35,33,31]
led_voltages=[1.5,1.7]
for led_voltage in led_voltages:

    nsamp = 2**18
    step = 1
    nstep = 80
    offset_bins = np.arange(-(nstep+1),(nstep+1))*step

    offsets = offset_bins*512.0/nsamp
    offsets = np.concatenate(([offsets.min()-20e-3,],offsets,[offsets.max()+20e-3]))

    print "measuring with LED at %f volts" % led_voltage
    fg.set_dc_voltage(led_voltage)
    ri.set_modulation_output(rate='low')
    print "setting attenuator to",attenlist[0]
    ri.set_dac_attenuator(attenlist[0])
    f0binned = np.round(f0s*source_on_freq_scale*nsamp/512.0)*512.0/nsamp
    measured_freqs = sweeps.prepare_sweep(ri,f0binned,offsets,nsamp=nsamp)
    print "loaded waveforms in", (time.time()-start),"seconds"

    sweep_data = sweeps.do_prepared_sweep(ri, nchan_per_step=atonce, reads_per_step=1)
    orig_sweep_data = sweep_data
    meas_cfs = []
    idxs = []
    delays = []
    for m in range(len(f0s)):
        fr,s21,errors = sweep_data.select_by_freq(f0s[m])
        thiscf = f0s[m]*source_on_freq_scale
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
    nsamp = 2**20
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
            thiscf = f0s[m]*source_on_freq_scale
            s21 = s21*np.exp(2j*np.pi*delay*fr)
            res = fit_best_resonator(fr,s21,errors=errors) #Resonator(fr,s21,errors=errors)
            fmin = fr[np.abs(s21).argmin()]
            print "s21 fmin", fmin, "original guess",thiscf,"this fit", res.f_0
            if ('a' in res.result.params) or use_fmin:
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
        if k == 0:
            ri.add_tone_freqs(np.array(meas_cfs))
            ri.select_bank(ri.tone_bins.shape[0]-1)
        else:
            best_bank = (np.abs((ri.tone_bins[:,0]*ri.fs/ri.tone_nsamp)-meas_cfs[0]).argmin())
            print "using bank",best_bank
            print "offsets:", ((ri.tone_bins[best_bank,:]*ri.fs/ri.tone_nsamp)-meas_cfs)
            ri.select_bank(best_bank)
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
                                         mmw_source_modulation_freq=0,
                                         zbd_voltage=x)
            df.sync()
            print "done with sweep"

    # Take modulated data
    ri.set_dac_attenuator(attenlist[0])
    ri.select_bank(ri.tone_bins.shape[0]-1)
    fg.set_square_wave(freq=mmw_source_modulation_freq,high_level=led_voltage)
    fg.enable_output(True)
    ri.set_modulation_output(rate=7)
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
        x = led_voltage

        tsg = df.add_timestream_data(dmod, ri, t0, tsg=tsg, mmw_source_freq=mmw_source_frequency,
                                     mmw_source_modulation_freq=mmw_source_modulation_freq,
                                     zbd_voltage=x)
        df.sync()
        print "done with sweep"

    # now do source off

    nsamp = 2**18
    step = 1
    nstep = 80
    offset_bins = np.arange(-(nstep+1),(nstep+1))*step

    offsets = offset_bins*512.0/nsamp
    offsets = np.concatenate(([offsets.min()-20e-3,],offsets,[offsets.max()+20e-3]))

    fg.enable_output(False)
    ri.set_modulation_output(rate='high')
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
        res = fit_best_resonator(fr,s21,errors=errors)
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
    nsamp = 2**20
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

    ri.add_tone_freqs(np.array(meas_cfs))
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
        x = 0.0

        tsg = df.add_timestream_data(dmod, ri, t0, tsg=tsg, mmw_source_freq=mmw_source_frequency,
                                     mmw_source_modulation_freq=0,
                                     zbd_voltage=x)
        df.sync()
        print "done with sweep"
print "completed in",((time.time()-start)/60.0),"minutes"
