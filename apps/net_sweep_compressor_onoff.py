import matplotlib
matplotlib.use('agg')
import numpy as np
import time
import sys
from kid_readout.utils import roach_interface,data_file,sweeps
from kid_readout.analysis.resonator import Resonator
from kid_readout.analysis.resonator import fit_best_resonator
from kid_readout.equipment.agilent_33220 import FunctionGenerator

fg = FunctionGenerator()
ri = roach_interface.RoachBasebandWide()
ri.initialize()
#ri.set_fft_gain(6)
#f0s = np.load('/home/gjones/workspace/apps/f8_fit_resonances.npy')
#f0s = np.load('/home/gjones/workspace/apps/first_pass_sc3x3_0813f9.npy')
#f0s = np.load('/home/gjones/workspace/apps/sc5x4_0813f10_first_pass.npy')#[:4]
#f0s = np.load('/home/gjones/workspace/readout/apps/sc3x3_0813f9_2014-02-11.npy')
#f0s = np.load('/home/gjones/workspace/readout/apps/sc3x3_0813f5_2014-02-27.npy')
f0s = np.load('/home/gjones/workspace/apps/sc5x4_0813f12.npy')
f0s.sort()
#f0s = f0s*(1-4e-5)

nf = len(f0s)
atonce = 4
if nf % atonce > 0:
    print "extending list of resonators to make a multiple of ",atonce
    f0s = np.concatenate((f0s,np.arange(1,1+atonce-(nf%atonce))+f0s.max()))

offsets = np.linspace(-4882.8125,4638.671875,20)#[5:15]
offsets = offsets
#offsets = np.concatenate(([-40e3,-20e3],offsets,[20e3,40e3]))/1e6
offsets = np.concatenate(([-40e3],offsets,[40e3]))/1e6
#offsets = offsets*4

nsamp = 2**20
step = 1
f0binned = np.round(f0s*nsamp/512.0)*512.0/nsamp
offset_bins = np.arange(-21,21)*step

offsets = offset_bins*512.0/nsamp
offsets = np.concatenate(([-20e-3,],offsets,[20e-3]))

print f0s
print offsets*1e6
print len(f0s)

#heater_voltages = 0.4*np.sqrt(np.arange(1,11))
heater_voltages = [0.2,0.3,0.4,0.5,0.6,0.8]
#heater_voltages = [.1, , 1.4]
#heater_voltages = np.hstack(([0.0],heater_voltages))

fg.set_dc_voltage(0)
#time.sleep(60*10)

if False:
    from kid_readout.utils.parse_srs import get_all_temperature_data
    while True:
        temp = get_all_temperature_data()[1][-1]
        print "mk stage at", temp
        if temp < 0.205:
            break
        time.sleep(300)
    time.sleep(120)
start = time.time()

ri.set_dac_attenuator(33.0)

for heater_voltage in heater_voltages:
    measured_freqs = sweeps.prepare_sweep(ri,f0binned,offsets,nsamp=nsamp)
    print "loaded waveforms in", (time.time()-start),"seconds"
    
    sweep_data = sweeps.do_prepared_sweep(ri, nchan_per_step=atonce, reads_per_step=8)
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
    f0binned = np.round(f0s*nsamp/512.0)*512.0/nsamp
    offset_bins = np.array([-8,-4,-2,-1,0,1,2,4])#np.arange(-4,4)*step
    
    offset_bins = np.concatenate(([-40,-20],offset_bins,[20,40]))
    offsets = offset_bins*512.0/nsamp
    
    meas_cfs = np.array(meas_cfs)
    f0binned = np.round(meas_cfs*nsamp/512.0)*512.0/nsamp
    f0s = f0binned 
    measured_freqs = sweeps.prepare_sweep(ri,f0binned,offsets,nsamp=nsamp)
    print "loaded updated waveforms in", (time.time()-start),"seconds"
    
    
    
    sys.stdout.flush()
    time.sleep(1)
    

#for heater_voltage in heater_voltages:
    df = data_file.DataFile(suffix='net_compressor_onoff')
    df.log_hw_state(ri)
    sweep_data = sweeps.do_prepared_sweep(ri, nchan_per_step=atonce, reads_per_step=8,sweep_data = orig_sweep_data)
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
    df.log_hw_state(ri)
    nsets = len(meas_cfs)/atonce
    print "*"*40
    print "*"*40
    print "prepare to take timeseries data."
    raw_input("Turn off compressor and press enter")
    print "starting to take data"
    tsg = None
    for iset in range(nsets):
        selection = range(len(meas_cfs))[iset::nsets]
        ri.select_fft_bins(selection)
        ri._sync()
        time.sleep(0.2)
        t0 = time.time()
        dmod,addr = ri.get_data_seconds(8,demod=True)
        print nsets,iset,tsg
        tsg = df.add_timestream_data(dmod, ri, t0, tsg=tsg)
    print "done taking data"
    raw_input("turn compressor back on (if not on already) and press enter")
    print "continuing...."
    df.sync()
    
    print "finished baseline data collection"
    print "Setting heater to ",heater_voltage
    fg.set_dc_voltage(heater_voltage)
    heat_start = time.time()
    while time.time() - heat_start < 1800.0:
        df.log_hw_state(ri)
        print "making sweep"
        sweep_data = sweeps.do_prepared_sweep(ri, nchan_per_step=atonce, reads_per_step=8)
        df.add_sweep(sweep_data)
        ri.select_bank(ri.tone_bins.shape[0]-1)
        ri._sync()
        nsets = len(meas_cfs)/atonce
        tsg = None
        for iset in range(nsets):
            selection = range(len(meas_cfs))[iset::nsets]
            ri.select_fft_bins(selection)
            ri._sync()
            time.sleep(0.2)
            t0 = time.time()
            dmod,addr = ri.get_data_seconds(4,demod=True)
            print nsets,iset,tsg
            tsg = df.add_timestream_data(dmod, ri, t0, tsg=tsg)
        df.sync()
        print "sleeping for 120 seconds"
        time.sleep(120*2)



    df.log_hw_state(ri)
    df.nc.sync()
    df.nc.close()
    
print "turning heater to 0.0"
fg.set_dc_voltage(0.0)
print "completed in",((time.time()-start)/60.0),"minutes"
