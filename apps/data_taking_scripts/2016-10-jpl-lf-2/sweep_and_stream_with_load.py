from kid_readout.interactive import *
import time
from kid_readout.equipment import agilent_33220

fg = agilent_33220.FunctionGenerator()
fg.set_load_ohms(1000)
fg.set_dc_voltage(0)
fg.enable_output(False)


#setup = hardware.Hardware()

ri = Roach2Baseband()

initial_f0s = np.load('/data/readout/resonances/2016-10-04-JPL-8x8-LF-2_firstcooldown_resonances.npy')/1e6

nf = len(initial_f0s)
atonce = 128
if nf % atonce > 0:
    print "extending list of resonators to make a multiple of ", atonce
    initial_f0s = np.concatenate((initial_f0s, np.arange(1, 1 + atonce - (nf % atonce)) + initial_f0s.max()))

nsamp = 2**18 #going above 2**18 with 128 simultaneous tones doesn't quite work yet
offsets = np.arange(-16,16)*512./nsamp

for heater_voltage in np.sqrt(np.linspace(0,5**2,16)):
    fg.set_dc_voltage(heater_voltage)
    fg.enable_output(True)
    for dac_atten in [20]:
        tic = time.time()
        ri.set_dac_atten(dac_atten)
        ncf = new_nc_file(suffix='%d_dB_load_heater_%.3f_V' % (dac_atten,heater_voltage))
        swpa = acquire.run_sweep(ri, tone_banks=initial_f0s[None,:] + offsets[:,None], num_tone_samples=nsamp,
                                     length_seconds=0, verbose=True,
                                 )
        ncf.write(swpa)
        current_f0s = []
        for sidx in range(initial_f0s.shape[0]):
            swp = swpa.sweep(sidx)
            res = swp.resonator
            print res.f_0, res.Q, res.current_result.redchi, (initial_f0s[sidx]*1e6-res.f_0)
            if np.abs(res.f_0 - initial_f0s[sidx]*1e6) > 200e3:
                current_f0s.append(initial_f0s[sidx]*1e6)
                print "using original frequency for ",initial_f0s[sidx]
            else:
                current_f0s.append(res.f_0)
        print "fits complete", (time.time()-tic)/60.
        current_f0s = np.array(current_f0s)/1e6
        current_f0s.sort()
        if np.any(np.diff(current_f0s)<0.015):
            print "problematic resonator collision:",current_f0s
            print "deltas:",np.diff(current_f0s)
            problems = np.flatnonzero(np.diff(current_f0s)<0.015)+1
            current_f0s[problems] = (current_f0s[problems-1] + current_f0s[problems+1])/2.0
        if np.any(np.diff(current_f0s)<0.015):
            print "repeated problematic resonator collision:",current_f0s
            print "deltas:",np.diff(current_f0s)
            problems = np.flatnonzero(np.diff(current_f0s)<0.015)+1
            current_f0s[problems] = (current_f0s[problems-1] + current_f0s[problems+1])/2.0
        ri.set_tone_freqs(current_f0s,nsamp)
        ri.select_fft_bins(range(initial_f0s.shape[0]))
        #raw_input("turn off compressor")
        fg.enable_output(True)
        on_at = time.time()
        meas = ri.get_measurement(num_seconds=300., description='stream while load is heating. load on at %f'%on_at)
        fg.enable_output(False)
        off_at = time.time()
        ncf.write(meas)
        meas = ri.get_measurement(num_seconds=300., description='stream while load is cooling. load off at %f' %off_at)
        ncf.write(meas)
        print "dac_atten %f heater voltage %.3f V done in %.1f minutes" % (dac_atten, heater_voltage, (time.time()-tic)/60.)
        ncf.close()
    print "waiting 30 minutes", heater_voltage
    time.sleep(1800)

