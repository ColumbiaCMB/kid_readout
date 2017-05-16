
from kid_readout.interactive import *
from kid_readout.equipment import hardware
from kid_readout.measurement import acquire
from kid_readout.roach import analog
from kid_readout.equipment import agilent_33220
import time

fg = agilent_33220.FunctionGenerator()
fg.set_load_ohms(1000)
fg.set_dc_voltage(0)
fg.enable_output(False)


ifboard = analog.HeterodyneMarkI()

setup = hardware.Hardware(ifboard)

ri = hardware_tools.r2_with_mk1()
ri.iq_delay=-1
ri.set_fft_gain(6)

initial_f0s = np.load('/data/readout/resonances/2016-12-01-JPL-HF-2-open.npy')/1e6
initial_lo = 1010.

nf = len(initial_f0s)
atonce = 64
if nf % atonce > 0:
    print "extending list of resonators to make a multiple of ", atonce
    initial_f0s = np.concatenate((initial_f0s, np.arange(1, 1 + atonce - (nf % atonce)) + initial_f0s.max()))

print len(initial_f0s)

nsamp = 2**17
step = 1
nstep = 32
offset_bins = np.arange(-(nstep + 1), (nstep + 1)) * step
offsets = offset_bins * 512.0 / nsamp

last_f0s = initial_f0s

for heater_voltage in np.sqrt(np.linspace(0,5**2,16)):
    fg.set_dc_voltage(heater_voltage)
    if heater_voltage == 0:
        print "heater voltage is 0 V, skipping wait"
    else:
        print "waiting 30 minutes", heater_voltage
        time.sleep(1800)
    fg.enable_output(True)
    for dac_atten in [2]:
        ri.set_dac_atten(dac_atten)
        tic = time.time()
        ncf = new_nc_file(suffix='%d_dB_load_heater_%.3f_V' % (dac_atten, heater_voltage))
        swpa = acquire.run_sweep(ri, tone_banks=last_f0s[None,:] + offsets[:,None], num_tone_samples=nsamp,
                                 length_seconds=0, state=setup.state(), verbose=True,
                                 description='bb sweep')
        print "resonance sweep done", (time.time()-tic)/60.
        ncf.write(swpa)
        current_f0s = []
        for sidx in range(last_f0s.shape[0]):
            swp = swpa.sweep(sidx)
            res = swp.resonator
            print res.f_0, res.Q, res.current_result.redchi, (last_f0s[sidx]*1e6-res.f_0)
            if np.abs(res.f_0 - last_f0s[sidx]*1e6) > 200e3:
                current_f0s.append(last_f0s[sidx]*1e6)
                print "using original frequency for ",last_f0s[sidx]
            else:
                current_f0s.append(res.f_0)
        print "fits complete", (time.time()-tic)/60.
        current_f0s = np.array(current_f0s)/1e6
        current_f0s.sort()
        if np.any(np.diff(current_f0s)<0.031):
            print "problematic resonator collision:",current_f0s
            print "deltas:",np.diff(current_f0s)
            problems = np.flatnonzero(np.diff(current_f0s)<0.031)+1
            current_f0s[problems] = (current_f0s[problems-1] + current_f0s[problems+1])/2.0
        if np.any(np.diff(current_f0s)<0.031):
            print "repeated problematic resonator collision:",current_f0s
            print "deltas:",np.diff(current_f0s)
            problems = np.flatnonzero(np.diff(current_f0s)<0.031)+1
            current_f0s[problems] = (current_f0s[problems-1] + current_f0s[problems+1])/2.0
        ri.set_tone_freqs(current_f0s,nsamp)
        ri.select_fft_bins(range(last_f0s.shape[0]))
        last_f0s = current_f0s
        meas = ri.get_measurement(num_seconds=30., state=setup.state(),description='stream with bb')
        ncf.write(meas)
        print "dac_atten %f heater voltage %.3f V done in %.1f minutes" % (dac_atten, heater_voltage, (time.time()-tic)/60.)
        ncf.close()


ri.set_dac_atten(20)
