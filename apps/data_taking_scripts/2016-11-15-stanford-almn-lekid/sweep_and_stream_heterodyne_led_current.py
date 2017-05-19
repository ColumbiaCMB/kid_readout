import time

import numpy as np

import kid_readout.equipment.hardware
from kid_readout.interactive import *
from kid_readout.equipment import hardware
from equipment.keithley.sourcemeter import SourceMeter
from kid_readout.measurement import acquire
from kid_readout.roach import analog


#fg = agilent_33220.FunctionGenerator()
keithley = SourceMeter('/dev/ttyUSB3')
print keithley.identify()
ifboard = analog.HeterodyneMarkI()
#led = kid_readout.equipment.hardware.Thing('cold_1550nm_led', dict(voltage=3.3,
#                                                                   source_impedance=150,
#                                                                   pulse_width=10e-6,
#                                                                   frequency=244.1,
#                                                                   ))#{'current': 1e-4})

setup = hardware.Hardware(ifboard,keithley)
keithley.set_current_source()

ri = hardware_tools.r2_with_mk1()
ri.iq_delay=-1
ri.set_fft_gain(6)

#initial_f0s = np.load('/data/readout/resonances/2016-06-18-jpl-hex-271-32-high-qi-lo-1210-resonances.npy')
#initial_f0s = initial_f0s/1e6
initial_f0_groups = [#[867.636, 926.75, 1009.072],
                     [1009.072, 1375.513, 1392.178],
                     [1603.5, 1713.939,]]
lo_freqs = [#850.0,
            #1350.0,
            1200.0,
            1780.0]
next_f0s=initial_f0_groups

for led_current in np.concatenate(([0],np.logspace(-6,-3,10))):
    keithley.set_current_amplitude(led_current)
    keithley.set_output(True)
    print "led current", led_current
    last_f0s = next_f0s
    next_f0s = []
    for lo_freq, initial_f0s in zip(lo_freqs,last_f0s):
        initial_f0s = np.array(initial_f0s)
        nf = len(initial_f0s)
        atonce = 4
        if nf % atonce > 0:
            print "extending list of resonators to make a multiple of ", atonce
            initial_f0s = np.concatenate((initial_f0s, np.arange(1, 1 + atonce - (nf % atonce)) + initial_f0s.max()))

        print len(initial_f0s)
        nsamp = 2**17
        step = 1
        nstep = 64
        offset_bins = np.arange(-(nstep + 1), (nstep + 1)) * step
        offsets = offset_bins * 512.0 / nsamp



        ri.set_lo(lo_freq)
        f0s = initial_f0s
        for dac_atten in [6,10]:
            ri.set_dac_atten(dac_atten)
            ri.set_modulation_output('low')
            tic = time.time()
            ncf = new_nc_file(suffix='%d_dB_dac' % dac_atten)
            swpa = acquire.run_sweep(ri, tone_banks=f0s[None,:] + offsets[:,None], num_tone_samples=nsamp,
                                     length_seconds=0, state=setup.state(), verbose=True,
                                     description='dark sweep')
            print "resonance sweep done", (time.time()-tic)/60.
            ncf.write(swpa)
            #print "sweep written", (time.time()-tic)/60.
            current_f0s = []
            for sidx in range(f0s.shape[0]):
                swp = swpa.sweep(sidx)
                res = swp.resonator
                print res.f_0, res.Q, res.current_result.redchi, (f0s[sidx]*1e6-res.f_0)
                if np.abs(res.f_0 - f0s[sidx]*1e6) > 200e3:
                    current_f0s.append(f0s[sidx]*1e6)
                    print "using original frequency for ",f0s[sidx]
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
            ri.select_fft_bins(range(f0s.shape[0]))
            #raw_input("turn off compressor")
            #meas = ri.get_measurement(num_seconds=30., state=setup.state(),description='source off stream')
            #ncf.write(meas)
            ri.set_modulation_output(6)
            meas = ri.get_measurement(num_seconds=30., state=setup.state(),description='source on stream')
            ncf.write(meas)

            print "dac_atten %f done in %.1f minutes" % (dac_atten, (time.time()-tic)/60.)
            ncf.close()
        next_f0s.append(current_f0s)

ri.set_dac_atten(20)
