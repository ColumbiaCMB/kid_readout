import time

import numpy as np
from equipment.custom import mmwave_source
from equipment.srs import lockin
from equipment.hittite import signal_generator

from kid_readout.interactive import *
from kid_readout.equipment import hardware
from kid_readout.measurement import acquire
from kid_readout.roach import analog

# fg = FunctionGenerator()
hittite = signal_generator.Hittite(ipaddr='192.168.0.200')
hittite.set_power(0)
hittite.on()
lockin = lockin.Lockin(LOCKIN_SERIAL_PORT)
tic = time.time()
print lockin.identification

source = mmwave_source.MMWaveSource()
source.set_attenuator_turns(3.0,3.0)
source.multiplier_input = 'hittite'
source.waveguide_twist_angle = 45
source.ttl_modulation_source = 'roach'

ifboard = analog.HeterodyneMarkI()

setup = hardware.Hardware(source, lockin, ifboard,hittite)
setup.hittite.set_freq(148e9/12.)

ri = hardware_tools.r2_with_mk1(1000.)
ri.iq_delay=-1
ri.set_fft_gain(6)
ri.demodulator.hardware_delay_samples = -ri.demodulator.hardware_delay_samples

#initial_f0s = np.load('/data/readout/resonances/2016-06-18-jpl-hex-271-32-high-qi-lo-1210-resonances.npy')
#initial_f0s = initial_f0s/1e6
initial_f0s = np.array([1177.4, 1186.0, 1192.4, 1194.46])
initial_lo = 1020.

nsamp = 2**17
step = 1
nstep = 48
offset_bins = np.arange(-(nstep + 1), (nstep + 1)) * step
offsets = offset_bins * 512.0 / nsamp



for (lo,f0s) in [(initial_lo,initial_f0s)]:
    ri.set_lo(lo)
    for dac_atten in [20,10,6,2]:
        ncf = new_nc_file(suffix='off_on_cw_%d_dB_dac' % dac_atten)
        ri.set_modulation_output('high')
        swpa = acquire.run_sweep(ri, tone_banks=f0s[None,:] + offsets[:,None], num_tone_samples=nsamp,
                                 length_seconds=0, state=setup.state(), verbose=True,
                                 description='source off sweep')
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
        if np.any(np.diff(current_f0s)<0.25):
            print "problematic resonator collision:",current_f0s
            print "deltas:",np.diff(current_f0s)
        ri.set_tone_freqs(current_f0s,nsamp)
        ri.select_fft_bins(range(f0s.shape[0]))
        meas = ri.get_measurement(num_seconds=120., state=setup.state(),description='source off stream')
        ncf.write(meas)

        ri.set_modulation_output('low')
        meas = ri.get_measurement(num_seconds=30., state=setup.state(),description='source on stream')
        ncf.write(meas)

        ri.set_modulation_output(7)
        time.sleep(1) # wait for source modulation to stabilize
        meas = ri.get_measurement(num_seconds=4., state=setup.state(),description='source modulated stream')
        ncf.write(meas)
        print "dac_atten %f done in %.1f minutes" % (dac_atten, (time.time()-tic)/60.)
        ncf.close()

ri.set_dac_atten(20)
