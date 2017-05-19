from kid_readout.interactive import *
from kid_readout.measurement import mmw_source_sweep, core

from equipment.custom import mmwave_source
from equipment.hittite import signal_generator
from equipment.srs import lockin

import time

hittite = signal_generator.Hittite(ipaddr='192.168.0.200')
hittite.set_power(0)
hittite.on()
lockin = lockin.Lockin(LOCKIN_SERIAL_PORT)
tic = time.time()
# lockin.sensitivity = 17
print lockin.identification
print lockin.identification
# print time.time()-tic
# tic = time.time()
# print lockin.state(measurement_only=True)
# print time.time()-tic
source = mmwave_source.MMWaveSource()
source.set_attenuator_turns(6.0,6.0)
source.multiplier_input = 'hittite'
source.waveguide_twist_angle = 0
source.ttl_modulation_source = 'roach'

setup = hardware.Hardware(hittite, source, lockin)

ri = Roach2Baseband()

initial_f0s = np.load('/data/readout/resonances/2017-05-08-bnl-hex-271-lo-1250-mhz-128.npy')/1e6

nf = len(initial_f0s)
atonce = 128
if nf % atonce > 0:
    print "extending list of resonators to make a multiple of ", atonce
    initial_f0s = np.concatenate((initial_f0s, np.arange(1, 1 + atonce - (nf % atonce)) + initial_f0s.max()))

nsamp = 2**18 #going above 2**18 with 128 simultaneous tones doesn't quite work yet
offsets = np.arange(-16,16)*512./nsamp

mmw_freqs = np.linspace(140e9, 165e9, 500)

ri.set_dac_atten(20)
setup.hittite.off()
ri.set_modulation_output('high')

tic = time.time()
ncf = new_nc_file(suffix='mmw_source_sweep')
swpa = acquire.run_sweep(ri, tone_banks=initial_f0s[None,:] + offsets[:,None], num_tone_samples=nsamp,
                             length_seconds=0, verbose=True,
                         )
sweepstream = mmw_source_sweep.MMWSweepList(swpa, core.IOList(), state=setup.state())
ncf.write(sweepstream)
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

setup.hittite.on()
ri.set_modulation_output(7)
for n, freq in enumerate(mmw_freqs):
    setup.hittite.set_freq(freq/12.)
    time.sleep(0.5)
    tic2= time.time()
    state=setup.state(fast=True)
    print time.time()-tic2
    meas = ri.get_measurement(num_seconds=1., state=state)
    print freq,(time.time()-tic2)
    sweepstream.stream_list.append(meas)


print "dac_atten %f done in %.1f minutes" % (20, (time.time()-tic)/60.)
ncf.close()

