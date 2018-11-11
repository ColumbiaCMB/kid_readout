import time

import numpy as np
from equipment.custom import mmwave_source
from equipment.hittite import signal_generator
from equipment.srs import lockin
from xystage import stepper


from kid_readout.interactive import *
from kid_readout.equipment import hardware
from kid_readout.measurement import mmw_source_sweep, core, acquire

logger.setLevel(logging.DEBUG)


hittite = signal_generator.Hittite(ipaddr='192.168.0.200')
hittite.set_power(0)
hittite.on()
hittite.set_freq(148e9/12)
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

hwp_motor = stepper.SimpleStepper(port='/dev/ttyACM2')

setup = hardware.Hardware(hwp_motor, source, lockin, hittite)

ri = Roach2Baseband()

ri.set_modulation_output(7)

initial_f0s = np.load('/data/readout/resonances/2017-06-JPL-8x8-LF-N1_single_horn_4_loaded2.npy')/1e6

#initial_f0s = all_f0s[:96][::6]

nf = len(initial_f0s)

atonce = 4
if nf % atonce > 0:
    print "extending list of resonators to make a multiple of ", atonce
    initial_f0s = np.concatenate((initial_f0s, np.arange(1, 1 + atonce - (nf % atonce)) + initial_f0s.max()))


print len(initial_f0s)
print initial_f0s

nsamp = 2**20
offsets = np.arange(-16,16)*512./nsamp


last_f0s = initial_f0s


mmw_freqs = np.linspace(140e9, 165e9, 128)

ri.set_dac_atten(35)


tic = time.time()
f0s = initial_f0s
#setup.hittite.off()

#high is off for initital
ri.set_modulation_output('high')


turnlist = np.arange(2.0,-0.1,-0.5)

for turn_num in turnlist:
    ri.set_modulation_output(7)
    raw_input('set attenuator knobs to %f turns & check lock-in range' %turn_num)
    source.set_attenuator_turns(turn_num, turn_num)
    #turn on source
    ri.set_modulation_output('low')
    ncf = new_nc_file(suffix='mmw_coherent_source_on_%.2f_turns' %turn_num)

    swpa = acquire.run_sweep(ri, tone_banks=f0s[None, :] + offsets[:, None], num_tone_samples=nsamp,
                             length_seconds=0.2,
                             verbose=True, state=setup.state())
    print "resonance sweep done", (time.time() - tic) / 60.

    print "sweep written", (time.time() - tic) / 60.
    current_f0s = []
    for sidx in range(swpa.num_channels):
        swp = swpa.sweep(sidx)
        res = swp.resonator
        print res.f_0, res.Q, res.delay * 1e6, res.current_result.redchi, (f0s[sidx] * 1e6 - res.f_0)
        if np.abs(f0s[sidx] * 1e6 - res.f_0) > 100e3:
            current_f0s.append(f0s[sidx] * 1e6)
            logger.info("Resonator index %d moved more than 100 kHz, keeping original value %.1f MHz" % (sidx,
                                                                                                         f0s[sidx]))
        else:
            current_f0s.append(res.f_0)
    print "fits complete", (time.time() - tic) / 60.
    current_f0s = np.array(current_f0s) / 1e6
    current_f0s.sort()
    bad_deltas = np.diff(current_f0s) < (256. / 2 ** 14) * 8
    if bad_deltas.sum():
        print "found bad deltas", bad_deltas.sum()
        current_f0s[np.nonzero(bad_deltas)] -= 0.1
        bad_deltas = np.diff(current_f0s) < (256. / 2 ** 14) * 8
        if bad_deltas.sum():
            print "found bad deltas", bad_deltas.sum()
            current_f0s[np.nonzero(bad_deltas)] -= 0.1

    ri.set_tone_freqs(current_f0s, nsamp=nsamp)
    ri.select_fft_bins(range(len(current_f0s)))
    print ri.fpga_fft_readout_indexes
    print np.diff(ri.fpga_fft_readout_indexes.astype('float')).min()

    raw_input("turn off compressor")
    meas = ri.get_measurement(num_seconds=30)
    raw_input("turn on compressor")

    #turn on modulation to get zbd voltage
    ri.set_modulation_output(7)
    time.sleep(2)

    sweep_stream_array = basic.SweepStreamArray(sweep_array=swpa, stream_array=meas, state=setup.state(fast=True),
                                                description='source on')
    ncf.write(sweep_stream_array)

    meas = ri.get_measurement(num_seconds=30)
    meas.state = setup.state(fast=True)

    sweep_stream_array = basic.SweepStreamArray(sweep_array=swpa, stream_array=meas, state=meas.state,
                                                description='chopped')
    ncf.write(sweep_stream_array)

    ncf.close()