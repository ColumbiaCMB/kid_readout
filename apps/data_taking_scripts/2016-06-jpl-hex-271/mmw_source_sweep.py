import time

import numpy as np
from equipment.custom import mmwave_source
from equipment.hittite import signal_generator
from equipment.srs import lockin

from kid_readout.interactive import *
from kid_readout.equipment import hardware
from kid_readout.measurement import mmw_source_sweep, core, acquire

logger.setLevel(logging.DEBUG)

# fg = FunctionGenerator()
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
source.waveguide_twist_angle = 45
source.ttl_modulation_source = 'roach'

setup = hardware.Hardware(hittite, source, lockin)

ri = hardware_tools.r2_with_mk1()#heterodyne.RoachHeterodyne(adc_valon='/dev/ttyUSB0')
ri.iq_delay = -1
ri.demodulator.hardware_delay_samples = - ri.demodulator.hardware_delay_samples
ri.set_modulation_output(7)

#inital_f0s = np.load('/data/readout/resonances/2016-06-18-jpl-hex-271-initial-lo-1210-resonances.npy')
inital_f0s = np.load('/data/readout/resonances/2016-06-22-lo-1210-128-resonances.npy')
inital_f0s = inital_f0s/1e6
lo = 1210.

nsamp = 2**17
step = 1
nstep = 48
offset_bins = np.arange(-(nstep + 1), (nstep + 1)) * step
offsets = offset_bins * 512.0 / nsamp

mmw_freqs = np.linspace(140e9, 165e9, 500)

ri.set_dac_atten(20)


for (lo,f0s) in [(1210.,inital_f0s),]:
    tic = time.time()
    ncf = new_nc_file(suffix='mmw_sweep_lo_%.1f' % lo)
    ri.set_lo(lo)
    setup.hittite.off()
    swpa = acquire.run_sweep(ri,tone_banks=f0s[None,:]+offsets[:,None],num_tone_samples=2**16,
                             length_seconds=0,
                      verbose=True, state=setup.state())
    print "resonance sweep done", (time.time()-tic)/60.
    sweepstream = mmw_source_sweep.MMWSweepList(swpa, core.IOList(), state=setup.state())
    ncf.write(sweepstream)
    print "sweep written", (time.time()-tic)/60.
    current_f0s = []
    for sidx in range(swpa.num_channels):
        swp = swpa.sweep(sidx)
        res = swp.resonator
        print res.f_0, res.Q, res.delay*1e6, res.current_result.redchi, (f0s[sidx]*1e6-res.f_0)
        if np.abs(f0s[sidx]*1e6-res.f_0) > 100e3:
            current_f0s.append(f0s[sidx]*1e6)
            logger.info("Resonator index %d moved more than 100 kHz, keeping original value %.1f MHz" % (sidx,
                                                                                                         f0s[sidx]))
        else:
            current_f0s.append(res.f_0)
    print "fits complete", (time.time()-tic)/60.
    current_f0s = np.array(current_f0s)/1e6
    current_f0s.sort()
    bad_deltas = np.diff(current_f0s) < (512./2**14)*8
    if bad_deltas.sum():
        print "found bad deltas", bad_deltas.sum()
        current_f0s[np.nonzero(bad_deltas)] -= 0.1
        bad_deltas = np.diff(current_f0s) < (512./2**14)*8
        if bad_deltas.sum():
            print "found bad deltas", bad_deltas.sum()
            current_f0s[np.nonzero(bad_deltas)] -= 0.1

    ri.set_tone_freqs(current_f0s,nsamp=nsamp)
    ri.select_fft_bins(range(128))
    print ri.fpga_fft_readout_indexes
    print np.diff(ri.fpga_fft_readout_indexes.astype('float')).min()
    setup.hittite.on()
    for n, freq in enumerate(mmw_freqs):
        setup.hittite.set_freq(freq/12.)
        time.sleep(0.5)
        tic2= time.time()
        state=setup.state(fast=True)
        print time.time()-tic2
        meas = ri.get_measurement(num_seconds=1., state=state)
        print freq,(time.time()-tic2)
        sweepstream.stream_list.append(meas)

    print "mm-wave sweep complete", (time.time()-tic)/60.
    ncf.close()