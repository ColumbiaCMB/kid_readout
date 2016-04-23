import time
import sys

import numpy as np

from kid_readout.roach import analog
from kid_readout import *
from kid_readout.measurement.acquire import acquire
from equipment.hittite import signal_generator
from equipment.custom import mmwave_source
from equipment.srs import lockin
from kid_readout.measurement.acquire import hardware
from kid_readout.measurement import mmw_source_sweep, core, basic

# fg = FunctionGenerator()
#hittite = signal_generator.Hittite(ipaddr='192.168.0.200')
#hittite.set_power(0)
#hittite.on()
lockin = lockin.Lockin('/dev/ttyUSB2')
tic = time.time()
print lockin.identification
print time.time()-tic
print lockin.state()
print time.time()-tic
source = mmwave_source.MMWaveSource()
source.set_attenuator_turns(5.0,5.0)
source.multiplier_input = 'thermal'
source.waveguide_twist_angle = 45
source.ttl_modulation_source = 'roach'

ifboard = analog.HeterodyneMarkI()

setup = hardware.Hardware(source,lockin,ifboard)
#setup.hittite.set_freq(148e9/12.)


ri = heterodyne.RoachHeterodyne(adc_valon='/dev/ttyUSB0')
ri.initialize()
#ri.initialize(use_config=False)
ri.iq_delay = 0


low_group = np.array([ 1119.07673511,  1126.0153982 ,  1133.70131753,  1135.31245427,
        1143.21576741,  1148.0169738 ,  1159.41786884,  1160.26943775, 1162.33,
        1176.56747357,  1177.70848608,  1181.33728799,  1186.05472363,
        1190.29108718,  1193.62180971,  1203.87755917,  1205.8392229 ,
        1207.88046178,  1220.05286255,  1221.41371349,  1235.26613658, 1243.71])

to_add = 32 - len(low_group)
low_group = np.hstack((low_group,np.arange(to_add)+low_group.max()+2.))
low_group_lo = 1110.0

high_group = np.array([1576.67969747,  1599.40494104,  1605.60503541,
        1609.7742154 ,
        1623.02,
                       1628.5448411 ,  1648.17537644,  1649.22774551,
        1650.36575278,  1662.37845567,  1664.1706628 ,  1681.28526333,
        1682.17350834,  1684.44471064,  1702.0820426 ,  1706.79635917,
        1714.66047719,
                       1720.7, 1724.39938475,  1752.05587464,  1767.515093])

to_add = 32 - len(high_group)
high_group = np.hstack((high_group,np.arange(to_add)+high_group.max()+2.))
high_group_lo = 1570.0

nsamp = 2**16
step = 1
nstep = 48
offset_bins = np.arange(-(nstep + 1), (nstep + 1)) * step
offsets = offset_bins * 512.0 / nsamp

mmw_freqs = np.linspace(140e9, 165e9, 500)

ri.set_dac_atten(2)


for (lo,f0s) in [(low_group_lo,low_group),
                 (high_group_lo, high_group)]:
    ri.set_lo(lo)
    tic = time.time()
    measured_frequencies = acquire.load_heterodyne_sweep_tones(ri,np.add.outer(offsets,f0s),num_tone_samples=nsamp)
    print "waveforms loaded", (time.time()-tic)/60.
    for dac_atten in [20,10,6,2]:
        ncf = new_nc_file(suffix='off_on_modulated_broadband_%d_dB_dac' % dac_atten)
        ri.set_modulation_output('high')
        swpa = acquire.run_loaded_sweep(ri,length_seconds=0,state=setup.state(),description='source off sweep')
        print "resonance sweep done", (time.time()-tic)/60.
        ncf.write(swpa)
        #print "sweep written", (time.time()-tic)/60.
        current_f0s = []
        for sidx in range(32):
            swp = swpa.sweep(sidx)
            res = lmfit_resonator.LinearResonatorWithCable(swp.frequency,swp.s21_points,swp.s21_points_error)
            print res.f_0, res.Q, res.current_result.redchi, (f0s[sidx]*1e6-res.f_0)
            if sidx not in [15,17] and np.abs(res.f_0 - f0s[sidx]*1e6) > 200e3:
                current_f0s.append(f0s[sidx]*1e6)
                print "using original frequency for ",f0s[sidx]
            else:
                current_f0s.append(res.f_0)
        print "fits complete", (time.time()-tic)/60.
        current_f0s = np.array(current_f0s)/1e6
        current_f0s.sort()
        if np.any(np.diff(current_f0s)<0.1):
            print "problematic resonator collision:",current_f0s
            print "deltas:",np.diff(current_f0s)
        ri.add_tone_freqs(current_f0s,overwrite_last=True)
        ri.select_bank(ri.tone_bins.shape[0]-1)
        ri.select_fft_bins(range(32))
        meas = ri.get_measurement(num_seconds=30., state=setup.state(),description='source off stream')
        ncf.write(meas)

        ri.set_modulation_output('low')
        meas = ri.get_measurement(num_seconds=30., state=setup.state(),description='source on stream')
        ncf.write(meas)

        ri.set_modulation_output(7)
        meas = ri.get_measurement(num_seconds=4., state=setup.state(),description='source modulated stream')
        ncf.write(meas)
        print "dac_atten %f done in %.1f minutes" % (dac_atten, (time.time()-tic)/60.)
        ncf.close()