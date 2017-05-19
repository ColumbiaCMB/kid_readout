import time
import sys

import numpy as np

from kid_readout.roach import r2baseband
from kid_readout.utils import data_file, r2sweeps, acquire
from kid_readout.analysis.resonator import fit_best_resonator
from kid_readout.analysis.khalil import delayed_generic_guess,delayed_generic_s21
from kid_readout.equipment import hittite_controller, lockin_controller


# fg = FunctionGenerator()
hittite = hittite_controller.hittiteController(addr='192.168.0.200')
lockin = lockin_controller.lockinController()
print lockin.get_idn()
ri = r2baseband.Roach2Baseband(adc_valon='/dev/ttyUSB4')
ri.set_fft_gain(6)
f0s = np.load('/data/readout/resonances/2015-12-20-0813f12-200mK.npy')

mmw_source_modulation_freq = ri.set_modulation_output(rate=7)
mmw_atten_turns = (7.0, 7.0)
print "modulating at: {}".format(mmw_source_modulation_freq),

nf = len(f0s)
atonce = 4
if nf % atonce > 0:
    print "extending list of resonators to make a multiple of ", atonce
    f0s = np.concatenate((f0s, np.arange(1, 1 + atonce - (nf % atonce)) + f0s.max()))

nsamp = 2**18
step = 1
nstep = 10
f0binned = np.round(f0s * nsamp / 512.0) * 512.0 / nsamp
offset_bins = np.arange(-(nstep + 1), (nstep + 1)) * step

offsets = offset_bins * 512.0 / nsamp
offsets = np.concatenate(([offsets.min() - 20e-3, ], offsets, [offsets.max() + 20e-3]))

print f0s
print offsets * 1e6
print len(f0s)

mmw_freqs = np.linspace(140e9, 163e9, 100)

use_fmin = False
atten = 46.0
hittite_power_levels = [0.0]
start = time.time()
for hittite_power_level in hittite_power_levels:
    suffix = "mmw_frequency_sweep_hittite_level_%.3f_dBm" % hittite_power_level

    hittite.set_freq(12.0)
    hittite.on()
    ri.set_modulation_output('high')
    print "setting attenuator to", atten
    ri.set_dac_attenuator(atten)
    orig_sweep_data=None
    if True:
        print "doing coarse sweep"

        coarse_sweep = r2sweeps.do_sweep(ri, center_freqs=f0binned, offsets=offsets,nsamp=nsamp,
                                       nchan_per_step=atonce, reads_per_step=2,)
        orig_sweep_data = coarse_sweep

        coarse_res = acquire.fit_sweep_data(coarse_sweep,model=delayed_generic_s21,guess=delayed_generic_guess,
                                            delay_estimate=ri.hardware_delay_estimate)

        coarse_f0 = np.array([res.f_0 for res in coarse_res])

    else:
        coarse_f0 = f0s
    print coarse_f0

    nsamp = 2 ** 21
    step = 1

    offset_bins = np.array([-8, -4, -2, -1, 0, 1, 2, 4])
    offset_bins = np.concatenate(([-40, -20], offset_bins, [20, 40]))
    offsets = offset_bins * 512.0 / nsamp

    meas_cfs = coarse_f0
    f0binned_meas = np.round(meas_cfs * nsamp / 512.0) * 512.0 / nsamp

    print "doing fine sweep"
    fine_sweep = r2sweeps.do_sweep(ri, center_freqs=f0binned_meas, offsets=offsets,nsamp=nsamp,
                                       nchan_per_step=atonce, reads_per_step=2,sweep_data=orig_sweep_data)

    fine_res = acquire.fit_sweep_data(fine_sweep,model=delayed_generic_s21,guess=delayed_generic_guess,
                                        delay_estimate=ri.hardware_delay_estimate)

    fine_f0 = np.array([res.f_0 for res in fine_res])

    print (coarse_f0-fine_f0)*1e6

    sys.stdout.flush()
    time.sleep(1)

    df = data_file.DataFile(suffix=suffix)
    df.nc.mmw_atten_turns = mmw_atten_turns
    df.nc.hittie_power_level = hittite_power_level
    df.log_hw_state(ri)
    df.add_sweep(fine_sweep)


    ri.set_tone_freqs(fine_f0,nsamp=nsamp)
    ri._sync()
    time.sleep(0.5)

    for probe_atten in [46]:
        ri.set_dac_atten(probe_atten)
        df.log_hw_state(ri)
        nsets = len(meas_cfs) / atonce
        tsg = None
        for iset in range(nsets):
            selection = range(len(meas_cfs))[iset::nsets]
            ri.select_fft_bins(selection)
            ri._sync()
            time.sleep(0.4)
            t0 = time.time()
            dmod, addr = ri.get_data_katcp(32*4*8*8)
            tsg = df.add_timestream_data(dmod, ri, t0, tsg=tsg)

    ri.set_modulation_output(7)
    for probe_atten in [46]:
        ri.set_dac_atten(probe_atten)
        df.log_hw_state(ri)
        nsets = len(meas_cfs) / atonce
        tsg = None
        for iset in range(nsets):
            selection = range(len(meas_cfs))[iset::nsets]
            ri.select_fft_bins(selection)
            ri._sync()
            time.sleep(0.4)
            t0 = time.time()
            dmod, addr = ri.get_data_katcp(32)
            tsg = df.add_timestream_data(dmod, ri, t0, tsg=tsg)

    ri.set_dac_atten(46)

    df.close()
    7/0
    hittite.on()
    hittite.set_power(hittite_power_level)

    df.log_hw_state(ri)
    nsets = len(meas_cfs) / atonce
    tsg = None
    for freq in mmw_freqs:
        hittite.set_freq(freq / 12.0)
        time.sleep(0.1)
        x, y, r, theta = lockin.get_data()

        print freq,  #nsets,iset,tsg
        for iset in range(nsets):
            selection = range(len(meas_cfs))[iset::nsets]
            ri.select_fft_bins(selection)
            ri._sync()
            time.sleep(0.4)
            t0 = time.time()
            dmod, addr = ri.get_data_katcp(32)
            tsg = df.add_timestream_data(dmod, ri, t0, tsg=tsg, mmw_source_freq=freq,
                                         mmw_source_modulation_freq=mmw_source_modulation_freq,
                                         zbd_voltage=x)
        df.sync()
        print "done with sweep"

    df.nc.close()

print "completed in", ((time.time() - start) / 60.0), "minutes"
