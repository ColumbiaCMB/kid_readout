import time
import sys

import numpy as np

from kid_readout.roach import heterodyne
from kid_readout.measurement.io import data_file
from kid_readout.measurement.legacy import sweeps
from kid_readout.analysis.resonator.legacy_resonator import fit_best_resonator
from kid_readout.equipment import hittite_controller, lockin_controller


# fg = FunctionGenerator()
hittite = hittite_controller.hittiteController(addr='192.168.0.200')
hittite.set_power(0)
hittite.on()
lockin = lockin_controller.lockinController()
print lockin.get_idn()
ri = heterodyne.RoachHeterodyne(adc_valon='/dev/ttyUSB0')
ri.initialize()
#ri.initialize(use_config=False)
ri.iq_delay = 0
#group_1_lo = 1020.0
#group_2_lo = 1410.0
#all_f0s = np.load('/data/readout/resonances/2016-01-13-jpl-2015-10-park-dark-32-resonances-split-at-1300.npy')

#group_1_f0 = all_f0s[all_f0s < 1300]
#group_2_f0 = all_f0s[all_f0s > 1300]

"""
all_f0s = np.load('/data/readout/resonances/2016-02-12-jpl-park-100nm-32-resonances.npy')
group_1_f0 = all_f0s[all_f0s<1500]
group_2_f0 = all_f0s[all_f0s>1800]

group_1_lo = 1220.0
group_2_lo = 1810.0
"""

all_f0s = np.load('/data/readout/resonances/2016-02-29-jpl-park-2015-10-40nm-al-niobium-gp-two-groups.npy')
group_1_f0 = all_f0s[all_f0s<1300]
group_2_f0 = all_f0s[all_f0s>1300]

group_1_lo = 1030.0
group_2_lo = 1420.0

f0s = group_2_f0#*0.9997
ri.set_lo(group_2_lo)
#responsive_resonances = np.load('/data/readout/resonances/2015-11-26-jpl-nevins-responsive-resonances.npy')

suffix = "mmw_frequency_sweep"
mmw_source_modulation_freq = ri.set_modulation_output(rate=7)
mmw_atten_turns = (7.0,7.0)
print "modulating at: {}".format(mmw_source_modulation_freq),

nf = len(f0s)
atonce = 16
if nf % atonce > 0:
    print "extending list of resonators to make a multiple of ", atonce
    f0s = np.concatenate((f0s, np.arange(1, 1 + atonce - (nf % atonce)) + f0s.max()))

mmw_freqs = np.linspace(140e9, 165e9, 500)

use_fmin = True
attenlist = [0]
start = time.time()
for group_num,(lo,f0s) in enumerate(zip([group_1_lo,group_2_lo],[group_1_f0,group_2_f0])):
    print "group",group_num,"lo",lo,"min f0",f0s.min()
    ri.set_lo(lo)
    for atten in attenlist:
        hittite.off()
        print "setting attenuator to", atten
        ri.set_dac_attenuator(atten)

        nsamp = 2**16
        step = 1
        nstep = 32
        f0binned = np.round(f0s * nsamp / 512.0) * 512.0 / nsamp
        offset_bins = np.arange(-(nstep + 1), (nstep + 1)) * step

        offsets = offset_bins * 512.0 / nsamp

        measured_freqs = sweeps.prepare_sweep(ri, f0binned, offsets, nsamp=nsamp)
        print "loaded waveforms in", (time.time() - start), "seconds"


        delay = -31.3
        print "median delay is ", delay

        df = data_file.DataFile(suffix=suffix)
        df.nc.mmw_atten_turns = mmw_atten_turns
        df.log_hw_state(ri)
        sweep_data = sweeps.do_prepared_sweep(ri, nchan_per_step=atonce, reads_per_step=2)
        df.add_sweep(sweep_data)
        meas_cfs = []
        idxs = []
        for m in range(len(f0s)):
            fr, s21, errors = sweep_data.select_by_freq(f0s[m])
            thiscf = f0s[m]
            s21 = s21 * np.exp(2j * np.pi * delay * fr)
            res = fit_best_resonator(fr, s21, errors=errors)  #Resonator(fr,s21,errors=errors)
            fmin = fr[np.abs(s21).argmin()]
            print "s21 fmin", fmin, "original guess", thiscf, "this fit", res.f_0
            if use_fmin:
                meas_cfs.append(fmin)
            else:
                if abs(res.f_0 - thiscf) > 2:
                    if abs(fmin - thiscf) > 2:
                        print "using original guess"
                        meas_cfs.append(thiscf)
                    else:
                        print "using fmin"
                        meas_cfs.append(fmin)
                else:
                    print "using this fit"
                    meas_cfs.append(res.f_0)
            idx = np.unravel_index(abs(measured_freqs - meas_cfs[-1]).argmin(), measured_freqs.shape)
            idxs.append(idx)
        print meas_cfs
        ri.add_tone_freqs(np.array(meas_cfs))
        ri.select_bank(ri.tone_bins.shape[0] - 1)
    #    ri.set_tone_freqs(responsive_resonances[:32],nsamp=2**15)
        ri.select_fft_bins(range(16))
        ri._sync()
        time.sleep(0.5)

        hittite.on()
        hittite.set_power(0)

        df.log_hw_state(ri)
        nsets = len(meas_cfs) / atonce
        tsg = None
        for iset in range(nsets):
            selection = range(len(meas_cfs))[iset::nsets]
            ri.select_fft_bins(selection)
            ri._sync()
            time.sleep(0.4)
            t0 = time.time()
            for freq in mmw_freqs:
                hittite.set_freq(freq / 12.0)
    #            ri._sync()
                time.sleep(1)
                tt = time.time()
                dmod, addr = ri.get_data(8)
                print time.time() - tt
                x, y, r, theta = lockin.get_data()

                print freq,  #nsets,iset,tsg
                tsg = df.add_timestream_data(dmod, ri, t0, tsg=tsg, mmw_source_freq=freq,
                                             mmw_source_modulation_freq=mmw_source_modulation_freq,
                                             zbd_voltage=r)
            df.sync()
            print "done with sweep"

        df.nc.close()

print "completed in", ((time.time() - start) / 60.0), "minutes"
