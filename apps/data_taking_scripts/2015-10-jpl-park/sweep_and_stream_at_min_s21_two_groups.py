__author__ = 'gjones'
import time
import sys

import numpy as np

from kid_readout.roach import heterodyne
from kid_readout.utils import data_file, sweeps
from kid_readout.equipment import hittite_controller, lockin_controller


hittite = hittite_controller.hittiteController(addr='192.168.0.200')
lockin = lockin_controller.lockinController()
print lockin.get_idn()
ri = heterodyne.RoachHeterodyne(adc_valon='/dev/ttyUSB0')
ri.iq_delay = 0
ri.set_lo(1410.0)

#group_1_lo = 1020.0
#group_2_lo = 1410.0
#all_f0s = np.load('/data/readout/resonances/2016-01-13-jpl-2015-10-park-dark-32-resonances-split-at-1300.npy') -0.5

#group_1_f0 = all_f0s[all_f0s < 1300]
#group_2_f0 = all_f0s[all_f0s > 1300]
"""
all_f0s = np.load('/data/readout/resonances/2016-02-12-jpl-park-100nm-32-resonances.npy')
group_1_f0 = all_f0s[all_f0s<1500]
group_2_f0 = all_f0s[all_f0s>1800]

group_1_lo = 1220.0
group_2_lo = 1810.0
"""

all_f0s = np.load('/data/readout/resonances/2016-02-20-jpl-park-2015-10-40nm-al-niobium-gp-two-groups.npy')
group_1_f0 = all_f0s[all_f0s<1300]
group_2_f0 = all_f0s[all_f0s>1300]

group_1_lo = 1030.0
group_2_lo = 1420.0


#responsive_resonances = np.load('/data/readout/resonances/2015-11-26-jpl-nevins-responsive-resonances.npy')

suffix = "sweep_and_stream"
mmw_source_modulation_freq = ri.set_modulation_output(rate=7)
mmw_source_frequency = 148e9
hittite.set_freq(mmw_source_frequency/12.0)
mmw_atten_turns = (6.0, 6.0)
#print "modulating at: {}".format(mmw_source_modulation_freq),

atonce = 16



df = data_file.DataFile(suffix=suffix)
df.nc.mmw_atten_turns = mmw_atten_turns
for group_num,(lo,f0s) in enumerate(zip([group_1_lo,group_2_lo],[group_1_f0,group_2_f0])):
    print "group",group_num,"lo",lo,"min f0",f0s.min()

    ri.set_lo(lo)
    nsamp = 2**16
    step = 1
    nstep = 32
    f0binned = np.round(f0s * nsamp / 512.0) * 512.0 / nsamp
    offset_bins = np.arange(-(nstep + 1), (nstep + 1)) * step

    offsets = offset_bins * 512.0 / nsamp
    measured_freqs = sweeps.prepare_sweep(ri, f0binned, offsets, nsamp=nsamp)
    for atten_index,dac_atten in enumerate([14,8,2,0,20]):
        print "at dac atten", dac_atten
        ri.set_dac_atten(dac_atten)
        ri.set_modulation_output('high')
        df.log_hw_state(ri)
        df.log_adc_snap(ri)
        sweep_data = sweeps.do_prepared_sweep(ri, nchan_per_step=atonce, reads_per_step=2)
        df.add_sweep(sweep_data)
        fmins = []
        for k in range(len(f0s)):
            fr, s21, errors = sweep_data.select_index(k)
            fmins.append(fr[np.abs(s21).argmin()])
        fmins.sort()
        ri.add_tone_freqs(np.array(fmins))
        ri.select_bank(ri.tone_bins.shape[0] - 1)
    #    ri.set_tone_freqs(responsive_resonances[:32],nsamp=2**15)
        ri.select_fft_bins(range(len(f0s)))
        ri._sync()
        time.sleep(0.5)

        print "taking data with source off"
        ri.set_modulation_output('high')
        df.log_hw_state(ri)
        nsets = len(f0s) / atonce
        tsg = None
        for iset in range(nsets):
            selection = range(len(f0s))[iset::nsets]
            ri.select_fft_bins(selection)
            ri._sync()
            time.sleep(0.4)
            t0 = time.time()
            dmod, addr = ri.get_data(256) # about 30 seconds of data
    #        x, y, r, theta = lockin.get_data()

            tsg = df.add_timestream_data(dmod, ri, t0, tsg=tsg)
            df.sync()

        print "taking data with source modulated"
        ri.set_modulation_output(7)
        df.log_hw_state(ri)
        nsets = len(f0s) / atonce
        tsg = None
        for iset in range(nsets):
            selection = range(len(f0s))[iset::nsets]
            ri.select_fft_bins(selection)
            ri._sync()
            time.sleep(0.4)
            t0 = time.time()
            dmod, addr = ri.get_data(16) # about 2 seconds of data
            x, y, r, theta = lockin.get_data()

            tsg = df.add_timestream_data(dmod, ri, t0, tsg=tsg,zbd_voltage=r,mmw_source_freq=mmw_source_frequency)
            df.sync()

        ri.set_modulation_output('high')

df.close()