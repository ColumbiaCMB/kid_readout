from __future__ import division
import time

import numpy as np

from kid_readout.roach import baseband
from kid_readout.utils import data_file, acquire, sweeps
from kid_readout.equipment import lockin_controller


f_initial = np.load('/data/readout/resonances/0813f12_130mK_fits.npy')

stream_time=1800
num_streams=2
suffix='long_timestreams_80mK'
coarse_exponent=19
fine_exponent=21

roach = baseband.RoachBaseband()
roach.set_modulation_output('high')
roach.set_dac_attenuator(38)

coarse_sweep_on = acquire.sweep(roach, f_initial, coarse_exponent)
f_fine_on = np.array([r.f_0 for r in acquire.fit_sweep_data(coarse_sweep_on)])


for loop_index,atten in enumerate([40,38,36,34,32]):

    roach.set_dac_attenuator(atten)

    df = data_file.DataFile(suffix=suffix)
    df.log_hw_state(roach)

    if True:#loop_index == 0 :
        fine_sweep_on = acquire.sweep(roach, f_fine_on, fine_exponent, transient_wait=0)
    else:
        fine_sweep_on = sweeps.do_prepared_sweep(roach,nchan_per_step=len(f_initial))
    f_fine_on = np.array([r.f_0 for r in acquire.fit_sweep_data(fine_sweep_on)])
    print("Compressor on: fine - coarse [Hz]: " + ', '.join(['{:.0f}'.format(1e6 * delta_f)
                                                              for delta_f in f_fine_on - f_initial]))
    df.add_sweep(fine_sweep_on)
    df.log_hw_state(roach)
    for k in range(num_streams):
        on_start_time = time.time()
        if k == 0:
            stream_on, address = acquire.timestream(roach, f_fine_on, stream_time, transient_wait=0)
        else:
            sweep = sweeps.do_prepared_sweep(roach,nchan_per_step=16)
            df.add_sweep(sweep)
            roach.select_bank(roach.tone_bins.shape[0] - 1)
            roach.select_fft_bins(range(roach.fft_bins.shape[1]))
            roach._sync()
            stream_on,addr = roach.get_data_seconds(stream_time)

        df.add_timestream_data(stream_on, roach, on_start_time)

        df.sync()
        print "finished", (k+1),"timestreams at attenuation",atten
    df.close()
    print("Wrote {}".format(df.filename))

