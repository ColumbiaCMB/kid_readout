from __future__ import division
import time

import numpy as np

from kid_readout.roach import baseband
from kid_readout.utils import data_file, acquire
from kid_readout.equipment import lockin_controller


def main(f_initial, attenuation, stream_time=30, suffix='compressor_noise', coarse_exponent=19, fine_exponent=21,
         modulation_state='high', modulation_rate=7, transient_wait=10, f_mmw_source=0,
         mmw_atten_turns=(np.nan, np.nan),num_streams=1):
    roach = baseband.RoachBaseband()
    roach.set_modulation_output(modulation_state)
    roach.set_dac_attenuator(attenuation)
    df = data_file.DataFile(suffix=suffix)
    df.log_hw_state(roach)

    def prompt():
        raw_input("Turn off the compressor and hit Enter to begin.")

    if modulation_state == 'low':
        lockin = lockin_controller.lockinController()
        df.nc.mmw_atten_turns = mmw_atten_turns

    # Compressor on
    coarse_sweep_on = acquire.sweep(roach, f_initial, coarse_exponent, transient_wait=transient_wait)
    f_coarse_on = np.array([r.f_0 for r in acquire.fit_sweep_data(coarse_sweep_on)])
    print("Compressor on: coarse - initial [Hz]: " + ', '.join(['{:.0f}'.format(1e6 * delta_f)
                                                              for delta_f in f_coarse_on - f_initial]))
    df.log_hw_state(roach)
    fine_sweep_on = acquire.sweep(roach, f_coarse_on, fine_exponent, transient_wait=transient_wait)
    f_fine_on = np.array([r.f_0 for r in acquire.fit_sweep_data(fine_sweep_on)])
    print("Compressor on: fine - coarse [Hz]: " + ', '.join(['{:.0f}'.format(1e6 * delta_f)
                                                              for delta_f in f_fine_on - f_coarse_on]))
    df.add_sweep(fine_sweep_on)
    df.log_hw_state(roach)
    on_start_time = time.time()
    stream_on, address = acquire.timestream(roach, f_fine_on, stream_time, transient_wait=transient_wait)
    if modulation_state == 'low':
        roach.set_modulation_output(modulation_rate)
        time.sleep(5)
        x, y, r, theta = lockin.get_data()
        roach.set_modulation_output('low')
    else:
        x=0
    df.add_timestream_data(stream_on, roach, on_start_time, mmw_source_freq=f_mmw_source, zbd_voltage=x)

    # Compressor off
    print("Preparing fine sweep with compressor off.")
    df.log_hw_state(roach)
    fine_sweep_off = acquire.sweep(roach, f_coarse_on, fine_exponent, transient_wait=transient_wait, run=prompt)
    print("Done with fine sweep. Turn on the compressor.")
    f_fine_off = np.array([r.f_0 for r in acquire.fit_sweep_data(fine_sweep_off)])
    print("off - on [Hz]: " + ', '.join(['{:.0f}'.format(1e6 * delta_f)
                                                              for delta_f in f_fine_off - f_fine_on]))
    df.add_sweep(fine_sweep_off)
    df.log_hw_state(roach)
    off_start_time = time.time()
    stream_off, address = acquire.timestream(roach, f_fine_on, stream_time, transient_wait=transient_wait, run=prompt)
    print("Done with stream. Turn on the compressor.")
    if modulation_state == 'low':
        roach.set_modulation_output(modulation_rate)
        time.sleep(5)
        x, y, r, theta = lockin.get_data()
        roach.set_modulation_output('high')
    else:
        x = np.nan
    df.add_timestream_data(stream_off, roach, off_start_time, mmw_source_freq=f_mmw_source, zbd_voltage=x)

    df.sync()
    df.close()
    print("Wrote {}".format(df.filename))


if __name__ == "__main__":
    main(np.load('/data/readout/resonances/current.npy')[range(11)+range(13,18)], 35, stream_time=30,
         suffix='compressor_noise_0.08_K', modulation_state='high')
