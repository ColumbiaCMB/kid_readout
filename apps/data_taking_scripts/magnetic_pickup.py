from __future__ import division
import time
import numpy as np
from kid_readout.measurement.acquire import legacy_acquire
from kid_readout.utils import roach_interface, data_file
from kid_readout.equipment import agilent_33220


def main(f_initial, attenuations, suffix="magnetic_pickup", stream_time=30, coarse_exponent=19, fine_exponent=21,
         transient_wait=10):
    roach = roach_interface.RoachBaseband()
    fg = agilent_33220.FunctionGenerator()

    n_coarse_samples = 2 ** coarse_exponent
    n_fine_samples = 2 ** fine_exponent
    coarse_frequency_resolution = roach.fs / n_coarse_samples
    fine_frequency_resolution = roach.fs / n_fine_samples
    coarse_offset_integers = legacy_acquire.offset_integers[coarse_exponent]
    fine_offset_integers = legacy_acquire.offset_integers[fine_exponent]
    f_coarse_offset = coarse_frequency_resolution * coarse_offset_integers
    f_fine_offset = fine_frequency_resolution * fine_offset_integers

    start_time = time.time()
    df = data_file.DataFile(suffix=suffix)

    for attenuation in attenuations:
        fg.enable_output(False)
        roach.set_dac_attenuator(attenuation)
        print("Set DAC attenuator to {:.1f} dB".format(attenuation))
        df.log_hw_state(roach)
        coarse_sweep = legacy_acquire.sweep(roach, f_initial, f_coarse_offset, n_coarse_samples)
        df.add_sweep(coarse_sweep)
        f_coarse_fit = np.array([r.f_0 for r in legacy_acquire.fit_sweep_data(coarse_sweep)])
        print("coarse [MHz]: " + ', '.join(['{:.3f}'.format(f) for f in f_coarse_fit]))

        df.log_hw_state(roach)
        fine_sweep = legacy_acquire.sweep(roach, f_coarse_fit, f_fine_offset, n_fine_samples)
        df.add_sweep(fine_sweep)
        f_fine_fit = np.array([r.f_0 for r in legacy_acquire.fit_sweep_data(fine_sweep)])
        print("coarse - fine [Hz]: " + ', '.join(['{:.3f}'.format(1e6 * diff)
                                                  for diff in f_fine_fit - f_coarse_fit]))

        f_measured = roach.add_tone_freqs(f_fine_fit)
        roach.select_fft_bins(np.arange(roach.tone_bins.shape[1]))  # Why?
        roach.select_bank(roach.fft_bins.shape[0] - 1)
        roach._sync()
        time.sleep(transient_wait)  # The above commands somehow create a transient that takes about 5 seconds to decay.
        df.log_hw_state(roach)
        off_start_time = time.time()
        off_stream, addresses = roach.get_data_seconds(stream_time, pow2=True)
        df.add_timestream_data(off_stream, roach, off_start_time)
        fg.enable_output(True)
        df.log_hw_state(roach)
        on_start_time = time.time()
        on_stream, addresses = roach.get_data_seconds(stream_time, pow2=True)
        df.add_timestream_data(on_stream, roach, on_start_time)

    fg.enable_output(False)
    df.close()
    print("Completed in {:.0f} minutes: {}".format((time.time() - start_time) / 60, df.filename))

if __name__ == '__main__':
    f_off = np.load('/home/data2/resonances/2014-12-06_140825_0813f8_fit_16.npy')
    attenuation_list = [41, 35]
    main(f_off, attenuation_list)


