from __future__ import division
import time

import numpy as np

from kid_readout.roach import baseband
from kid_readout.utils import data_file, sweeps, acquire
from kid_readout.equipment import lockin_controller


def main(f_initial_off, f_initial_on, attenuations, f_mmw_source=0, suffix="mmw", hittite_power=0, long_stream_time=30,
         modulated_stream_time=4, coarse_exponent=19, fine_exponent=21, modulation_rate_integer=7):
    if f_mmw_source:
        from kid_readout.equipment import hittite_controller

        frequency_multiplication_factor = 12
        hittite = hittite_controller.hittiteController()
        hittite.set_power(hittite_power)  # in dBm
        hittite.set_freq(f_mmw_source / frequency_multiplication_factor)  # in Hz
        hittite.on()

    lockin = lockin_controller.lockinController()
    print(lockin.get_idn())

    roach = baseband.RoachBaseband()

    n_coarse_samples = 2 ** coarse_exponent
    n_fine_samples = 2 ** fine_exponent
    coarse_frequency_resolution = roach.fs / n_coarse_samples
    fine_frequency_resolution = roach.fs / n_fine_samples
    # Drop the 31st point so as to not force the added points to occupy the 32nd memory slot, which may not work.
    coarse_offset_integers = acquire.offset_integers[coarse_exponent]  # [:-1]
    fine_offset_integers = acquire.offset_integers[fine_exponent]  # [:-1]
    f_coarse_offset = coarse_frequency_resolution * coarse_offset_integers
    f_fine_offset = fine_frequency_resolution * fine_offset_integers

    while True:
        f_source_modulation = roach.set_modulation_output(modulation_rate_integer)
        print("\nSet source modulation frequency to {:.1f} Hz. Check the lock-in.".format(f_source_modulation))
        try:
            mmw_attenuator_turns = float(
                raw_input("Enter the value to which both attenuators are set, or hit Enter to stop recording data: "))
        except ValueError:
            break

        start_time = time.time()
        df = data_file.DataFile(suffix=suffix)
        df.nc.mmw_atten_turns = (mmw_attenuator_turns, mmw_attenuator_turns)
        maximum_attenuation = max(attenuations)
        print("Setting DAC attenuator to maximum requested attenuation of {:.1f} dB.".format(maximum_attenuation))
        roach.set_dac_attenuator(maximum_attenuation)

        # At the lowest readout power, record a coarse sweep with the source off and a modulated stream at the fit
        # source-off resonances.
        sweeps.prepare_sweep(roach, f_initial_off, f_coarse_offset, n_coarse_samples)
        coarse_sweep_off, stream_mod_off = acquire.mmw_source_sweep_and_stream(df, roach, lockin, modulated_stream_time, False,
                                                                    f_mmw_source, 'high', modulation_rate_integer,
                                                                    modulation_rate_integer)
        f_coarse_fit_off = np.array([r.f_0 for r in acquire.fit_sweep_data(coarse_sweep_off)])
        print("Source off: coarse - initial [Hz]: " + ', '.join(['{:.0f}'.format(1e6 * diff)
                                                                 for diff in f_coarse_fit_off - f_initial_off]))

        # At the lowest readout power, record a fine sweep and long stream with the source off.
        sweeps.prepare_sweep(roach, f_coarse_fit_off, f_fine_offset, n_fine_samples)
        fine_sweep_off, stream_off = acquire.mmw_source_sweep_and_stream(df, roach, lockin, long_stream_time, False,
                                                              f_mmw_source, 'high', 'high', modulation_rate_integer)
        f_fine_fit_off = np.array([r.f_0 for r in acquire.fit_sweep_data(fine_sweep_off)])
        print("Source off: fine - coarse [Hz]: " + ', '.join(['{:.0f}'.format(1e6 * diff)
                                                              for diff in f_fine_fit_off - f_coarse_fit_off]))

        # At the lowest readout power, record a coarse sweep with the source on and a modulated stream at the fit
        # source-on resonances.
        sweeps.prepare_sweep(roach, f_initial_on, f_coarse_offset, n_coarse_samples)
        coarse_sweep_on, stream_mod_on = acquire.mmw_source_sweep_and_stream(df, roach, lockin, modulated_stream_time, False,
                                                                  f_mmw_source,
                                                                  'low', modulation_rate_integer,
                                                                  modulation_rate_integer)
        f_coarse_fit_on = np.array([r.f_0 for r in acquire.fit_sweep_data(coarse_sweep_on)])
        print("Source on: coarse - initial [Hz]: " + ', '.join(['{:.0f}'.format(1e6 * diff)
                                                                for diff in f_coarse_fit_on - f_initial_on]))

        # Use these frequencies for all subsequent sweeps, and add an additional waveform for each stream.
        print("\nSetting fine sweep frequencies for source-on measurements.")
        sweeps.prepare_sweep(roach, f_coarse_fit_on, f_fine_offset, n_fine_samples)

        for k, attenuation in enumerate(attenuations):
            print("\nSource-on measurement {} of {}: DAC attenuator at {:.1f} dB.".format(k + 1, len(attenuations),
                                                                                          attenuation))
            roach.set_dac_attenuator(attenuation)
            fine_sweep_on, stream = acquire.mmw_source_sweep_and_stream(df, roach, lockin, long_stream_time,
                                                             k > 0,  # overwrite after the first.
                                                             f_mmw_source, 'low', 'low', modulation_rate_integer)
            f_fine_fit_on = [r.f_0 for r in acquire.fit_sweep_data(fine_sweep_on)]
            print("Source on: fine - coarse [Hz]: " + ', '.join(['{:.0f}'.format(1e6 * diff)
                                                                 for diff in f_fine_fit_on - f_coarse_fit_on]))

        df.close()
        print("Completed in {:.0f} minutes: {}".format((time.time() - start_time) / 60, df.filename))

    # Clean up.
    if f_mmw_source:
        hittite.off()
        hittite.disconnect()


if __name__ == '__main__':
    f_off = np.load('/home/data2/resonances/2014-12-06_140825_0813f8_fit_16.npy')
    f_on = (1 - 0e-6) * f_off
    attenuation_list = [41, 38, 35, 32, 29, 26, 23]
    f_mmw = 0  #156e9

    main(f_off, f_on, attenuation_list, f_mmw_source=f_mmw, suffix='mmw_noise_narrowband')
