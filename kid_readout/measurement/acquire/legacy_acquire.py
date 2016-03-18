from __future__ import division
import time

import numpy as np

from kid_readout.roach import baseband
from kid_readout.measurement.io import data_file
from kid_readout.measurement.acquire import sweeps
from kid_readout.analysis.resonator.legacy_resonator import Resonator
from kid_readout.analysis.resonator.khalil import bifurcation_s21, bifurcation_guess
from kid_readout.equipment import lockin_controller

# TODO: move the memory management to the Roach class
BYTES_PER_SAMPLE = 4
EFFECTIVE_DRAM_CAPACITY = 2 ** 28  # 256 MB


def memory_usage_bytes(n_waveforms, n_samples):
    return n_waveforms * n_samples * BYTES_PER_SAMPLE


# These are intended to be good offsets to use with 2^key tone samples. Each array almost fills the Roach memory, with
# enough space left for one more waveform at the same number of tone samples.
offset_integers = {18: np.arange(-127, 128),
                   19: np.arange(-63, 64),
                   20: np.concatenate([np.arange(-42, -20, 2),
                                       np.arange(-20, 21),
                                       np.arange(22, 44, 2)]),
                   21: np.concatenate([np.arange(-35, -10, 5),
                                       np.arange(-10, 11),
                                       np.arange(15, 40, 5)]),
                   22: np.array([-36, -28, -15, -10, -6, -3, -1,
                                 0, 1, 3, 6, 10, 15, 28, 36])}


def round_frequencies(frequencies, sample_frequency, n_samples):
    frequency_resolution = sample_frequency / n_samples
    return frequency_resolution * np.round(frequencies / frequency_resolution)


def combine_center_and_offset_frequencies(center_frequencies, offset_frequencies):
    """
    Combine center and offset frequencies into a single array that matches the Roach format.

    :param center_frequencies: an array containing center frequencies of the sweep.
    :param offset_frequencies: an array containing offset frequencies of the sweep.
    :return: an array with shape (offset_frequencies.size, center_frequencies.size) containing the combined frequencies.
    """
    return center_frequencies[np.newaxis, :] + offset_frequencies[:, np.newaxis]


def fit_sweep_data(sweep_data, model=bifurcation_s21, guess=bifurcation_guess, delay_estimate=0):
    resonators = []
    for i in np.unique(sweep_data.sweep_indexes):
        f, s21, errors = sweep_data.select_index(i)
        s21 *= np.exp(2j * np.pi * delay_estimate * f)
        resonators.append(Resonator(f, s21, errors=errors, model=model, guess=guess))
    return sorted(resonators, key=lambda r: r.f_0)


def sweep(roach, center_frequencies, sample_exponent, offset_frequencies=None, reads_per_step=2, transient_wait=0,
          run=lambda: None):
    n_samples = 2 ** sample_exponent
    if offset_frequencies is None:
        frequency_resolution = roach.fs / n_samples
        offset_frequencies = frequency_resolution * offset_integers[sample_exponent]
    sweeps.prepare_sweep(roach, center_frequencies, offset_frequencies, n_samples)
    roach._sync()
    time.sleep(transient_wait)
    run()
    sweep_data = sweeps.do_prepared_sweep(roach, nchan_per_step=len(center_frequencies), reads_per_step=reads_per_step)
    return sweep_data


def timestream(roach, frequencies, time_in_seconds, pow2=True, overwrite_last=False, transient_wait=0,
               run=lambda: None):
    roach.add_tone_freqs(frequencies, overwrite_last=overwrite_last)
    roach.select_bank(roach.tone_bins.shape[0] - 1)
    roach._sync()
    time.sleep(transient_wait)
    run()
    data, address = roach.get_data_seconds(time_in_seconds, demod=True, pow2=pow2)
    return data, address


# TODO: rename
def sweeps_and_streams(f_initial, attenuations, suffix='', coarse_exponent=19, fine_exponent=21, long_stream_time=30,
                       short_stream_time=4, roach_wait=10):
    roach = baseband.RoachBaseband()
    f_modulation = roach.set_modulation_output('high')

    n_coarse_samples = 2 ** coarse_exponent
    n_fine_samples = 2 ** fine_exponent
    coarse_frequency_resolution = roach.fs / n_coarse_samples
    fine_frequency_resolution = roach.fs / n_fine_samples
    coarse_offset_integers = offset_integers[coarse_exponent]
    fine_offset_integers = offset_integers[fine_exponent]
    f_coarse_offset = coarse_frequency_resolution * coarse_offset_integers
    f_fine_offset = fine_frequency_resolution * fine_offset_integers

    start_time = time.time()
    df = data_file.DataFile(suffix=suffix)
    maximum_attenuation = max(attenuations)
    print("Setting DAC attenuator to maximum requested attenuation of {:.1f} dB.".format(maximum_attenuation))
    roach.set_dac_attenuator(maximum_attenuation)

    # At the lowest readout power, record a coarse sweep and a short stream
    sweeps.prepare_sweep(roach, f_initial, f_coarse_offset, n_coarse_samples)
    df.log_hw_state(roach)
    coarse_sweep_data = sweeps.do_prepared_sweep(roach, nchan_per_step=roach.tone_bins.shape[0], reads_per_step=2)
    df.add_sweep(coarse_sweep_data)
    df.sync()
    coarse_f_fit = np.array([r.f_0 for r in fit_sweep_data(coarse_sweep_data)])
    print("coarse - initial [Hz]: " + ', '.join(['{:.0f}'.format(1e6 * diff) for diff in coarse_f_fit - f_initial]))
    roach.add_tone_freqs(coarse_f_fit, overwrite_last=True)
    roach.select_bank(roach.fft_bins.shape[0] - 1)
    roach._sync()
    time.sleep(roach_wait)  # The above commands somehow create a transient that takes about 5 seconds to decay.
    df.log_hw_state(roach)
    stream_start_time = time.time()
    stream, addresses = roach.get_data_seconds(short_stream_time, pow2=True)
    df.add_timestream_data(stream, roach, stream_start_time, mmw_source_modulation_freq=f_modulation)
    df.sync()

    # Use these frequencies for all subsequent sweeps, and add an additional waveform for each stream.
    print("\nSetting fine sweep frequencies.")
    sweeps.prepare_sweep(roach, coarse_f_fit, f_fine_offset, n_fine_samples)

    for k, attenuation in enumerate(attenuations):
        print("\nMeasurement {} of {}: DAC attenuator at {:.1f} dB.".format(k + 1, len(attenuations), attenuation))
        roach.set_dac_attenuator(attenuation)
        fine_sweep_data = sweeps.do_prepared_sweep(roach, nchan_per_step=roach.tone_bins.shape[0], reads_per_step=2)
        df.add_sweep(fine_sweep_data)
        df.sync()
        fine_f_fit = np.array([r.f_0 for r in fit_sweep_data(fine_sweep_data)])
        print("fine - coarse [Hz]: " + ', '.join(['{:.0f}'.format(1e6 * diff) for diff in fine_f_fit - coarse_f_fit]))
        f_stream = roach.add_tone_freqs(coarse_f_fit, overwrite_last=k>0)  # Overwrite after the first
        print("stream detuning [ppm]: " + ', '.join(['{:.0f}'.format(1e6 * x) for x in (f_stream / fine_f_fit - 1)]))
        roach.select_bank(roach.fft_bins.shape[0] - 1)
        roach._sync()
        time.sleep(roach_wait)  # The above commands somehow create a transient that takes about 5 seconds to decay.
        df.log_hw_state(roach)
        stream_start_time = time.time()
        stream, addresses = roach.get_data_seconds(long_stream_time, pow2=True)
        df.add_timestream_data(stream, roach, stream_start_time, mmw_source_modulation_freq=f_modulation)
        df.sync()

    df.close()
    print("Completed in {:.0f} minutes: {}".format((time.time() - start_time) / 60, df.filename))



def mmw_source_sweep_and_stream(df, roach, lockin, approximate_stream_time, overwrite_last, f_mmw_source,
                                sweep_modulation_rate, stream_modulation_rate, measurement_modulation_rate,
                                roach_wait=10, lockin_wait=5, verbose=True):
    f_sweep_modulation = roach.set_modulation_output(sweep_modulation_rate)
    if verbose:
        print("Sweep modulation state {}: frequency {:.2f} Hz.".format(sweep_modulation_rate, f_sweep_modulation))
    df.log_hw_state(roach)
    sweep = sweeps.do_prepared_sweep(roach, nchan_per_step=roach.tone_bins.shape[0], reads_per_step=2)
    df.add_sweep(sweep)
    df.sync()
    f_fit = np.array([r.f_0 for r in fit_sweep_data(sweep)])

    f_measurement_modulation = roach.set_modulation_output(measurement_modulation_rate)
    f_stream_measured = roach.add_tone_freqs(f_fit, overwrite_last=overwrite_last)  # Use this delay for settling
    if verbose:
        print("Stream tone separations [MHz]: {}".format(np.diff(sorted(f_stream_measured))))
    time.sleep(lockin_wait)
    x, y, r, theta = lockin.get_data()
    if verbose:
        print("Lock-in measured {:.4g} V at frequency {:.2f} Hz.".format(x, f_measurement_modulation))

    f_stream_modulation = roach.set_modulation_output(stream_modulation_rate)
    if verbose:
        print("Modulation state {} for {:.0f} second stream: frequency {:.2f} Hz.".format(stream_modulation_rate,
                                                                                          approximate_stream_time,
                                                                                          f_stream_modulation))
    # After 2015-05-05, select_fft_bins is no longer necessary since select_bank now selects FFT bins too.
    roach.select_bank(roach.fft_bins.shape[0] - 1)
    roach._sync()
    time.sleep(roach_wait)  # The above commands somehow create a transient that takes about 5 seconds to decay.
    df.log_hw_state(roach)
    start_time = time.time()
    stream, addresses = roach.get_data_seconds(approximate_stream_time, pow2=True)
    df.add_timestream_data(stream, roach, start_time, mmw_source_freq=f_mmw_source,
                           mmw_source_modulation_freq=f_stream_modulation, zbd_voltage=x)
    df.sync()
    return sweep, stream


def mmw_source_power_step(f_initial_off, f_initial_on, attenuations, f_mmw_source=0, suffix="mmw", hittite_power=0,
                          long_stream_time=30,
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
    coarse_offset_integers = offset_integers[coarse_exponent]
    fine_offset_integers = offset_integers[fine_exponent]
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
        coarse_sweep_off, stream_mod_off = mmw_source_sweep_and_stream(df, roach, lockin, modulated_stream_time,
                                                                       False,
                                                                       f_mmw_source, 'high',
                                                                       modulation_rate_integer,
                                                                       modulation_rate_integer)
        f_coarse_fit_off = np.array([r.f_0 for r in fit_sweep_data(coarse_sweep_off)])
        print("Source off: coarse - initial [Hz]: " + ', '.join(['{:.0f}'.format(1e6 * diff)
                                                                 for diff in f_coarse_fit_off - f_initial_off]))

        # At the lowest readout power, record a fine sweep and long stream with the source off.
        sweeps.prepare_sweep(roach, f_coarse_fit_off, f_fine_offset, n_fine_samples)
        fine_sweep_off, stream_off = mmw_source_sweep_and_stream(df, roach, lockin, long_stream_time, False,
                                                                 f_mmw_source, 'high', 'high',
                                                                 modulation_rate_integer)
        f_fine_fit_off = np.array([r.f_0 for r in fit_sweep_data(fine_sweep_off)])
        print("Source off: fine - coarse [Hz]: " + ', '.join(['{:.0f}'.format(1e6 * diff)
                                                              for diff in f_fine_fit_off - f_coarse_fit_off]))

        # At the lowest readout power, record a coarse sweep with the source on and a modulated stream at the fit
        # source-on resonances.
        sweeps.prepare_sweep(roach, f_initial_on, f_coarse_offset, n_coarse_samples)
        coarse_sweep_on, stream_mod_on = mmw_source_sweep_and_stream(df, roach, lockin, modulated_stream_time,
                                                                     False,
                                                                     f_mmw_source,
                                                                     'low', modulation_rate_integer,
                                                                     modulation_rate_integer)
        f_coarse_fit_on = np.array([r.f_0 for r in fit_sweep_data(coarse_sweep_on)])
        print("Source on: coarse - initial [Hz]: " + ', '.join(['{:.0f}'.format(1e6 * diff)
                                                                for diff in f_coarse_fit_on - f_initial_on]))

        # Use these frequencies for all subsequent sweeps, and add an additional waveform for each stream.
        print("\nSetting fine sweep frequencies for source-on measurements.")
        sweeps.prepare_sweep(roach, f_coarse_fit_on, f_fine_offset, n_fine_samples)

        for k, attenuation in enumerate(attenuations):
            print("\nSource-on measurement {} of {}: DAC attenuator at {:.1f} dB.".format(k + 1, len(attenuations),
                                                                                          attenuation))
            roach.set_dac_attenuator(attenuation)
            fine_sweep_on, stream = mmw_source_sweep_and_stream(df, roach, lockin, long_stream_time,
                                                                k > 0,  # overwrite after the first.
                                                                f_mmw_source, 'low', 'low',
                                                                modulation_rate_integer)
            f_fine_fit_on = [r.f_0 for r in fit_sweep_data(fine_sweep_on)]
            print("Source on: fine - coarse [Hz]: " + ', '.join(['{:.0f}'.format(1e6 * diff)
                                                                 for diff in f_fine_fit_on - f_coarse_fit_on]))

        df.close()
        print("Completed in {:.0f} minutes: {}".format((time.time() - start_time) / 60, df.filename))

    # Clean up.
    if f_mmw_source:
        hittite.off()
        hittite.disconnect()


# Everything below is old and may not work.

def record_sweep(roach, center_frequencies, offset_frequencies, attenuation, n_samples, suffix, interactive=False):
    n_channels = center_frequencies.size
    df = data_file.DataFile(suffix=suffix)
    print("Writing data to " + df.filename)
    print("Setting DAC attenuator to {:.1f} dB".format(attenuation))
    roach.set_dac_attenuator(attenuation)
    print("Sweep memory usage is {:.1f} MB of {:.1f} MB capacity.".format(
        memory_usage_bytes(offset_frequencies.shape[0], n_samples) / 2 ** 20, EFFECTIVE_DRAM_CAPACITY / 2 ** 20))

    measured_frequencies = sweeps.prepare_sweep(roach, center_frequencies, offset_frequencies, n_samples)
    roach._sync()
    time.sleep(0.2)
    df.log_hw_state(roach)
    if interactive:
        raw_input("Hit enter to begin recording frequency sweep.")
    else:
        print("Recording frequency sweep.")
    sweep_start_time = time.time()
    sweep_data = sweeps.do_prepared_sweep(roach, nchan_per_step=n_channels, reads_per_step=8)
    print("Elapsed time {:.0f} seconds. Writing to disk.".format(time.time() - sweep_start_time))
    df.add_sweep(sweep_data)
    df.sync()
    resonators = fit_sweep_data(sweep_data)
    fit_f0s = np.array([r.f_0 for r in resonators])
    print("Initial frequencies in MHz are " + ', '.join(['{:.3f}'.format(f0) for f0 in center_frequencies]))
    print("initial - fit [Hz]: " +
          ', '.join(['{:.0f}'.format(1e6 * delta_f) for delta_f in center_frequencies - fit_f0s]))
    df.nc.close()
    return df.filename, resonators


def record_sweeps_on_off(roach, nominal_frequencies, attenuation, n_samples, suffix):
    df = data_file.DataFile(suffix=suffix)
    roach.set_dac_attenuator(attenuation)
    measured_frequencies = roach.set_tone_frequencies(nominal_frequencies, nsamp=n_samples)
    df.log_hw_state(roach)
    raw_input("Hit enter to begin sweep with compressor on.")
    start1 = time.time()
    sweep_data = sweeps.do_prepared_sweep(roach, nchan_per_step=nominal_frequencies.size, reads_per_step=8)
    print("Sweep completed in {:.0f} seconds.".format(time.time() - start1))
    df.add_sweep(sweep_data)
    df.sync()
    raw_input("Hit enter to begin sweep with compressor off.")
    start2 = time.time()
    sweep_data = sweeps.do_prepared_sweep(roach, nchan_per_step=nominal_frequencies.size, reads_per_step=8)
    print("Sweep completed in {:.0f} seconds.".format(time.time() - start2))
    df.add_sweep(sweep_data)
    df.sync()
    df.nc.close()


def record_timestream(roach, nominal_frequencies, attenuation, n_samples, suffix, time_in_seconds):
    df = data_file.DataFile(suffix=suffix)
    roach.set_dac_attenuator(attenuation)
    measured_frequencies = roach.set_tone_frequencies(nominal_frequencies, nsamp=n_samples)
    roach.select_fft_bins(np.arange(roach.tone_bins.shape[1]))
    df.log_hw_state(roach)
    raw_input("Hit enter to begin recording timestreams.")
    start = time.time()
    data, address = roach.get_data_seconds(time_in_seconds, demod=True, pow2=True)
    print("Elapsed time {:.0f} seconds.".format(time.time() - start))
    df.add_timestream_data(data, roach, start)
    df.sync()
    df.nc.close()


def sweep_fit_timestream(roach, center_frequencies, offset_frequencies, sweep_n_samples, timestream_n_samples,
                         attenuation, suffix,
                         time_in_seconds, interactive=False, coarse_multiplier=3):
    df = data_file.DataFile(suffix=suffix)
    print("Writing data to " + df.filename)
    print("Setting DAC attenuator to {:.1f} dB".format(attenuation))
    roach.set_dac_attenuator(attenuation)
    print("Sweep memory usage is {:.1f} MB of {:.1f} MB capacity.".format(
        memory_usage_bytes(offset_frequencies.shape[0], sweep_n_samples) / 2 ** 20, EFFECTIVE_DRAM_CAPACITY / 2 ** 20))

    # Do a preliminary sweep to make sure the main sweeps are centered properly.
    sweeps.prepare_sweep(roach, center_frequencies, coarse_multiplier * offset_frequencies, sweep_n_samples)
    roach._sync()
    time.sleep(0.2)
    if interactive:
        raw_input("Hit enter to record preliminary frequency sweep.")
    else:
        print("Recording preliminary frequency sweep.")
    coarse_sweep_data = sweeps.do_prepared_sweep(roach, nchan_per_step=center_frequencies.size, reads_per_step=8)
    coarse_resonators = fit_sweep_data(coarse_sweep_data)
    coarse_f0s = np.array([r.f_0 for r in coarse_resonators])
    fine_center_frequencies = round_frequencies(coarse_f0s, roach.fs, sweep_n_samples)

    # Now do the actual sweep and save it.
    sweeps.prepare_sweep(roach, fine_center_frequencies, offset_frequencies, sweep_n_samples)
    roach._sync()
    time.sleep(0.2)
    df.log_hw_state(roach)
    if interactive:
        raw_input("Hit enter to begin recording frequency sweep.")
    else:
        print("Recording frequency sweep.")
    sweep_start_time = time.time()
    sweep_data = sweeps.do_prepared_sweep(roach, nchan_per_step=center_frequencies.size, reads_per_step=8)
    print("Elapsed time {:.0f} seconds. Writing to disk.".format(time.time() - sweep_start_time))
    df.add_sweep(sweep_data)
    df.sync()
    resonators = fit_sweep_data(sweep_data)
    fine_f0s = np.array([r.f_0 for r in resonators])
    print("Initial frequencies in MHz are " + ', '.join(['{:.3f}'.format(f0) for f0 in center_frequencies]))
    print("initial - fit [Hz]: " +
          ', '.join(['{:.0f}'.format(1e6 * delta_f) for delta_f in center_frequencies - fine_f0s]))

    timestream_measured_frequencies = roach.set_tone_frequencies(fine_f0s, nsamp=timestream_n_samples)
    roach.select_fft_bins(np.arange(roach.tone_bins.shape[1]))
    roach._sync()
    time.sleep(0.2)
    df.log_hw_state(roach)
    print("measured - fit [Hz]: " +
          ', '.join(['{:.0f}'.format(1e6 * delta_f) for delta_f in timestream_measured_frequencies - fine_f0s]))
    # This delay was added because the above lines cause some kind of transient signal that takes about three seconds
    # to decay. This was showing up at the beginning of the timestreams with interactive=False.
    minimum_wait = 5
    wait_start_time = time.time()
    if interactive:
        raw_input("Hit enter to begin recording {:.0f} second timestream.".format(time_in_seconds))
    else:
        print("Recording {:.0f} second timestream.".format(time_in_seconds))
    while time.time() - wait_start_time < minimum_wait:
        time.sleep(0.1)
    timestream_start_time = time.time()
    data, address = roach.get_data_seconds(time_in_seconds, demod=True, pow2=True)
    print("Elapsed time {:.0f} seconds. Writing to disk.".format(time.time() - timestream_start_time))
    df.add_timestream_data(data, roach, timestream_start_time)
    df.sync()
    df.nc.close()
    return df.filename


def coarse_fit_fine(roach, center_frequencies, coarse_offset_integers, coarse_n_samples, fine_offset_integers,
                    fine_n_samples):
    coarse_frequency_resolution = roach.fs / coarse_n_samples
    coarse_sweep_data = sweep(roach, center_frequencies, coarse_frequency_resolution * coarse_offset_integers,
                              coarse_n_samples)
    coarse_resonators = fit_sweep_data(coarse_sweep_data)
    fine_frequency_resolution = roach.fs / fine_n_samples
    fit_center_frequencies = np.array([r.f_0 for r in coarse_resonators])
    fine_center_frequencies = round_frequencies(fit_center_frequencies, roach.fs, fine_n_samples)
    fine_sweep_data = sweep(roach, fine_center_frequencies, fine_frequency_resolution * fine_offset_integers,
                            fine_n_samples)
    return coarse_sweep_data, fine_sweep_data
