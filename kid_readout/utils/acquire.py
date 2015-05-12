from __future__ import division
import time

import numpy as np

from kid_readout.utils import data_file, sweeps
from kid_readout.analysis.resonator import Resonator
from kid_readout.analysis.khalil import bifurcation_s21, bifurcation_guess


BYTES_PER_SAMPLE = 4
EFFECTIVE_DRAM_CAPACITY = 2 ** 28  # 256 MB

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
                                 0, 1, 3, 6, 10, 15, 21, 28, 36])}


def round_frequencies(frequencies, sample_frequency, n_samples):
    frequency_resolution = sample_frequency / n_samples
    return frequency_resolution * np.round(frequencies / frequency_resolution)


def memory_usage_bytes(n_waveforms, n_samples):
    return n_waveforms * n_samples * BYTES_PER_SAMPLE


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


def mmw_source_sweep_and_stream(df, roach, lockin, approximate_stream_time, overwrite_last, f_mmw_source,
                     sweep_modulation_rate, stream_modulation_rate, measurement_modulation_rate,
                     transient_wait=10, lockin_wait=5, verbose=True):
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
    # The next two lines were in the wrong order. This may have led to all the code using this overwrite_last technique
    # to have read out from the wrong FFT bin. Need to check.
    # After 2015-05-05, select_fft_bins is no longer necessary since select_bank now selects FFT bins too.
    #roach.select_fft_bins(np.arange(roach.tone_bins.shape[1]))
    roach.select_bank(roach.fft_bins.shape[0] - 1)
    roach._sync()
    time.sleep(transient_wait)  # The above commands somehow create a transient that takes about 5 seconds to decay.
    df.log_hw_state(roach)
    start_time = time.time()
    stream, addresses = roach.get_data_seconds(approximate_stream_time, pow2=True)
    df.add_timestream_data(stream, roach, start_time, mmw_source_freq=f_mmw_source,
                           mmw_source_modulation_freq=f_stream_modulation, zbd_voltage=x)
    df.sync()
    return sweep, stream


def sweep(roach, center_frequencies, sample_exponent, offset_frequencies=None, reads_per_step=2, transient_wait=0,
          run=lambda: None):
    n_samples = 2**sample_exponent
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


# TODO: fix broken references
def record_sweeps_on_off(roach, nominal_frequencies, attenuation, n_samples, suffix):
    df = data_file.DataFile(suffix=suffix)
    roach.set_dac_attenuator(attenuation)
    measured_frequencies = roach.set_tone_frequencies(nominal_frequencies, nsamp=n_samples)
    df.log_hw_state(roach)
    raw_input("Hit enter to begin sweep with compressor on.")
    start1 = time.time()
    sweep_data = sweeps.do_prepared_sweep(roach, nchan_per_step=f0s.size, reads_per_step=8)
    print("Sweep completed in {:.0f} seconds.".format(time.time() - start1))
    df.add_sweep(sweep_data)
    df.sync()
    raw_input("Hit enter to begin sweep with compressor off.")
    start2 = time.time()
    sweep_data = sweeps.do_prepared_sweep(roach, nchan_per_step=f0s.size, reads_per_step=8)
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
