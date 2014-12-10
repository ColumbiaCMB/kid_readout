from __future__ import division
import numpy as np
import time

from kid_readout.utils import roach_interface, data_file, sweeps
from kid_readout.analysis.resonator import Resonator, fit_resonator
from kid_readout.analysis.khalil import delayed_generic_s21, delayed_auto_guess
from kid_readout.analysis.khalil import bifurcation_s21, bifurcation_guess

# Both of these numbers may be wrong: check!
BYTES_PER_SAMPLE = 4
EFFECTIVE_DRAM_CAPACITY = 2 ** 28  # 256 MB

# This array almost fills the Roach memory when using 2^21 samples
offset_integers_31 = np.array([-40, -30, -25, -20, -15, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
                               0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40])


def round_frequencies(freqs, sample_freq, n_samples):
    freq_resolution = sample_freq / n_samples
    return freq_resolution * np.round(freqs / freq_resolution)


def memory_usage_bytes(n_waveforms, n_samples):
    return n_waveforms * n_samples * BYTES_PER_SAMPLE


def combine_center_and_offset_frequencies(center_freqs, offset_freqs):
    """
    Combine center and offset frequencies into a single array that matches the Roach format.

    :param center_freqs: an array containing center frequencies of the sweep.
    :param offset_freqs: an array containing offset frequencies of the sweep.
    :return: an array with shape (offset_freqs.size, center_freqs.size) containing the combined frequencies.
    """
    return center_freqs[np.newaxis, :] + offset_freqs[:, np.newaxis]


def fit_sweep_data(sweep_data, model=bifurcation_s21, guess=bifurcation_guess):
    resonators = []
    for i in np.unique(sweep_data.sweep_indexes):
        f, s21, errors = sweep_data.select_index(i)
        resonators.append(Resonator(f, s21, errors=errors, model=model, guess=guess))
    return sorted(resonators, key=lambda r: r.f_0)


def sweep(roach, center_freqs, offset_freqs, n_samples, reads_per_step=2):
    sweeps.prepare_sweep(roach, center_freqs, offset_freqs, n_samples)
    sweep_data = sweeps.do_prepared_sweep(roach, nchan_per_step=len(center_freqs), reads_per_step=reads_per_step)
    return sweep_data


def timestream(roach, freqs, n_samples, time_in_seconds, pow2=True):
    measured_freqs = roach.set_tone_freqs(freqs, nsamp=n_samples)
    roach.select_fft_bins(np.arange(roach.tone_bins.shape[1]))
    data, address = roach.get_data_seconds(time_in_seconds, demod=True, pow2=pow2)
    return data, address


def record_sweep(roach, center_freqs, offset_freqs, attenuation, n_samples, suffix, interactive=False):
    n_channels = center_freqs.size
    df = data_file.DataFile(suffix=suffix)
    print("Writing data to " + df.filename)
    print("Setting DAC attenuator to {:.1f} dB".format(attenuation))
    roach.set_dac_attenuator(attenuation)
    print("Sweep memory usage is {:.1f} MB of {:.1f} MB capacity.".format(
          memory_usage_bytes(offset_freqs.shape[0], n_samples) / 2 ** 20, EFFECTIVE_DRAM_CAPACITY / 2 ** 20))

    measured_freqs = sweeps.prepare_sweep(roach, center_freqs, offset_freqs, n_samples)
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
    print("Initial frequencies in MHz are " + ', '.join(['{:.3f}'.format(f0) for f0 in center_freqs]))
    print("initial - fit [Hz]: " +
          ', '.join(['{:.0f}'.format(1e6 * delta_f) for delta_f in center_freqs - fit_f0s]))
    df.nc.close()
    return df.filename, resonators


# TODO: fix broken references
def record_sweeps_on_off(roach, nominal_freqs, attenuation, n_samples, suffix):
    df = data_file.DataFile(suffix=suffix)
    roach.set_dac_attenuator(attenuation)
    measured_freqs = roach.set_tone_freqs(nominal_freqs, nsamp=n_samples)
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


def record_timestream(roach, nominal_freqs, attenuation, n_samples, suffix, time_in_seconds):
    df = data_file.DataFile(suffix=suffix)
    roach.set_dac_attenuator(attenuation)
    measured_freqs = roach.set_tone_freqs(nominal_freqs, nsamp=n_samples)
    roach.select_fft_bins(np.arange(roach.tone_bins.shape[1]))
    df.log_hw_state(roach)
    raw_input("Hit enter to begin recording timestreams.")
    start = time.time()
    data, address = roach.get_data_seconds(time_in_seconds, demod=True, pow2=True)
    print("Elapsed time {:.0f} seconds.".format(time.time() - start))
    df.add_timestream_data(data, roach, start)
    df.sync()
    df.nc.close()


def sweep_fit_timestream(roach, center_freqs, offset_freqs, sweep_n_samples, timestream_n_samples, attenuation, suffix,
                         time_in_seconds, interactive=False, coarse_multiplier=3):
    df = data_file.DataFile(suffix=suffix)
    print("Writing data to " + df.filename)
    print("Setting DAC attenuator to {:.1f} dB".format(attenuation))
    roach.set_dac_attenuator(attenuation)
    print("Sweep memory usage is {:.1f} MB of {:.1f} MB capacity.".format(
          memory_usage_bytes(offset_freqs.shape[0], sweep_n_samples) / 2 ** 20, EFFECTIVE_DRAM_CAPACITY / 2 ** 20))

    # Do a preliminary sweep to make sure the main sweeps are centered properly.
    sweeps.prepare_sweep(roach, center_freqs, coarse_multiplier * offset_freqs, sweep_n_samples)
    roach._sync()
    time.sleep(0.2)
    if interactive:
        raw_input("Hit enter to record preliminary frequency sweep.")
    else:
        print("Recording preliminary frequency sweep.")
    coarse_sweep_data = sweeps.do_prepared_sweep(roach, nchan_per_step=center_freqs.size, reads_per_step=8)
    coarse_resonators = fit_sweep_data(coarse_sweep_data)
    coarse_f0s = np.array([r.f_0 for r in coarse_resonators])
    fine_center_freqs = round_frequencies(coarse_f0s, roach.fs, sweep_n_samples)

    # Now do the actual sweep and save it.
    sweeps.prepare_sweep(roach, fine_center_freqs, offset_freqs, sweep_n_samples)
    roach._sync()
    time.sleep(0.2)
    df.log_hw_state(roach)
    if interactive:
        raw_input("Hit enter to begin recording frequency sweep.")
    else:
        print("Recording frequency sweep.")
    sweep_start_time = time.time()
    sweep_data = sweeps.do_prepared_sweep(roach, nchan_per_step=center_freqs.size, reads_per_step=8)
    print("Elapsed time {:.0f} seconds. Writing to disk.".format(time.time() - sweep_start_time))
    df.add_sweep(sweep_data)
    df.sync()
    resonators = fit_sweep_data(sweep_data)
    fine_f0s = np.array([r.f_0 for r in resonators])
    print("Initial frequencies in MHz are " + ', '.join(['{:.3f}'.format(f0) for f0 in center_freqs]))
    print("initial - fit [Hz]: " +
          ', '.join(['{:.0f}'.format(1e6 * delta_f) for delta_f in center_freqs - fine_f0s]))

    timestream_measured_freqs = roach.set_tone_freqs(fine_f0s, nsamp=timestream_n_samples)
    roach.select_fft_bins(np.arange(roach.tone_bins.shape[1]))
    roach._sync()
    time.sleep(0.2)
    df.log_hw_state(roach)
    print("measured - fit [Hz]: " +
          ', '.join(['{:.0f}'.format(1e6 * delta_f) for delta_f in timestream_measured_freqs - fine_f0s]))
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


def coarse_fit_fine(roach, center_freqs, offset_integers, n_samples, coarse_multiplier=3):
    freq_resolution = roach.fs / n_samples
    coarse_sweep_data = sweep(roach, center_freqs, coarse_multiplier * freq_resolution * offset_integers, n_samples)
    coarse_resonators = fit_sweep_data(coarse_sweep_data)
    fit_center_freqs = np.array([r.f_0 for r in coarse_resonators])
    fine_center_freqs = round_frequencies(fit_center_freqs, roach.fs, n_samples)
    fine_sweep_data = sweep(roach, fine_center_freqs, freq_resolution * offset_integers, n_samples)
    return coarse_sweep_data, fine_sweep_data
