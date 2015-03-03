import numpy as np
import time
from kid_readout.utils import roach_interface, data_file, sweeps, acquire
from kid_readout.equipment import lockin_controller


def source_on():
    return roach.set_modulation_output(rate='low')


def source_off():
    return roach.set_modulation_output(rate='high')


def source_modulate(rate=7):
    return roach.set_modulation_output(rate=rate)


def sweep_and_stream(df, roach, lockin, n_channels, approximate_stream_length, banks, overwrite_last,
                     mmw_source_frequency, mmw_source_modulation_freq, transient_wait=10):
    df.log_hw_state(roach)
    fine_sweep = sweeps.do_prepared_sweep(roach, nchan_per_step=n_channels, reads_per_step=2, banks=banks)
    df.add_sweep(fine_sweep)
    df.sync()
    fit_f0s = np.array([r.f_0 for r in acquire.fit_sweep_data(fine_sweep)])

    timestream_frequencies = roach.add_tone_freqs(fit_f0s, overwrite_last=overwrite_last)
    roach.select_fft_bins(np.arange(n_channels))
    roach.select_bank(roach.fft_bins.shape[0]-1)
    roach._sync()
    time.sleep(transient_wait)  # The above commands create a transient that takes about 5 seconds to decay.
    df.log_hw_state(roach)
    start_time = time.time()
    stream, addresses = roach.get_data_seconds(approximate_stream_length, pow2=True)
    x, y, r, theta = lockin.get_data()  # I don't think this will actually work, but maybe it doesn't matter.
    df.add_timestream_data(stream, roach, start_time, mmw_source_freq=mmw_source_frequency,
                           mmw_source_modulation_freq=mmw_source_modulation_freq, zbd_voltage=x)
    df.sync()
    return fit_f0s


# run from here:
lockin = lockin_controller.lockinController()
print lockin.get_idn()

roach = roach_interface.RoachBaseband()

suffix = "temperature_step_with_mmw_noise"
mmw_source_frequency = -1.0

f0s = np.load('/home/data2/resonances/2014-12-06_140825_0813f8_fit_16.npy')

coarse_exponent = 19
coarse_n_samples = 2**coarse_exponent
coarse_frequency_resolution = roach.fs / coarse_n_samples  # about 1 kHz
coarse_offset_integers = acquire.offset_integers[coarse_exponent] - 32
coarse_offset_frequencies = coarse_frequency_resolution * coarse_offset_integers

fine_exponent = 21
fine_n_samples = 2**fine_exponent
fine_frequency_resolution = roach.fs / fine_n_samples  # about 0.25 kHz
# Drop the 31st point so as to not force the added points to occupy the 32nd memory slot, which may not work.
fine_offset_integers = acquire.offset_integers[fine_exponent][:-1]
fine_offset_frequencies = fine_frequency_resolution * fine_offset_integers

attenuations = [41, 38, 35, 32, 29, 26, 23]
approximate_stream_length = 30  # in seconds

while True:
    mmw_source_modulation_freq = source_modulate()
    print("\nSet source modulation frequency to {:.1f} Hz. Check the lock-in.".format(mmw_source_modulation_freq))
    print("Enter mmw attenuator values as a tuple i.e.: 6.5,6.5 or type exit to stop collecting data.")
    mmw_atten_str = raw_input("mmw attenuator values: ")
    if mmw_atten_str == 'exit':
        break
    else:
        mmw_atten_turns = eval(mmw_atten_str)


    start_time = time.time()
    df = data_file.DataFile(suffix=suffix)
    df.nc.mmw_atten_turns = mmw_atten_turns

    # Take a coarse sweep with the source off at the lowest attenuation to get the approximate resonance frequencies.
    source_off()
    maximum_attenuation = max(attenuations)
    print("Setting DAC attenuator to {:.1f} dB for coarse sweep.".format(maximum_attenuation))
    roach.set_dac_attenuator(maximum_attenuation)
    coarse_measured_frequencies = sweeps.prepare_sweep(roach, f0s, coarse_offset_frequencies, coarse_n_samples)
    df.log_hw_state(roach)
    coarse_sweep = sweeps.do_prepared_sweep(roach, nchan_per_step=f0s.size, reads_per_step=2)
    df.add_sweep(coarse_sweep)
    coarse_resonators = acquire.fit_sweep_data(coarse_sweep)
    coarse_fit_f0s = np.array([r.f_0 for r in coarse_resonators])
    print("coarse - initial [Hz]: " + ', '.join(['{:.0f}'.format(1e6 * diff) for diff in coarse_fit_f0s - f0s]))

    # Record one stream at the lowest attenuation.
    mmw_source_modulation_freq = source_modulate()
    print("Set source modulation frequency to {:.1f} Hz.".format(mmw_source_modulation_freq))
    modulated_measured_frequencies = roach.set_tone_freqs(coarse_fit_f0s, nsamp=fine_n_samples)
    roach.select_fft_bins(np.arange(f0s.size))
    roach._sync()
    time.sleep(1)
    df.log_hw_state(roach)
    modulated_start_time = time.time()
    modulated_stream, addresses = roach.get_data_seconds(4)
    x, y, r, theta = lockin.get_data()
    df.add_timestream_data(modulated_stream, roach, modulated_start_time, mmw_source_freq=mmw_source_frequency,
                           mmw_source_modulation_freq=mmw_source_modulation_freq, zbd_voltage=x)
    df.sync()

    # Use these frequencies for all subsequent sweeps, and add an additional waveform for each stream.
    print("Setting sweep frequencies.")
    fine_frequencies = sweeps.prepare_sweep(roach, coarse_fit_f0s, fine_offset_frequencies, fine_n_samples)

    for k, attenuation in enumerate(attenuations):
        print("\nSetting DAC attenuator to {:.1f} dB.".format(attenuation))
        roach.set_dac_attenuator(attenuation)
        roach._sync()
        time.sleep(1)

        source_off_modulation_freq = source_off()
        print("Source off.")
        time.sleep(1)
        off_fit_f0s = sweep_and_stream(df, roach, lockin, f0s.size, approximate_stream_length,
                                       range(fine_offset_frequencies.size), True,
                                       mmw_source_frequency, source_off_modulation_freq)
        print("off - initial [Hz]: " + ', '.join(['{:.0f}'.format(1e6 * diff) for diff in off_fit_f0s - f0s]))

    df.close()
    print("Completed in {:.0f} minutes.".format((time.time() - start_time) / 60))
