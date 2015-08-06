import numpy as np
import time
from kid_readout.utils import roach_interface, data_file, sweeps, acquire
from kid_readout.equipment import hittite_controller

# Configuration
attenuations = [41]
probe_channel = 1
dummy_frequency = 90
probe_detuning = 2
hittite_power = -18
#hittite_powers = [0]
#probe_channels = [0]
#dummy_frequencies = [90] * len(probe_channels)  # MHz
#probe_detunings = [2] * len(probe_channels)  # MHz
suffix = 'electrical_crosstalk_channel_{}_power_{:.1f}_dBm'.format(probe_channel, hittite_power)
approximate_stream_length = 30  # in seconds
coarse_exponent = 19
fine_exponent = 21

# Start
start_time = time.time()
tuned = data_file.DataFile(suffix=suffix + '_tuned')
detuned = data_file.DataFile(suffix=suffix + '_detuned')
off = data_file.DataFile(suffix=suffix + '_off')
hittite = hittite_controller.hittiteController()
hittite.off()
roach = roach_interface.RoachBaseband()

# Drop the 31st point so as to not force the added points to occupy the 32nd memory slot, which may not work.
coarse_n_samples = 2**coarse_exponent
coarse_frequency_resolution = roach.fs / coarse_n_samples  # about 1 kHz
coarse_offset_integers = acquire.offset_integers[coarse_exponent][:-1] - 32
coarse_offset_frequencies = coarse_frequency_resolution * coarse_offset_integers
fine_n_samples = 2**fine_exponent
fine_frequency_resolution = roach.fs / fine_n_samples  # about 0.25 kHz
fine_offset_integers = acquire.offset_integers[fine_exponent][:-1]
fine_offset_frequencies = fine_frequency_resolution * fine_offset_integers
f0s = np.load('/home/data2/resonances/2014-12-06_140825_0813f8_fit_16.npy')
n_channels = f0s.size

def sweep_and_stream(df, roach, n_channels, approximate_stream_length, banks, overwrite_last, transient_wait=10):
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
    df.add_timestream_data(stream, roach, start_time)
    df.sync()
    return fit_f0s


# Take a coarse sweep with the source off at the lowest attenuation to get the approximate resonance frequencies.
maximum_attenuation = max(attenuations)
print("Setting DAC attenuator to {:.1f} dB for coarse sweep.".format(maximum_attenuation))
roach.set_dac_attenuator(maximum_attenuation)
coarse_measured_frequencies = sweeps.prepare_sweep(roach, f0s, coarse_offset_frequencies, coarse_n_samples)
coarse_sweep = sweeps.do_prepared_sweep(roach, nchan_per_step=n_channels, reads_per_step=2)
coarse_resonators = acquire.fit_sweep_data(coarse_sweep)
coarse_f0s = np.array([r.f_0 for r in coarse_resonators])
print("coarse - initial [Hz]: " + ', '.join(['{:.0f}'.format(1e6 * diff) for diff in coarse_f0s - f0s]))

#for probe_channel, dummy_frequency, probe_detuning in zip(probe_channels, dummy_frequencies, probe_detunings):
print("Probing channel {}. Setting sweep frequencies.".format(probe_channel))
probe_frequency = coarse_f0s[probe_channel]
frequencies = coarse_f0s.copy()
frequencies[probe_channel] = dummy_frequency
order = frequencies.argsort()
fine_frequencies = sweeps.prepare_sweep(roach, frequencies[order], fine_offset_frequencies, fine_n_samples)
#for k, hittite_power in enumerate(hittite_powers):
#    for attenuation in attenuations:
for k, attenuation in enumerate(attenuations):
    print("\nSetting DAC attenuator to {:.1f} dB.".format(attenuation))
    roach.set_dac_attenuator(attenuation)

    print("Setting Hittite to {:.1f} dBm.".format(hittite_power))
    hittite.set_power(hittite_power)

    print("Setting Hittite on resonance at {:.3f} MHz".format(probe_frequency))
    hittite.set_freq(1e6 * probe_frequency)
    hittite.on()
    time.sleep(1)
    tuned_f0s = sweep_and_stream(tuned, roach, n_channels, approximate_stream_length,
                              range(fine_offset_frequencies.size), k > 0)  # overwrite after the first sweep.
    print("tuned - coarse [Hz]: " + ', '.join(['{:.0f}'.format(1e6 * diff)
                                               for diff in tuned_f0s - coarse_f0s[order]]))

    print("Detuning Hittite to {:.3f} + {:.3f} MHz".format(probe_frequency, probe_detuning))
    hittite.set_freq(1e6 * (probe_frequency + probe_detuning))
    time.sleep(1)
    detuned_f0s = sweep_and_stream(detuned, roach, n_channels, approximate_stream_length,
                              range(fine_offset_frequencies.size), True)
    print("detuned - coarse [Hz]: " + ', '.join(['{:.0f}'.format(1e6 * diff)
                                                 for diff in detuned_f0s - coarse_f0s[order]]))

    print("Hittite off.")
    hittite.off()
    time.sleep(1)
    off_f0s = sweep_and_stream(off, roach, n_channels, approximate_stream_length,
                              range(fine_offset_frequencies.size), True)
    print("off - coarse [Hz]: " + ', '.join(['{:.0f}'.format(1e6 * diff) for diff in off_f0s - coarse_f0s[order]]))

detuned.close()
tuned.close()
off.close()
print("Completed in {:.0f} minutes.".format((time.time() - start_time) / 60))
