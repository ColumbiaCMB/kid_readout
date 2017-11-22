"""
Measure resonators, one at a time, with the readout tone centered in the filterbank bin.
"""
from __future__ import division
import time

import numpy as np

from kid_readout.roach import analog, calculate, hardware_tools
from kid_readout.measurement import acquire, basic
from kid_readout.equipment import hardware, agilent_33220

acquire.show_settings()
acquire.show_git_status()
logger = acquire.get_script_logger(__file__)

# Parameters
suffix = 'hittite'
attenuations = [20]
fft_gains = [5]
f_center_all = 1e6 * np.array([2522.24, 2605.96, 2723.65, 2787.96, 3851.13])
f_center = f_center_all[1:2] # select a subset of the frequencies
fractional_frequency_shift = 0
f_center *= (1 + fractional_frequency_shift)
df_baseband_target = 5e3
fine_sweep_num_linewidths = 5
Q_max_expected = 50e3
df_coarse_sweep = f_center.min() / Q_max_expected  # A coarse sweep with a resolution of one linewidth should work
df_total = 4e6  # The total span of the baseband tones
df_lo = 2.5e3  # This is the smallest resolution available
f_baseband_minimum = 10e6  # Keep the tones away from the LO by at least this frequency.
sweep_length_seconds = 0  # Take the minimum amount of data, in this case one block
stream_length_seconds = 10

# Hardware
conditioner = analog.HeterodyneMarkII()
shield = hardware.Thing(name='magnetic_shield_pocket', state={'orientation': 'horizontal'})
hittite = hardware.Thing(name='hittite', state={'output_dBm': 0})
hw = hardware.Hardware(conditioner, shield, hittite)
ri = hardware_tools.r1h11_with_mk2(initialize=True, use_config=False)
ri.adc_valon.set_ref_select(0)  # internal
ri.lo_valon.set_ref_select(1)  # external

# Calculate sweep parameters, LO and baseband sweep frequencies
ri_state = ri.state
tone_sample_exponent = int(np.round(np.log2(ri_state.adc_sample_rate / df_baseband_target)))
df_baseband = ri_state.adc_sample_rate / 2 ** tone_sample_exponent
logger.info("Baseband resolution is {:.0f} Hz using 2^{:d} samples".format(df_baseband, tone_sample_exponent))
num_sweep_tones = min(int(df_total / df_baseband), ri.max_num_waveforms(2 ** tone_sample_exponent))
logger.info("Using {:d} tones".format(num_sweep_tones))
f_baseband = f_baseband_minimum + ri.state.adc_sample_rate / 2**tone_sample_exponent * np.arange(num_sweep_tones)
logger.info("Coarse sweep span is {:.1f} MHz".format(1e-6 * f_baseband.ptp()))
coarse_stride = max(df_coarse_sweep // df_baseband, 1)
logger.info("Coarse sweep resolution is {:.0f} Hz".format(coarse_stride * df_baseband))
f_lo_center = df_lo * np.round((f_center - f_baseband.mean()) / df_lo)

# Run
npd = acquire.new_npy_directory(suffix=suffix)
tic = time.time()
try:
    ri.set_tone_baseband_freqs(freqs=1e-6 * f_baseband[:, np.newaxis], nsamp=2 ** tone_sample_exponent)
    for lo_index, f_lo in enumerate(f_lo_center):
        ri.set_lo(lomhz=1e-6 * f_lo, chan_spacing=1e-6 * df_lo)
        for attenuation_index, (attenuation, fft_gain) in enumerate(zip(attenuations, fft_gains)):
            ri.set_dac_attenuator(attenuation)
            ri.set_fft_gain(fft_gain)
            state = hw.state()
            state['lo_index'] = lo_index
            coarse_sweep = acquire.run_loaded_sweep(ri, length_seconds=sweep_length_seconds,
                                                    tone_bank_indices=np.arange(0, num_sweep_tones, coarse_stride))[0]
            npd.write(coarse_sweep)
            coarse_f_r = coarse_sweep.resonator.f_0
            coarse_Q = coarse_sweep.resonator.Q
            logger.info("Coarse sweep f_r = {:.3f} MHz +/- {:.0f} Hz".format(1e-6 * coarse_f_r,
                                                                             coarse_sweep.resonator.f_0_error))
            logger.info("Coarse sweep Q = {:.0f} +/- {:.0f}".format(coarse_Q, coarse_sweep.resonator.Q_error))
            df_filterbank = calculate.stream_sample_rate(ri_state)
            f_baseband_bin_center = df_filterbank * np.round(f_baseband.mean() / df_filterbank)
            f_lo_fine = df_lo * np.round((coarse_f_r - f_baseband_bin_center) / df_lo)
            ri.set_lo(lomhz=1e-6 * f_lo_fine, chan_spacing=1e-6 * df_lo)
            fine_indices = np.where(np.abs(f_lo_fine + f_baseband - coarse_f_r) <=
                                    (fine_sweep_num_linewidths / 2) * (coarse_f_r / coarse_Q))[0]
            fine_sweep = acquire.run_loaded_sweep(ri, length_seconds=sweep_length_seconds,
                                                  tone_bank_indices=fine_indices)[0]
            ri.select_bank(np.argmin(np.abs(f_baseband_bin_center - f_baseband)))
            ri.select_fft_bins(np.array([0]))
            print("Frequency in Hz is {:.1f}".format(1e6 * ri.tone_frequencies[ri.bank][0]))
            power = float(raw_input("Attach the Hittite output to the cryostat input and enter the power in dBm: "))
            state['hittite']['output_dBm'] = power
            logger.info("Recording {:.1f} s stream".format(stream_length_seconds))
            stream = ri.get_measurement(num_seconds=stream_length_seconds, demod=True)[0]
            sweep_stream = basic.SingleSweepStream(sweep=fine_sweep, stream=stream, state=state)
            npd.write(sweep_stream)
            npd.write(ri.get_adc_measurement())
            raw_input("Reconnect the roach output to the cryostat input.")
            state['hittite']['output_dBm'] = 0
finally:
    npd.close()
    print("Wrote {}".format(npd.root_path))
    print("Elapsed time {:.0f} minutes.".format((time.time() - tic) / 60))
