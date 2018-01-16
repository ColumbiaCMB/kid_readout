"""
Measure resonators, one at a time, with the readout tone centered in the filterbank bin.
"""
from __future__ import division
import time


import numpy as np

from kid_readout.roach import analog, calculate, hardware_tools, tools
from kid_readout.measurement import acquire, basic
from kid_readout.equipment import hardware, starcryo_temps
from equipment.srs import lockin
from equipment.custom import mmwave_source
from kid_readout.settings import LOCKIN_SERIAL_PORT
import resonances

acquire.show_settings()
acquire.show_git_status()
logger = acquire.get_script_logger(__file__)

# Parameters
suffix = 'mmw'
df_baseband_target = 30e3
f_sweep_span = 3e6  # The total span of the baseband tones
f_lo_spacing = 2.5e3  # This is the smallest resolution available
f_baseband_minimum = 100e6  # Keep the tones away from the LO by at least this frequency.
sweep_length_seconds = 0.05
stream_length_seconds = 30

# Resonance frequencies
band_dict = resonances.dict_180_mK
fractional_frequency_shift = 0
band_name = '2758'  # '3317'
all_f_initial = (1 + fractional_frequency_shift) * band_dict[band_name][1:3]
attenuations_list = [all_f_initial.size * (25, 35, 45)]

# Hardware
temperature = starcryo_temps.Temperature()
lock = lockin.SR830(serial_device=LOCKIN_SERIAL_PORT)
lock.identification  # This seems to be necessary to wake up the lockin
mmw = mmwave_source.MMWaveSource()
mmw.set_attenuator_ticks(0, 0)
mmw.multiplier_input = 'thermal'
mmw.ttl_modulation_source = "roach_2"
mmw.waveguide_twist_angle = 0
conditioner = analog.HeterodyneMarkII()
hw = hardware.Hardware(temperature, lock, mmw, conditioner)
ri = hardware_tools.r2h11_with_mk2(initialize=True, use_config=False)
ri.set_modulation_output('high')
ri.iq_delay = -1
ri.adc_valon.set_ref_select(1)  # external
assert np.all(ri.adc_valon.get_phase_locks())
ri.lo_valon.set_ref_select(1)  # external
assert np.all(ri.lo_valon.get_phase_locks())

# Calculate sweep parameters, LO and baseband sweep frequencies
ri_state = ri.state
tone_sample_exponent = int(np.round(np.log2(ri_state.adc_sample_rate / df_baseband_target)))
df_baseband = ri_state.adc_sample_rate / 2 ** tone_sample_exponent
df_filterbank = ri_state.adc_sample_rate / ri_state.num_filterbank_channels
num_sweep_tones = int(f_sweep_span / df_baseband)
logger.info("Sweeps will use {:d} tones spanning {:.1f} MHz with resolution {:.0f} Hz (2^{:d} samples)".format(
    num_sweep_tones, 1e-6 * f_sweep_span, df_baseband, tone_sample_exponent))
n_baseband = (f_baseband_minimum + f_sweep_span / 2) // df_baseband + np.arange(num_sweep_tones)
f_baseband = df_baseband * n_baseband

# Run
npd = acquire.new_npy_directory(suffix=suffix)
tic = time.time()
try:
    for f_index, (f_initial, attenuations) in enumerate(zip(all_f_initial, attenuations_list)):
        logger.info("Measuring resonator {:d} of {:d}".format(f_index + 1, all_f_initial.size))
        f_lo_initial = f_initial - f_baseband.mean()
        assert np.all(ri.adc_valon.get_phase_locks())
        assert np.all(ri.lo_valon.get_phase_locks())
        #tools.set_and_attempt_external_phase_lock(ri=ri, f_lo=1e-6 * f_lo_initial, f_lo_spacing=1e-6 * f_lo_spacing)
        ri.set_lo(lomhz=1e-6 * f_lo_initial, chan_spacing=1e-6 * f_lo_spacing)
        # Take the initial sweep using the minimum power
        ri.set_dac_attenuator(max(attenuations))
        ri.set_tone_baseband_freqs(freqs=1e-6 * np.array([f_baseband[0]]), nsamp=2 ** tone_sample_exponent)
        time.sleep(1)
        tools.optimize_fft_gain(ri, fraction_of_maximum=0.5)
        time.sleep(1)
        initial_state = hw.state()
        initial_state['f_index'] = f_index
        initial_sweep = acquire.run_sweep(ri=ri, tone_banks=1e-6 * (f_lo_initial + f_baseband[:, np.newaxis]),
                                          num_tone_samples=2 ** tone_sample_exponent,
                                          length_seconds=sweep_length_seconds, state=initial_state, verbose=True)[0]
        npd.write(initial_sweep)
        f_fit = initial_sweep.resonator.f_0
        logger.info("Initial sweep f_r = {:.3f} MHz +/- {:.0f} Hz".format(1e-6 * f_fit,
                                                                          initial_sweep.resonator.f_0_error))
        logger.info("Initial sweep Q = {:.0f} +/- {:.0f}".format(
            initial_sweep.resonator.Q, initial_sweep.resonator.Q_error))
        f_baseband_bin_center = df_filterbank * np.round(f_baseband.mean() / df_filterbank)
        f_lo_final = f_lo_spacing * np.round((f_fit - f_baseband_bin_center) / f_lo_spacing)
        logger.info("f_lo_final + f_baseband_bin_center - f_r_initial = {:.3f} Hz".format(
            f_lo_final + f_baseband_bin_center - f_fit))
        #tools.set_and_attempt_external_phase_lock(ri=ri, f_lo=1e-6 * f_lo_final, f_lo_spacing=1e-6 * f_lo_spacing)
        ri.set_lo(lomhz=1e-6 * f_lo_final, chan_spacing=1e-6 * f_lo_spacing)
        for attenuation_index, attenuation in enumerate(attenuations):
            ri.set_dac_attenuator(attenuation)
            ri.set_tone_baseband_freqs(freqs=1e-6 * np.array([f_baseband[0]]), nsamp=2 ** tone_sample_exponent)
            time.sleep(1)
            tools.optimize_fft_gain(ri, fraction_of_maximum=0.5)
            time.sleep(1)
            sweep = acquire.run_sweep(ri=ri, tone_banks=1e-6 * (f_lo_final + f_baseband[:, np.newaxis]),
                                      num_tone_samples=2 ** tone_sample_exponent,
                                      length_seconds=sweep_length_seconds, state=hw.state(), verbose=True)[0]
            ri.set_tone_baseband_freqs(freqs=np.array([1e-6 * f_baseband_bin_center]), nsamp=2 ** tone_sample_exponent)
            logger.info("f_lo_final + f_baseband_bin_center - f_r = {:.3f} Hz".format(
                f_lo_final + f_baseband_bin_center- sweep.resonator.f_0))
            logger.info("Recording {:.1f} s stream with source off".format(stream_length_seconds))
            off_stream = ri.get_measurement(num_seconds=stream_length_seconds, demod=True, state=hw.state())[0]
            ri.set_modulation_output(7)
            time.sleep(3)  # Let the lock-in catch up
            logger.info("Recording {:.1f} s stream with source modulating".format(stream_length_seconds))
            mod_stream = ri.get_measurement(num_seconds=stream_length_seconds, demod=True, state=hw.state())[0]
            ri.set_modulation_output('low')
            logger.info("Recording {:.1f} s stream with source on".format(stream_length_seconds))
            on_stream = ri.get_measurement(num_seconds=stream_length_seconds, demod=True, state=hw.state())[0]
            ri.set_modulation_output('high')
            sssl = basic.SingleSweepStreamList(single_sweep=sweep, stream_list=[off_stream, mod_stream, on_stream],
                                               state={'f_index': f_index, 'attenuation_index': attenuation_index})
            npd.write(sssl)
            npd.write(ri.get_adc_measurement())
finally:
    ri.set_modulation_output('high')
    npd.close()
    print("Wrote {}".format(npd.root_path))
    print("Elapsed time {:.0f} minutes.".format((time.time() - tic) / 60))
