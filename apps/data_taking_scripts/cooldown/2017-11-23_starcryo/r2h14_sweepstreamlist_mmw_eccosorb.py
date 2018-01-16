"""
Measure one group of resonances simultaneously.
"""
from __future__ import division
import time

import numpy as np

from kid_readout.roach import analog, hardware_tools, tools
from kid_readout.measurement import acquire, basic
from kid_readout.equipment import hardware, starcryo_temps
from equipment.srs import lockin
from equipment.custom import mmwave_source
from kid_readout.settings import LOCKIN_SERIAL_PORT, CRYOGENIC_HWP_MOTOR_SERIAL_PORT
import resonances

acquire.show_settings()
acquire.show_git_status()
logger = acquire.get_script_logger(__file__)

# Parameters
suffix = 'mmw_eccosorb'
df_baseband_target = 30e3
f_sweep_span_maximum = 3e6
f_lo_spacing = 2.5e3
f_baseband_minimum = 10e6  # Keep the tones away from the LO by at least this frequency.
f_baseband_maximum = 200e6  # Place dummy tones above this frequency
sweep_length_seconds = 0.01
stream_length_seconds = 5

# Resonance frequencies
band_dict = resonances.dict_180_mK
fractional_frequency_shift = 0
for f_center in band_dict.values():
    assert f_center.ptp() < f_baseband_maximum - f_baseband_minimum
bands = band_dict.values()[3:]
attenuations_list = [(10,), (25,), (20,), (23,), (20,), (10,), (5,), (5,)][3:]

# Hardware
temperature = starcryo_temps.Temperature()
lock = lockin.SR830(serial_device=LOCKIN_SERIAL_PORT)
lock.identification  # This seems to be necessary to wake up the lockin
mmw = mmwave_source.MMWaveSource()
mmw.set_attenuator_ticks(0, 0)
mmw.multiplier_input = 'thermal'
mmw.ttl_modulation_source = 'roach_2'
mmw.waveguide_twist_angle = 0
heater = hardware.Thing(name='heater', state={'voltage': 1.70, 'resistance': 1021})
conditioner = analog.HeterodyneMarkII()
hw = hardware.Hardware(temperature, lock, mmw, heater, conditioner)
ri = hardware_tools.r2h14_with_mk2(initialize=True, use_config=False)
ri.set_modulation_output('high')
ri.iq_delay = -1
ri.adc_valon.set_ref_select(0)  # internal
assert np.all(ri.adc_valon.get_phase_locks())

# Calculate sweep parameters, LO and baseband sweep frequencies
ri_state = ri.state
tone_sample_exponent = int(np.round(np.log2(ri_state.adc_sample_rate / df_baseband_target)))
df_baseband = ri_state.adc_sample_rate / 2 ** tone_sample_exponent
df_filterbank = ri_state.adc_sample_rate / ri_state.num_filterbank_channels

# Run
npd = acquire.new_npy_directory(suffix=suffix)
tic = time.time()
try:
    for band_index, (f_center, attenuations) in enumerate(zip(bands, attenuations_list)):
        logger.info("Measuring band {:d} of {:d}".format(band_index + 1, len(bands)))
        f_sweep_span = min(f_sweep_span_maximum, np.diff(f_center).min())
        num_sweep_tones = int(f_sweep_span / df_baseband)
        logger.info("Sweeps will use {:d} tones spanning {:.1f} MHz with resolution {:.0f} Hz (2^{:d} samples)".format(
            num_sweep_tones, 1e-6 * f_sweep_span, df_baseband, tone_sample_exponent))
        f_center *= 1 + fractional_frequency_shift
        f_lo = f_center.min() - f_baseband_minimum - f_sweep_span / 2
        num_dummy_frequencies = 2 ** int(np.ceil(np.log2(f_center.size))) - f_center.size
        logger.info("Padding {:d} resonances with {:d} dummy frequencies".format(f_center.size, num_dummy_frequencies))
        f_dummy = f_baseband_maximum + 4 * df_filterbank * np.arange(1, num_dummy_frequencies + 1)
        f_center_baseband = np.concatenate((f_center - f_center.min() + f_baseband_minimum + f_sweep_span / 2, f_dummy))
        n_center_baseband = np.round(f_center_baseband / df_baseband).astype(int)
        n_sweep_offset = np.arange(-num_sweep_tones // 2, num_sweep_tones // 2)
        f_baseband = df_baseband * (n_center_baseband[np.newaxis, :] + n_sweep_offset[:, np.newaxis])
        assert np.all(ri.adc_valon.get_phase_locks())
        tools.set_and_attempt_external_phase_lock(ri=ri, f_lo=1e-6 * f_lo, f_lo_spacing=1e-6 * f_lo_spacing)
        for attenuation_index, attenuation in enumerate(attenuations):
            ri.set_dac_attenuator(attenuation)
            ri.set_tone_baseband_freqs(freqs=1e-6 * np.array([f_baseband[0, :]]), nsamp=2 ** tone_sample_exponent)
            time.sleep(1)
            tools.optimize_fft_gain(ri, fraction_of_maximum=0.5)
            time.sleep(1)
            sweep_array = acquire.run_sweep(ri=ri, tone_banks=1e-6 * (f_lo + f_baseband),
                                            num_tone_samples=2 ** tone_sample_exponent,
                                            length_seconds=sweep_length_seconds, state=hw.state())
            fit_f_r = np.array([sweep_array[number].resonator.f_0 for number in range(sweep_array.num_channels)])
            logger.info("Fit resonance frequencies [MHz] {}".format(', '.join(
                "{:.1f}".format(1e-6 * f_r) for f_r in fit_f_r)))
            fit_Q = [sweep_array[number].resonator.Q for number in range(sweep_array.num_channels)]
            logger.info("Fit quality factors {}".format(', '.join(
                '{:.3g}'.format(Q) for Q in fit_Q)))
            ri.set_tone_freqs(freqs=1e-6 * np.array(fit_f_r), nsamp=2 ** tone_sample_exponent)
            logger.info("Recording {:.1f} s stream with source off".format(stream_length_seconds))
            stream_array_off = ri.get_measurement(num_seconds=stream_length_seconds, demod=True, state=hw.state())
            ri.set_modulation_output(7)
            time.sleep(3)  # Let the lock-in catch up
            logger.info("Recording {:.1f} s stream with source modulating".format(stream_length_seconds))
            stream_array_mod = ri.get_measurement(num_seconds=stream_length_seconds, demod=True, state=hw.state())
            ri.set_modulation_output('low')
            logger.info("Recording {:.1f} s stream with source on".format(stream_length_seconds))
            stream_array_on = ri.get_measurement(num_seconds=stream_length_seconds, demod=True, state=hw.state())
            ri.set_modulation_output('high')
            sweep_stream_list = basic.SweepStreamList(sweep=sweep_array,
                                                      stream_list=[stream_array_off, stream_array_mod, stream_array_on],
                                                      state={'band_index': band_index,
                                                             'attenuation_index': attenuation_index,
                                                             'num_dummy_frequencies': num_dummy_frequencies})
            npd.write(sweep_stream_list)
            npd.write(ri.get_adc_measurement())
finally:
    ri.set_modulation_output('high')
    npd.close()
    print("Wrote {}".format(npd.root_path))
    print("Elapsed time {:.0f} minutes.".format((time.time() - tic) / 60))
