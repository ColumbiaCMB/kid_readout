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

acquire.show_settings()
acquire.show_git_status()

import logging
logger = acquire.get_script_logger(__file__, level=logging.DEBUG)


# Parameters
suffix = 'test'
attenuations = [0]
f_center = 1e6 * np.array([3420.5])
fractional_frequency_shift = 0
f_center *= (1 + fractional_frequency_shift)
df_baseband_target = 15e3
fine_sweep_num_linewidths = 5
f_sweep_span = 2e6  # The total span of the baseband tones
coarse_stride = 32
f_lo_spacing = 2.5e3  # This is the smallest resolution available
f_baseband_minimum = 100e6  # Keep the tones away from the LO by at least this frequency.
sweep_length_seconds = 0.01
stream_length_seconds = 10

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
ri = hardware_tools.r2h11nc_with_mk2(initialize=True, use_config=False)
ri.set_modulation_output('high')
ri.iq_delay = -1
ri.adc_valon.set_ref_select(0)  # internal
assert np.all(ri.adc_valon.get_phase_locks())

# Calculate sweep parameters, LO and baseband sweep frequencies
ri_state = ri.state
tone_sample_exponent = int(np.round(np.log2(ri_state.adc_sample_rate / df_baseband_target)))
df_baseband = ri_state.adc_sample_rate / 2 ** tone_sample_exponent
num_sweep_tones = int(f_sweep_span / df_baseband)
f_baseband = f_baseband_minimum + ri.state.adc_sample_rate / 2 ** tone_sample_exponent * np.arange(num_sweep_tones)
f_lo_center = f_lo_spacing * np.round((f_center - f_baseband.mean()) / f_lo_spacing)
logger.info("Fine sweep using {:d} tones spanning {:.1f} MHz with resolution {:.0f} Hz (2^{:d} samples)".format(
    num_sweep_tones, 1e-6 * f_baseband.ptp(), df_baseband, tone_sample_exponent))
logger.info("Coarse sweep using {:d} tones spanning {:.1f} MHz with resolution {:.0f} Hz (2^{:d} samples)".format(
    num_sweep_tones // coarse_stride, 1e-6 * f_baseband.ptp(), coarse_stride * df_baseband, tone_sample_exponent))

# Run
npd = acquire.new_npy_directory(suffix=suffix)
tic = time.time()
try:
    for lo_index, f_lo in enumerate(f_lo_center):
        assert np.all(ri.adc_valon.get_phase_locks())
        tools.set_and_attempt_external_phase_lock(ri=ri, f_lo=1e-6 * f_lo, f_lo_spacing=1e-6 * f_lo_spacing)
        for attenuation_index, attenuation in enumerate(attenuations):
            ri.set_dac_attenuator(attenuation)
            #ri.set_tone_baseband_freqs(freqs=1e-6 * np.array([f_baseband[0]]), nsamp=2 ** tone_sample_exponent)
            #time.sleep(1)
            #tools.optimize_fft_gain(ri, fraction_of_maximum=0.5)
            ri.set_fft_gain(4)
            coarse_state = hw.state()
            coarse_state['lo_index'] = lo_index
            coarse_state['attenuation_index'] = attenuation_index
            coarse_sweep = acquire.run_sweep(ri=ri, tone_banks=1e-6 * (f_lo + f_baseband[::coarse_stride, np.newaxis]),
                                             num_tone_samples=2 ** tone_sample_exponent,
                                             length_seconds=stream_length_seconds, state=coarse_state,
                                             verbose=True)[0]
            npd.write(coarse_sweep)
            coarse_f_r = coarse_sweep.resonator.f_0
            coarse_Q = coarse_sweep.resonator.Q
            logger.info("Coarse sweep f_r = {:.3f} MHz +/- {:.0f} Hz".format(1e-6 * coarse_f_r,
                                                                             coarse_sweep.resonator.f_0_error))
            logger.info("Coarse sweep Q = {:.0f} +/- {:.0f}".format(coarse_Q, coarse_sweep.resonator.Q_error))
            raise Exception()
            df_filterbank = calculate.stream_sample_rate(ri_state)
            f_baseband_bin_center = df_filterbank * np.round(f_baseband.mean() / df_filterbank)
            f_lo_fine = f_lo_spacing * np.round((coarse_f_r - f_baseband_bin_center) / f_lo_spacing)
            assert np.all(ri.adc_valon.get_phase_locks())
            tools.set_and_attempt_external_phase_lock(ri=ri, f_lo=1e-6 * f_lo, f_lo_spacing=1e-6 * f_lo_spacing)
            #fine_indices = np.where(np.abs(f_lo_fine + f_baseband - coarse_f_r) <=
            #                        (fine_sweep_num_linewidths / 2) * (coarse_f_r / coarse_Q))[0]
            fine_indices = np.arange(f_baseband.size)
            fine_sweep = acquire.run_sweep(ri=ri, tone_banks=1e-6 * (f_lo + f_baseband[fine_indices, np.newaxis]),
                                           num_tone_samples=2 ** tone_sample_exponent,
                                           length_seconds=stream_length_seconds, state=hw.state())[0]
            ri.set_tone_freqs(np.array([]))

            logger.info("Recording {:.1f} s stream with source off".format(stream_length_seconds))
            off_stream = ri.get_measurement(num_seconds=stream_length_seconds, demod=True, state=hw.state())[0]
            ri.set_modulation_output(7)
            logger.info("Recording {:.1f} s stream with source modulating".format(stream_length_seconds))
            mod_stream = ri.get_measurement(num_seconds=stream_length_seconds, demod=True, state=hw.state())[0]
            ri.set_modulation_output('high')
            sweep_stream_list = basic.SingleSweepStreamList(single_sweep=fine_sweep,
                                                            stream_list=[off_stream, mod_stream],
                                                            state={'lo_index': lo_index,
                                                                   'attenuation_index': attenuation_index})
            npd.write(sweep_stream_list)
            npd.write(ri.get_adc_measurement())
finally:
    ri.set_modulation_output('high')
    ri.set_dac_attenuator(62)
    npd.close()
    print("Wrote {}".format(npd.root_path))
    print("Elapsed time {:.0f} minutes.".format((time.time() - tic) / 60))
