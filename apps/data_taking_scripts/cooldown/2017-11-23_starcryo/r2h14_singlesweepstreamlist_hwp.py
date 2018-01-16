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
from xystage import stepper
from kid_readout.settings import LOCKIN_SERIAL_PORT, CRYOGENIC_HWP_MOTOR_SERIAL_PORT

acquire.show_settings()
acquire.show_git_status()
logger = acquire.get_script_logger(__file__)

# Parameters
suffix = 'hwp'
attenuations = [20]
"""
f_center = 1e6 * np.array([
    2757.5,
    2778.3,
    2792.0,
    2816.0,
    2872.0,  # low Q
    2921.5,
    2998.5,
    3001.0,
    3085.0,
    3229.0, # done
    3316.5,
    3347.0,
    3370.5,
    3420.5,
    3922.0  # no-man's land
    ])
"""
f_center = 1e6 * np.array([2757.5])
fractional_frequency_shift = 0
f_center *= (1 + fractional_frequency_shift)
df_baseband_target = 30e3
fine_sweep_num_linewidths = 5
f_sweep_span = 3e6  # The total span of the baseband tones
coarse_stride = 2
f_lo_spacing = 2.5e3  # This is the smallest resolution available
f_baseband_minimum = 100e6  # Keep the tones away from the LO by at least this frequency.
sweep_length_seconds = 0.01
stream_length_seconds = 2
num_hwp_angles = 100
num_hwp_increments_per_angle = 1
wait_between_increments = 0
wait_between_angles = 5

# Hardware
temperature = starcryo_temps.Temperature()
lock = lockin.SR830(serial_device=LOCKIN_SERIAL_PORT)
lock.identification  # This seems to be necessary to wake up the lockin
mmw = mmwave_source.MMWaveSource()
mmw.set_attenuator_ticks(0, 0)
mmw.multiplier_input = 'thermal'
mmw.ttl_modulation_source = "roach_2"
mmw.waveguide_twist_angle = 0
hwp = stepper.SimpleStepper(port=CRYOGENIC_HWP_MOTOR_SERIAL_PORT)
time.sleep(3)
hwp.increment()
time.sleep(1)
hwp.decrement()
conditioner = analog.HeterodyneMarkII()
hw = hardware.Hardware(temperature, lock, mmw, hwp, conditioner)
ri = hardware_tools.r2h14_with_mk2(initialize=True, use_config=False)
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
logger.info("Coarse sweep using {:d} tones spanning {:.1f} MHz with resolution {:.0f} Hz (2^{:d} samples)".format(
    num_sweep_tones // coarse_stride, 1e-6 * f_baseband.ptp(), coarse_stride * df_baseband, tone_sample_exponent))
logger.info("Fine sweep using {:d} tones spanning {:.1f} MHz with resolution {:.0f} Hz (2^{:d} samples)".format(
    num_sweep_tones, 1e-6 * f_baseband.ptp(), df_baseband, tone_sample_exponent))

# Run
npd = acquire.new_npy_directory(suffix=suffix)
tic = time.time()
try:
    for lo_index, f_lo in enumerate(f_lo_center):
        assert np.all(ri.adc_valon.get_phase_locks())
        tools.set_and_attempt_external_phase_lock(ri=ri, f_lo=1e-6 * f_lo, f_lo_spacing=1e-6 * f_lo_spacing)
        for attenuation_index, attenuation in enumerate(attenuations):
            ri.set_dac_attenuator(attenuation)
            ri.set_tone_baseband_freqs(freqs=1e-6 * np.array([f_baseband[0]]), nsamp=2 ** tone_sample_exponent)
            time.sleep(1)
            tools.optimize_fft_gain(ri, fraction_of_maximum=0.5)
            time.sleep(1)
            coarse_state = hw.state()
            coarse_state['lo_index'] = lo_index
            coarse_state['attenuation_index'] = attenuation_index
            coarse_sweep = acquire.run_sweep(ri=ri, tone_banks=1e-6 * (f_lo + f_baseband[::coarse_stride, np.newaxis]),
                                             num_tone_samples=2 ** tone_sample_exponent,
                                             length_seconds=sweep_length_seconds, state=coarse_state,
                                             verbose=True)[0]
            npd.write(coarse_sweep)
            coarse_f_r = coarse_sweep.resonator.f_0
            coarse_Q = coarse_sweep.resonator.Q
            logger.info("Coarse sweep f_r = {:.3f} MHz +/- {:.0f} Hz".format(1e-6 * coarse_f_r,
                                                                             coarse_sweep.resonator.f_0_error))
            logger.info("Coarse sweep Q = {:.0f} +/- {:.0f}".format(coarse_Q, coarse_sweep.resonator.Q_error))
            df_filterbank = calculate.stream_sample_rate(ri_state)
            f_baseband_bin_center = df_filterbank * np.round(f_baseband.mean() / df_filterbank)
            f_lo_fine = f_lo_spacing * np.round((coarse_f_r - f_baseband_bin_center) / f_lo_spacing)
            ri.set_lo(lomhz=1e-6 * f_lo_fine, chan_spacing=1e-6 * f_lo_spacing)
            fine_sweep = acquire.run_sweep(ri=ri, tone_banks=1e-6 * (f_lo + f_baseband[:, np.newaxis]),
                                           num_tone_samples=2 ** tone_sample_exponent,
                                           length_seconds=sweep_length_seconds, state=hw.state(), verbose=True)[0]
            ri.set_tone_baseband_freqs(freqs=np.array([1e-6 * f_baseband_bin_center]), nsamp=2 ** tone_sample_exponent)
            logger.info("f_r - (f_lo + f_baseband_bin_center) = {:.4f} Hz".format(
                fine_sweep.resonator.f_0 - (f_lo_fine + f_baseband_bin_center)))
            ri.set_modulation_output(7)
            time.sleep(3)  # The lock-in apparently takes some time to catch up.
            streams = []
            for hwp_index in range(num_hwp_angles):
                logger.info("Recording {:.1f} s stream with source modulating at HWP angle {:d} of {:d}".format(
                    stream_length_seconds, hwp_index, num_hwp_angles))
                state = hw.state()
                state['hwp_index'] = hwp_index
                streams.append(ri.get_measurement(num_seconds=stream_length_seconds, demod=True, state=state)[0])
                logger.info("Incrementing HWP by {:d} steps".format(num_hwp_increments_per_angle))
                for _ in range(num_hwp_increments_per_angle):
                    hwp.increment()
                    time.sleep(wait_between_increments)
                time.sleep(wait_between_angles)
            sweep_stream_list = basic.SingleSweepStreamList(single_sweep=fine_sweep, stream_list=streams,
                                                            state={'lo_index': lo_index,
                                                                   'attenuation_index': attenuation_index})
            npd.write(sweep_stream_list)
            npd.write(ri.get_adc_measurement())
finally:
    ri.set_modulation_output('high')
    npd.close()
    print("Wrote {}".format(npd.root_path))
    print("Elapsed time {:.0f} minutes.".format((time.time() - tic) / 60))
