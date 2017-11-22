"""
Record one SweepStreamArray for 8 resonators.
"""
from __future__ import division
import time

import numpy as np

from kid_readout.roach import r2baseband, analog
from kid_readout.measurement import acquire, basic
from kid_readout.equipment import hardware
from kid_readout.settings import ROACH2_IP, ROACH2_VALON
from kid_readout.equipment import starcryo_temps

acquire.show_settings()
acquire.show_git_status()
logger = acquire.get_script_logger(__file__)

# Parameters
suffix = 'magnetic_shield_high_4'
all_f0_MHz = np.array([68.272, 73.035, 93.147, 95.155, 105.228, 119.202, 123.462, 126.594])
f0_MHz = all_f0_MHz[4:]
attenuations = [41, 47, 53, 59]
fft_gains = [6, 6, 6, 6]
tone_sample_exponent_coarse = 18
tone_sample_exponent_fine = 20
sweep_length_seconds = 0.1
stream_length_seconds = 30
num_sweep_tones_coarse = 128
integer_shift_coarse = -num_sweep_tones_coarse // 4  # Shift the center of the coarse sweep
num_sweep_tones_fine = 64


# Hardware
conditioner = analog.Baseband()
shield = hardware.Thing(name='magnetic_shield', state={})
hw = hardware.Hardware(conditioner, shield)
ri = r2baseband.Roach2Baseband(roachip=ROACH2_IP, adc_valon=ROACH2_VALON)
ri.initialize(use_config=False)

# Calculate frequencies
num_tone_samples_coarse = 2 ** tone_sample_exponent_coarse
f_resolution_coarse = ri.state.adc_sample_rate / num_tone_samples_coarse
offset_integers_coarse = np.arange(-num_sweep_tones_coarse // 2, num_sweep_tones_coarse // 2) + integer_shift_coarse
offset_frequencies_coarse_MHz = 1e-6 * f_resolution_coarse * offset_integers_coarse
sweep_frequencies_coarse_MHz = offset_frequencies_coarse_MHz[:, np.newaxis] + f0_MHz[np.newaxis, :]

num_tone_samples_fine = 2 ** tone_sample_exponent_fine
f_resolution_fine = ri.state.adc_sample_rate / num_tone_samples_fine
offset_integers_fine = np.arange(-num_sweep_tones_fine // 2, num_sweep_tones_fine // 2)
offset_frequencies_fine_MHz = 1e-6 * f_resolution_fine * offset_integers_fine

logger.info("Coarse frequency spacing is {:.3f} kHz".format(1e3 * (offset_frequencies_coarse_MHz[1] -
                                                                   offset_frequencies_coarse_MHz[0])))
logger.info("Coarse sweep span is {:.3f} MHz".format(offset_frequencies_coarse_MHz.ptp()))
logger.info("Fine frequency spacing is {:.3f} kHz".format(1e3 * (offset_frequencies_fine_MHz[1] -
                                                                 offset_frequencies_fine_MHz[0])))
logger.info("Fine sweep span is {:.3f} MHz".format(offset_frequencies_fine_MHz.ptp()))
raw_input("Press enter to continue or ctrl-C to abort.")

# Run
ncf = acquire.new_nc_file(suffix=suffix)
tic = time.time()
try:
    for fft_gain, attenuation in zip(fft_gains, attenuations):
        ri.set_fft_gain(fft_gain)
        ri.set_dac_attenuator(attenuation)
        state = hw.state()
        start_time = time.time()  # Use this time to retrieve the temperature (see below).
        logger.info("Recording {:.1f} s coarse sweep around MHz frequencies {}".format(sweep_length_seconds, f0_MHz))
        sweep_array_coarse = acquire.run_sweep(ri=ri, tone_banks=sweep_frequencies_coarse_MHz,
                                               length_seconds=sweep_length_seconds,
                                               num_tone_samples=num_tone_samples_coarse, state=state)
        ncf.write(sweep_array_coarse)
        fit_f0_coarse_MHz = np.array([1e-6 * sweep_array_coarse[n].resonator.f_0
                                      for n in range(sweep_array_coarse.num_channels)])
        logger.info("coarse - initial [kHz]: {}".format(', '.join(['{:.3f}'.format(1e3 * df0)
                                                                   for df0 in fit_f0_coarse_MHz - f0_MHz])))
        sweep_frequencies_fine_MHz = offset_frequencies_fine_MHz[:, np.newaxis] + fit_f0_coarse_MHz[np.newaxis, :]
        logger.info("Recording {:.1f} s fine sweep around MHz frequencies {}".format(sweep_length_seconds,
                                                                                     fit_f0_coarse_MHz))
        sweep_array_fine = acquire.run_sweep(ri=ri, tone_banks=sweep_frequencies_fine_MHz,
                                             length_seconds=sweep_length_seconds,
                                             num_tone_samples=num_tone_samples_fine)
        fit_f0_fine_MHz = np.array([1e-6 * sweep_array_fine[n].resonator.f_0
                                    for n in range(sweep_array_fine.num_channels)])
        logger.info("fine - coarse [kHz]: {}".format(', '.join(['{:.3f}'.format(1e3 * df0)
                                                                for df0 in fit_f0_fine_MHz - fit_f0_coarse_MHz])))
        f_stream_MHz = ri.set_tone_freqs(np.array(fit_f0_fine_MHz), nsamp=num_tone_samples_fine)
        ri.select_bank(0)
        ri.select_fft_bins(np.arange(f_stream_MHz.size))
        # Retrieve the start temperature now to avoid the log file warning.
        state['temperature'] = {'package': starcryo_temps.get_temperatures_at(start_time)[0]}
        logger.info("Recording {:.1f} s streams at MHz frequencies {}".format(stream_length_seconds, f_stream_MHz))
        stream_array = ri.get_measurement(num_seconds=stream_length_seconds)
        sweep_stream_array = basic.SweepStreamArray(sweep_array=sweep_array_fine, stream_array=stream_array,
                                                    state=state,
                                                    description='attenuation {:.1f} dB'.format(attenuation))
        ncf.write(sweep_stream_array)
        # Record an ADCSnap with the stream tones playing.
        ncf.write(ri.get_adc_measurement())
finally:
        ncf.close()
        print("Wrote {}".format(ncf.root_path))
        print("Elapsed time {:.0f} minutes.".format((time.time() - tic) / 60))
