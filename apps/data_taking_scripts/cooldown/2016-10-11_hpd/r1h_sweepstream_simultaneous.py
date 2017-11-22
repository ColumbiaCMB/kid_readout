"""
Measure several resonators per LO frequency and record SweepStreamArrays.
"""
import time

import numpy as np

from kid_readout.roach import hardware_tools, analog
from kid_readout.measurement import acquire, basic
from kid_readout.equipment import hardware

acquire.show_settings()
acquire.show_git_status()
logger = acquire.get_script_logger(__file__)

# Parameters
suffix = 'sweep_stream_simultaneous_locked'
frequency_shift_ppm = 0
low_f0_MHz = np.array([2254.837, 2326.842, 2483.490, 2580]) * (1 - 1e-6 * frequency_shift_ppm)
high_f0_MHz = np.array([3313.270, 3378.300, 3503.600, 3524.435]) * (1 - 1e-6 * frequency_shift_ppm)
bands_MHz = [low_f0_MHz, high_f0_MHz]
all_lo_MHz = [2400, 3450]  # Set these by hand for now
lo_round_to_MHz = 0.1
attenuations = [20, 30]
sweep_interval = 4
tone_sample_exponent = 18
sweep_length_seconds = 0.1
stream_length_seconds = 60
num_sweep_tones = 255

# Hardware
conditioner = analog.HeterodyneMarkII()
magnet = hardware.Thing(name='magnet_array', state={'orientation': 'up',
                                                    'distance_from_base_mm': 276})
hw = hardware.Hardware(conditioner, magnet)
ri = hardware_tools.r1_with_mk2()
ri.initialize(use_config=False)
ri.adc_valon.set_ref_select(1)  # external
ri.lo_valon.set_ref_select(0)  # internal

# Calculate LO and baseband frequencies
num_tone_samples = 2**tone_sample_exponent
f_resolution = ri.state.adc_sample_rate / num_tone_samples
offset_integers = sweep_interval * (np.arange(num_sweep_tones) - np.floor(num_sweep_tones / 2))
offset_frequencies_MHz = 1e-6 * f_resolution * offset_integers
logger.info("Frequency spacing is {:.1f} kHz".format(1e-3 * sweep_interval * f_resolution))
logger.info("Sweep span is {:.1f} MHz".format(offset_frequencies_MHz.ptp()))

# Run
npd = acquire.new_npy_directory(suffix=suffix)
tic = time.time()
try:
    for band_MHz, lo_MHz in zip(bands_MHz, all_lo_MHz):
        ri.set_lo(lomhz=lo_MHz, chan_spacing=lo_round_to_MHz)
        logger.info("Set LO to {:.3f} MHz".format(lo_MHz))
        offset_array_MHz = (offset_frequencies_MHz[:, np.newaxis] + (band_MHz - lo_MHz)[np.newaxis, :])
        ri.set_tone_baseband_freqs(offset_array_MHz, nsamp=num_tone_samples)
        for attenuation_index, attenuation in enumerate(attenuations):
            assert np.all(ri.adc_valon.get_phase_locks())
            assert np.all(ri.lo_valon.get_phase_locks())
            ri.set_dac_attenuator(attenuation)
            state = hw.state()
            sweep_array = acquire.run_loaded_sweep(ri, length_seconds=sweep_length_seconds,
                                                   tone_bank_indices=np.arange(num_sweep_tones))
            fit_f0_MHz = 1e-6 * np.array([sweep_array[index].resonator.f_0
                                          for index in range(sweep_array.num_channels)])
            logger.info("Fit resonance frequencies in MHz are {}".format(fit_f0_MHz))
            # Overwrite the last waveform after the first loop.
            is_not_first_loop = attenuation_index > 0
            f_stream_MHz = ri.add_tone_freqs(freqs=fit_f0_MHz, overwrite_last=is_not_first_loop)
            ri.select_bank(num_sweep_tones)
            ri.select_fft_bins(np.arange(f_stream_MHz.size))
            logger.info("Recording {:.1f} s streams at MHz frequencies {}".format(stream_length_seconds, f_stream_MHz))
            stream_array = ri.get_measurement(num_seconds=stream_length_seconds)
            sweep_stream = basic.SweepStreamArray(sweep_array=sweep_array, stream_array=stream_array, state=state)
            npd.write(sweep_stream)
            npd.write(ri.get_adc_measurement())
finally:
    npd.close()
    print("Wrote {}".format(npd.root_path))
    print("Elapsed time {:.0f} minutes.".format((time.time() - tic) / 60))
