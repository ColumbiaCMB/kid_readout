"""
Measure one resonator per LO frequency. Since each measurement has only one channel, record SingleSweepStreams.
"""
import time

import numpy as np

from kid_readout.roach import analog, hardware_tools
from kid_readout.measurement import acquire, basic
from kid_readout.equipment import hardware

acquire.show_settings()
acquire.show_git_status()
logger = acquire.get_script_logger(__file__)

# Parameters
suffix = 'uncontrolled'
all_f0_MHz = np.array([2522.24, 2605.96, 2723.65, 2787.96, 3851.13])
f0_MHz = all_f0_MHz
frequency_shift = 1
f0_MHz *= frequency_shift
attenuations = [10, 20, 30]
sweep_interval = 2
tone_sample_exponent = 18
sweep_length_seconds = 0.1
stream_length_seconds = 10
lo_round_to_MHz = 0.1
f_minimum = 10e6  # Keep the tones away from the LO by at least this frequency.
f_stream_offset_MHz = 10  # Set a second tone away from the resonance by this amount
num_sweep_tones = 127
fft_gain = 3

# Hardware
conditioner = analog.HeterodyneMarkII()
shield = hardware.Thing(name='mu_metal_pocket', state={'orientation': 'horizontal'})
hw = hardware.Hardware(conditioner, shield)
ri = hardware_tools.r1h11_with_mk2(initialize=True, use_config=False)
ri.adc_valon.set_ref_select(0)  # internal
ri.lo_valon.set_ref_select(1)  # external
ri.set_fft_gain(fft_gain)

# Calculate LO and baseband frequencies
num_tone_samples = 2**tone_sample_exponent
f_resolution = ri.state.adc_sample_rate / num_tone_samples
minimum_integer = int(f_minimum / f_resolution)
offset_integers = minimum_integer + sweep_interval * np.arange(num_sweep_tones)
offset_frequencies_MHz = 1e-6 * f_resolution * offset_integers
offset_array_MHz = offset_frequencies_MHz[:, np.newaxis] + np.array([0, f_stream_offset_MHz])[np.newaxis, :]
all_lo_MHz = lo_round_to_MHz * np.round((f0_MHz - offset_frequencies_MHz.mean()) / lo_round_to_MHz)
logger.info("Frequency spacing is {:.1f} kHz".format(1e3 * (offset_frequencies_MHz[1] - offset_frequencies_MHz[0])))
logger.info("Sweep span is {:.1f} MHz".format(offset_frequencies_MHz.ptp()))

# Run
npd = acquire.new_npy_directory(suffix=suffix)
tic = time.time()
try:
    ri.set_tone_baseband_freqs(offset_array_MHz, nsamp=num_tone_samples)
    for lo_index, lo_MHz in enumerate(all_lo_MHz):
        ri.set_lo(lomhz=lo_MHz, chan_spacing=lo_round_to_MHz)
        for attenuation_index, attenuation in enumerate(attenuations):
            assert np.all(ri.adc_valon.get_phase_locks())
            assert np.all(ri.lo_valon.get_phase_locks())
            ri.set_dac_attenuator(attenuation)
            state = hw.state()
            state['lo_index'] = lo_index
            sweep_array = acquire.run_loaded_sweep(ri, length_seconds=sweep_length_seconds,
                                                   tone_bank_indices=np.arange(num_sweep_tones),
                                                   demod=True)
            on_sweep = sweep_array[0]
            off_sweep = sweep_array[1]
            off_sweep.state = state
            f0_MHz = 1e-6 * on_sweep.resonator.f_0
            logger.info("Fit resonance frequency is {:.3f} MHz".format(f0_MHz))
            # Overwrite the last waveform after the first loop.
            is_not_first_loop = (lo_index > 0) or (attenuation_index > 0)
            f_stream_MHz = ri.add_tone_freqs(np.array([f0_MHz, f0_MHz + f_stream_offset_MHz]),
                                             overwrite_last=is_not_first_loop)
            ri.select_bank(num_sweep_tones)
            ri.select_fft_bins(np.arange(f_stream_MHz.size))
            logger.info("Recording {:.1f} s streams at MHz frequencies {}".format(stream_length_seconds, f_stream_MHz))
            stream_array = ri.get_measurement(num_seconds=stream_length_seconds,
                                              demod=True)
            on_stream = stream_array[0]
            off_stream = stream_array[1]
            off_stream.state = state
            sweep_stream = basic.SingleSweepStream(sweep=on_sweep, stream=on_stream, state=state,
                                                   description='f_0 = {:.1f}'.format(f0_MHz))
            npd.write(sweep_stream)
            npd.write(off_sweep)
            npd.write(off_stream)
            # Record an ADCSnap with the stream tones playing.
            npd.write(ri.get_adc_measurement())
finally:
    npd.close()
    print("Wrote {}".format(npd.root_path))
    print("Elapsed time {:.0f} minutes.".format((time.time() - tic) / 60))
