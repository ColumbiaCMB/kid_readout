"""
For each resonator, take one sweep then a bunch of streams at different places across the filterbank bin, moving the
baseband frequency and LO frequency in opposite directions.

This seems to fail because the phase and/or amplitude of the streams doesn't match the sweep.
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
suffix = 'r1h11_lo_sweep'
wait = 5
fft_gain = 3
num_tones_sweep = 255
f_minimum = 10e6  # Keep the tones away from the LO by at least this frequency.
f_lo_round_to_MHz = 2.5e-3  # Allegedly the minimum resolution
all_f0_MHz = np.array([2254.837, 2326.842, 2483.490, 3313.270, 3378.300, 3503.600, 3524.435])
f0 = all_f0_MHz[4:5]
attenuations = [33]
tone_sample_exponent_sweep = 16
tones_per_bin_exponent = 3  # Eight points per bin
offsets_stream = range(-2 ** (tones_per_bin_exponent - 1), 2 ** (tones_per_bin_exponent - 1) + 1)  # Hit both bin edges
length_seconds_sweep = 0.1
length_seconds_stream = 10


# Hardware
conditioner = analog.HeterodyneMarkII()
magnet = hardware.Thing(name='magnet_array', state={'orientation': 'up',
                                                    'distance_from_base_mm': 276})
hw = hardware.Hardware(conditioner, magnet)
ri = hardware_tools.r1h11_with_mk2(initialize=True, use_config=False)
ri.adc_valon.set_ref_select(0)  # internal
ri.lo_valon.set_ref_select(1)  # external
ri.set_fft_gain(fft_gain)

# Calculate LO and baseband frequencies
roach_state = ri.state
f_tone_sweep = roach_state.adc_sample_rate / 2 ** tone_sample_exponent_sweep
f_filterbank = roach_state.adc_sample_rate / roach_state.num_filterbank_channels
minimum_integer_sweep = int(f_minimum / f_tone_sweep)
offset_integers_sweep = minimum_integer_sweep + np.arange(num_tones_sweep)
offset_frequencies_MHz_sweep = 1e-6 * f_tone_sweep * offset_integers_sweep
offset_array_MHz_sweep = offset_frequencies_MHz_sweep[:, np.newaxis]
all_f_lo_MHz_sweep = f_lo_round_to_MHz * np.round((f0 - offset_frequencies_MHz_sweep.mean()) / f_lo_round_to_MHz)
logger.info("Frequency spacing is {:.1f} kHz".format(1e3 * (offset_frequencies_MHz_sweep[1] -
                                                            offset_frequencies_MHz_sweep[0])))
logger.info("Sweep span is {:.1f} MHz".format(offset_frequencies_MHz_sweep.ptp()))
num_tone_samples_stream = 2 ** tones_per_bin_exponent * roach_state.num_filterbank_channels
f_tone_stream = roach_state.adc_sample_rate / num_tone_samples_stream

# Run
npd = acquire.new_npy_directory(suffix=suffix)
tic = time.time()
try:
    ri.set_tone_baseband_freqs(offset_array_MHz_sweep, nsamp=2 ** tone_sample_exponent_sweep)
    for lo_index, f_lo_MHz_sweep in enumerate(all_f_lo_MHz_sweep):
        ri.set_lo(lomhz=f_lo_MHz_sweep, chan_spacing=f_lo_round_to_MHz)
        logger.info("Set LO to {:.3f} MHz".format(f_lo_MHz_sweep))
        for attenuation_index, attenuation in enumerate(attenuations):
            assert np.all(ri.adc_valon.get_phase_locks())
            assert np.all(ri.lo_valon.get_phase_locks())
            ri.set_dac_attenuator(attenuation)
            state = hw.state()
            state['lo_index'] = lo_index
            state['lo_valon'] = {'frequency_a': 1e6 * ri.lo_valon.get_frequency_a(),
                                 'frequency_b': 1e6 * ri.lo_valon.get_frequency_b()}
            sweep = acquire.run_loaded_sweep(ri, length_seconds=length_seconds_sweep,
                                             tone_bank_indices=np.arange(num_tones_sweep), demod=True, state=state)[0]
            sweep.number = None
            f0 = sweep.resonator.f_0
            logger.info("Fit resonance frequency is {:.3f} MHz".format(1e-6 * f0))
            n_filterbank = int(round(f_minimum / f_filterbank + 1))
            logger.info("Filterbank bin index is {:d}".format(n_filterbank))
            n_tone_center = 2 ** tones_per_bin_exponent * n_filterbank
            logger.info("Center tone bin index is {:d}".format(n_tone_center))
            streams = []
            for offset in offsets_stream:
                logger.info("Recording {:.1f} s stream at tone offset {:d}".format(length_seconds_stream, offset))
                n_tone = n_tone_center + offset
                f_lo_MHz_stream = f_lo_round_to_MHz * round(1e-6 * (f0 - f_tone_stream * n_tone) / f_lo_round_to_MHz)
                ri.set_lo(lomhz=f_lo_MHz_stream, chan_spacing=f_lo_round_to_MHz)
                logger.info("Set LO to {:.6f} MHz".format(f_lo_MHz_stream))
                ri.set_tone_bins(bins=np.array([n_tone]), nsamp=num_tone_samples_stream)
                logger.info("Resetting filterbank bin from {:d} to {:d}".format(ri.fft_bins[0, 0], n_filterbank))
                ri.fft_bins = np.atleast_2d(np.array([n_filterbank]))
                ri.select_bank(0)
                ri.select_fft_bins(np.array([0]))
                time.sleep(wait)
                state = hw.state()
                state['lo_index'] = lo_index
                state['lo_valon'] = {'frequency_a': 1e6 * ri.lo_valon.get_frequency_a(),
                                     'frequency_b': 1e6 * ri.lo_valon.get_frequency_b()}
                stream = ri.get_measurement(num_seconds=length_seconds_stream, demod=True, state=state)[0]
                stream.number = None
                streams.append(stream)
                npd.write(ri.get_adc_measurement())
            sssl = basic.SingleSweepStreamList(single_sweep=sweep, stream_list=streams,
                                               description='f_0 = {:.1f}'.format(f0))
            npd.write(sssl)
finally:
    npd.close()
    print("Wrote {}".format(npd.root_path))
    print("Elapsed time {:.0f} minutes.".format((time.time() - tic) / 60))
