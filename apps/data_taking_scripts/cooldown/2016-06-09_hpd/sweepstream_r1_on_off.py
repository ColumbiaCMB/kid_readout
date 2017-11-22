"""
Measure one resonator per LO frequency. Since each measurement has only one channel, record SingleSweepStreams.
"""
import time
import logging

import numpy as np

from kid_readout.utils import log
from kid_readout.roach import hardware_tools, analog
from kid_readout.measurement import acquire, basic
from kid_readout.equipment import hardware
from kid_readout.settings import CRYOSTAT
if CRYOSTAT.lower() == 'hpd':
    from kid_readout.equipment import hpd_temps as temps
elif CRYOSTAT.lower() == 'starcryo':
    from kid_readout.equipment import starcryo_temps as temps
else:
    raise ValueError("Unknown cryostat: {}".format(repr(CRYOSTAT)))

logger = logging.getLogger('kid_readout')
logging.getLogger('kid_readout.roach').setLevel(logging.WARNING)
stream_handler = log.default_handler
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)
logger.addHandler(log.file_handler(__file__))

# Parameters
magnet_distance_mm = 220
all_f0_MHz = np.array([2432, 3488, 3629, 3800])
f0_MHz = all_f0_MHz
frequency_shift = 1 - 0
f0_MHz *= frequency_shift
attenuations = [10, 20, 30, 40, 50, 60]
sweep_interval = 32
tone_sample_exponent = 18
sweep_length_seconds = 0.1
stream_length_seconds = 30
lo_round_to_MHz = 2
f_minimum = 10e6  # Keep the tones away from the LO by at least this frequency.
f_stream_offset_MHz = 10  # Set a second tone away from the resonance by this amount
num_sweep_tones = 127

# Hardware
conditioner = analog.HeterodyneMarkII()
magnet = hardware.Thing('canceling_magnet_quincunx',
                        {'orientation': 'up',
                         'ruler_to_base_mm': magnet_distance_mm})
hw = hardware.Hardware(conditioner, magnet)
ri = hardware_tools.r1_with_mk2()
ri.set_modulation_output('high')

# Calculate LO and baseband frequencies
num_tone_samples = 2**tone_sample_exponent
f_resolution = ri.state.adc_sample_rate / num_tone_samples
minimum_integer = int(f_minimum / f_resolution)
offset_integers = minimum_integer + sweep_interval * np.arange(num_sweep_tones)
offset_frequencies_MHz = 1e-6 * f_resolution * offset_integers
offset_array_MHz = offset_frequencies_MHz[:, np.newaxis] + np.array([0, f_stream_offset_MHz])[np.newaxis, :]
lo_MHz = lo_round_to_MHz * np.round((f0_MHz - offset_frequencies_MHz.mean()) / lo_round_to_MHz)
logger.info("Frequency spacing is {:.1f} kHz".format(1e3 * (offset_frequencies_MHz[1] - offset_frequencies_MHz[0])))
logger.info("Sweep span is {:.1f} MHz".format(offset_frequencies_MHz.ptp()))

# Run
ncf = acquire.new_nc_file(suffix='sweep_stream_on_off')
tic = time.time()
try:
    ri.set_tone_baseband_freqs(offset_array_MHz, nsamp=num_tone_samples)
    for lo_index, lo in enumerate(lo_MHz):
        ri.set_lo(lomhz=lo, chan_spacing=lo_round_to_MHz)
        logger.info("Set LO to {:.3f} MHz".format(lo))
        for attenuation in attenuations:
            ri.set_dac_attenuator(attenuation)
            logger.info("Set DAC attenuation to {:.1f} dB".format(attenuation))
            state = hw.state()
            state['lo_index'] = lo_index
            state['temperature'] = {'package': temps.get_temperature_at(time.time())}
            sweep_array = acquire.run_loaded_sweep(ri, length_seconds=sweep_length_seconds,
                                                   tone_bank_indices=np.arange(num_sweep_tones))
            on_sweep = sweep_array[0]
            off_sweep = sweep_array[1]
            f0_MHz = 1e-6 * on_sweep.resonator.f_0
            logger.info("Fit resonance frequency is {:.3f} MHz".format(f0_MHz))
            # Overwrite the last waveform after the first loop.
            is_first_loop = (lo == lo_MHz[0]) & (attenuation == attenuations[0])
            f_stream_MHz = ri.add_tone_freqs(np.array([f0_MHz, f0_MHz + f_stream_offset_MHz]),
                                             overwrite_last=~is_first_loop)
            # NB: it may be true that select_bank() has to be called before select_fft_bins()
            ri.select_bank(num_sweep_tones)
            ri.select_fft_bins(np.arange(f_stream_MHz.size))
            logger.info("Recording {:.1f} s streams at MHz frequencies {}".format(stream_length_seconds, f_stream_MHz))
            stream_array = ri.get_measurement(num_seconds=stream_length_seconds)
            on_stream = stream_array[0]
            off_stream = stream_array[1]
            sweep_stream = basic.SingleSweepStream(sweep=on_sweep, stream=on_stream, state=state,
                                                   description='f_0 = {:.1f}'.format(f0_MHz))
            logger.debug("Writing on-resonance SingleSweepStream.")
            ncf.write(sweep_stream)
            logger.debug("Writing off-resonance SingleSweep.")
            ncf.write(off_sweep)
            logger.debug("Writing off-resonance SingleStream.")
            ncf.write(off_stream)
            # Record an ADCSnap with the stream tones playing.
            logger.debug("Recording ADCSnap.")
            adc_snap = ri.get_adc_measurement()
            logger.debug("Writing ADCSnap.")
            ncf.write(adc_snap)
finally:
        ncf.close()
        print("Wrote {}".format(ncf.root_path))
        print("Elapsed time {:.0f} minutes.".format((time.time() - tic) / 60))
