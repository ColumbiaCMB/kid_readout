"""
Record one SweepStreamArray for 3 resonators plus 1 dummy at each attenuation.
"""
import time

import numpy as np

from kid_readout.roach import r2baseband, analog
from kid_readout.measurement import acquire, basic
from kid_readout.equipment import hardware
from kid_readout.settings import ROACH2_IP, ROACH2_VALON
from kid_readout.equipment import starcryo_temps

acquire.show_settings()
logger = acquire.get_script_logger(__file__)

# Parameters
dummy_MHz = 170
f0_MHz = np.array([78.350, 116.164, 160.815, dummy_MHz])
attenuations = [35]
tone_sample_exponent = 21  # The linewidth may be as small as 250 Hz.
sweep_length_seconds = 0.1
stream_length_seconds = 30
num_sweep_tones = 128

# Hardware
conditioner = analog.Baseband()
shield = hardware.Thing(name='magnetic_shield_bucket', state={})
hw = hardware.Hardware(conditioner, shield)
ri = r2baseband.Roach2Baseband(roachip=ROACH2_IP, adc_valon=ROACH2_VALON)
ri.set_modulation_output('high')
ri.set_fft_gain(6)

# Calculate baseband frequencies
num_tone_samples = 2**tone_sample_exponent
f_resolution = ri.state.adc_sample_rate / num_tone_samples
offset_integers = np.arange(-num_sweep_tones / 2, num_sweep_tones / 2)
offset_frequencies_MHz = 1e-6 * f_resolution * offset_integers
sweep_frequencies_MHz = offset_frequencies_MHz[:, np.newaxis] + f0_MHz[np.newaxis, :]
logger.info("Frequency spacing is {:.3f} kHz".format(1e3 * (offset_frequencies_MHz[1] - offset_frequencies_MHz[0])))
logger.info("Sweep span is {:.3f} MHz".format(offset_frequencies_MHz.ptp()))

# Run
ncf = acquire.new_nc_file(suffix='magnetic_shield_compressor_off')
tic = time.time()
try:
    for attenuation in attenuations:
        ri.set_dac_attenuator(attenuation)
        logger.info("Set DAC attenuation to {:.1f} dB".format(attenuation))
        state = hw.state()
        state['temperature'] = {'package': starcryo_temps.get_temperatures_at(time.time())[0]}
        raw_input("Hit enter when temperatures have recovered.")
        logger.info("Recording {:.1f} s sweep at MHz center frequencies {}".format(sweep_length_seconds, f0_MHz))
        sweep_array = acquire.run_sweep(ri=ri, tone_banks=sweep_frequencies_MHz, length_seconds=sweep_length_seconds,
                                        num_tone_samples=num_tone_samples)
        fit_f0_MHz = np.array([1e-6 * sweep_array[n].resonator.f_0 for n in range(sweep_array.num_channels)])
        logger.info("Fit - initial [kHz]: {}".format(', '.join(['{:.3f}'.format(df0) for df0 in fit_f0_MHz - f0_MHz])))
        f_stream_MHz = ri.set_tone_freqs(np.array(fit_f0_MHz), nsamp=num_tone_samples)
        ri.select_bank(0)
        ri.select_fft_bins(np.arange(f_stream_MHz.size))
        logger.info("Turning compressor off to record stream.")
        raw_input("Hit enter to continue.")
        logger.info("Recording {:.1f} s streams at MHz frequencies {}".format(stream_length_seconds, f_stream_MHz))
        stream_array = ri.get_measurement(num_seconds=stream_length_seconds)
        logger.info("Turning compressor on.")
        sweep_stream_array = basic.SweepStreamArray(sweep_array=sweep_array, stream_array=stream_array,
                                                    description='attenuation {:.1f} dB'.format(attenuation))
        logger.debug("Writing SweepStreamArray.")
        ncf.write(sweep_stream_array)
        # Record an ADCSnap with the stream tones playing.
        logger.debug("Recording ADCSnap.")
        adc_snap = ri.get_adc_measurement()
        logger.debug("Writing ADCSnap.")
        ncf.write(adc_snap)
finally:
        ncf.close()
        print("Wrote {}".format(ncf.root_path))
        print("Elapsed time {:.0f} minutes.".format((time.time() - tic) / 60))
