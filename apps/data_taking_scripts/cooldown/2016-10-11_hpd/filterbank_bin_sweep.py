"""
Sweep one tone across its filter bank bin.
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
suffix = 'filterbank_bin'
fft_gain = 3
lo_MHz = 3000
baseband_MHz = 100
lo_round_to_MHz = 0.1
dac_attenuation = 0
tones_per_bin_exponent = 3
stream_length_blocks = 32
wait = 5

# Hardware
conditioner = analog.HeterodyneMarkII()
magnet = hardware.Thing(name='magnet_array', state={'orientation': 'up',
                                                    'distance_from_base_mm': 276})
hw = hardware.Hardware(conditioner, magnet)
ri = hardware_tools.r1_with_mk2()
ri.initialize(use_config=False)
ri.set_fft_gain(fft_gain)
# External is 1 and internal is 0
ri.adc_valon.set_ref_select(0)
ri.lo_valon.set_ref_select(1)

# Calculate tone bin integers
f_filterbank_MHz = ri.fs / ri.nfft
n_filterbank = int(np.round(baseband_MHz / f_filterbank_MHz))
tone_sample_exponent = int(np.log2(ri.nfft) + tones_per_bin_exponent)
center_integer = 2**tones_per_bin_exponent * n_filterbank
tone_integers = center_integer + np.arange(-4 * 2**tones_per_bin_exponent, 4 * 2**tones_per_bin_exponent + 1)

# Acquire
npd = acquire.new_npy_directory(suffix=suffix)
tic = time.time()
try:
    ri.set_lo(lomhz=lo_MHz, chan_spacing=lo_round_to_MHz)
    ri.set_dac_attenuator(dac_attenuation)
    for tone_integer in tone_integers:
        ri.set_tone_bins(bins=np.array([tone_integer]), nsamp=2**tone_sample_exponent)
        ri.fft_bins = np.atleast_2d(np.array([n_filterbank]))
        ri.select_bank(0)
        ri.select_fft_bins(np.array([0]))
        time.sleep(wait)
        npd.write(ri.get_measurement_blocks(num_blocks=stream_length_blocks, demod=False, state=hw.state()))
        npd.write(ri.get_adc_measurement())
finally:
    npd.close()
    print("Wrote {}".format(npd.root_path))
    print("Elapsed time {:.0f} minutes.".format((time.time() - tic) / 60))
