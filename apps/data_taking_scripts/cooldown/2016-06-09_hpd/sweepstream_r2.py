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
logging.getLogger('kid_readout.roach').setLevel(logging.DEBUG)
logger.addHandler(log.default_handler)
logger.setLevel(logging.INFO)

# Parameters
all_f0_MHz = np.array([2432, 3488, 3629, 3800])
f0_MHz = all_f0_MHz
attenuations = [24, 30, 36, 42, 48, 54, 60]
num_offsets = 100
span_MHz = 10
minimum_MHz = 10
round_to_MHz = 2
lo_MHz = round_to_MHz * np.round((f0_MHz - span_MHz / 2 - minimum_MHz) / round_to_MHz)
offsets_MHz = np.linspace(minimum_MHz, minimum_MHz + span_MHz, num_offsets)
num_tone_samples = 2**19
sweep_length_seconds = 0.1
stream_length_seconds = 60

# Hardware
conditioner = analog.HeterodyneMarkII()
magnet = hardware.Thing('canceling_magnet_quincunx',
                        {'orientation': 'up',
                         'ruler_to_base_mm': 55})
hw = hardware.Hardware(conditioner, magnet)
ri = hardware_tools.r2_with_mk2()
ri.set_modulation_output('high')

# Run
ncf = acquire.new_nc_file(suffix='sweep_stream')
tic = time.time()
try:
    for lo in lo_MHz:
        for attenuation in attenuations:
            ri.set_dac_attenuator(attenuation)
            logger.info("Set DAC attenuation to {:.1f} dB".format(attenuation))
            state = hw.state()
            state['temperature'] = {'package': temps.get_temperature_at(time.time())}
            tone_banks = (lo + offsets_MHz)[:, np.newaxis]  # Transform to shape (num_offsets, 1)
            ri.set_lo(lomhz=lo, chan_spacing=round_to_MHz)
            logger.info("Set LO to {:.3f} MHz".format(lo))
            sweep_array = acquire.run_sweep(ri, tone_banks=tone_banks, num_tone_samples=num_tone_samples,
                                            length_seconds=sweep_length_seconds)
            single_sweep = sweep_array[0]
            f0_MHz = 1e-6 * single_sweep.resonator.f_0
            ri.set_tone_freqs(np.array([f0_MHz]), nsamp=num_tone_samples)
            ri.select_fft_bins(np.array([0]))
            stream_array = ri.get_measurement(num_seconds=stream_length_seconds)
            single_stream = stream_array[0]
            sweep_stream = basic.SingleSweepStream(sweep=single_sweep, stream=single_stream, state=state,
                                                   description='f_0 = {:.1f}'.format(f0_MHz))
            ncf.write(sweep_stream)
finally:
        ncf.close()
        print("Wrote {}".format(ncf.root_path))
        print("Elapsed time {:.0f} minutes.".format((time.time() - tic) / 60))
