from time import sleep

import numpy as np

from kid_readout.equipment import hardware, starcryo_temps
from kid_readout.roach import analog, hardware_tools
from kid_readout.measurement import core, basic, acquire

acquire.show_settings()
acquire.show_git_status()
logger = acquire.get_script_logger(__file__)

# Parameters
wait = 4
length_seconds = 0.01
offset_frequency = np.linspace(10, 200, 4)
num_tone_samples = 2**15
dac_attenuation = 62  # The maximum
lo_frequency = 3000

# Hardware
temperature = starcryo_temps.Temperature()
conditioner = analog.HeterodyneMarkII()
hw = hardware.Hardware(temperature, conditioner)
ri = hardware_tools.r2h14_with_mk2(initialize=True, use_config=False)
ri.set_dac_attenuator(dac_attenuation)
ri.set_lo(lomhz=lo_frequency)
ri.set_tone_freqs(freqs=lo_frequency + offset_frequency, nsamp=num_tone_samples)
ri.select_fft_bins(np.arange(offset_frequency.size))
ri.set_modulation_output('high')
ri.iq_delay = -1
ri.adc_valon.set_ref_select(1)  # external
assert np.all(ri.adc_valon.get_phase_locks())
ri.lo_valon.set_ref_select(1)  # external
assert np.all(ri.lo_valon.get_phase_locks())

# Acquire
sweep = basic.SweepArray(core.IOList(), description="T_c measurement")
npd = acquire.new_npy_directory(suffix='Tc')
npd.write(sweep)
try:
    while True:
        state = hw.state()
        if state.temperature.package_ruox4550_temperature > 15:
            break
        sweep.stream_arrays.append(ri.get_measurement(num_seconds=length_seconds, state=state))
        sleep(wait)
finally:
    npd.close()
    print("Wrote {}".format(npd.root_path))
