from kid_readout.interactive import *
import valon_synth
from equipment.srs import lockin
from xystage import stepper
import resonances

r2 = hardware_tools.r2h14_with_mk2(initialize=True, use_config=False)
#r2 = hardware_tools.r2h11_with_mk2(initialize=True, use_config=False)
r2.set_modulation_output('high')

lov = r2.lo_valon
adcv = r2.adc_valon
sa = valon_synth.SYNTH_A
sb = valon_synth.SYNTH_B

lock = lockin.SR830(serial_device=LOCKIN_SERIAL_PORT)
print(lock.identification)

hwp = stepper.SimpleStepper(port=CRYOGENIC_HWP_MOTOR_SERIAL_PORT)


def set_a_frequency():
    r2.set_dac_attenuator(0)
    r2.set_lo(3000)
    r2.set_tone_baseband_freqs(np.array([100]), nsamp=2**14)