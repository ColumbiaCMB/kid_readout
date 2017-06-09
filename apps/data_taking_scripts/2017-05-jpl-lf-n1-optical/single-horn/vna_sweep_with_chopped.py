from kid_readout.interactive import *

import time

import numpy as np
from equipment.custom import mmwave_source
from equipment.hittite import signal_generator
from equipment.srs import lockin
from xystage import stepper

from kid_readout.equipment import hardware
from kid_readout.measurement import mmw_source_sweep, core, acquire

logger.setLevel(logging.DEBUG)

lockin = lockin.Lockin(LOCKIN_SERIAL_PORT)
tic = time.time()
# lockin.sensitivity = 17
print lockin.identification
print lockin.identification

source = mmwave_source.MMWaveSource()
source.set_attenuator_turns(6.0,6.0)
source.multiplier_input = 'thermal'
source.waveguide_twist_angle = 0
source.ttl_modulation_source = 'roach'

hwp_motor = stepper.SimpleStepper(port='/dev/ttyACM2')

setup = hardware.Hardware(hwp_motor, source, lockin)

ri = Roach2Baseband()

ri.set_modulation_output('low')

#setup = hardware.Hardware()

ri = Roach2Baseband()

#turn on source
ri.set_modulation_output(7)
raw_input('set attenuator knobs to 3 turns & check lock-in range')

for dac_atten in [10]:
    ri.set_dac_atten(dac_atten)

    df = acquire.new_nc_file(suffix='vna_dac_atten_%.1f_dB_1_turns_chopped' % dac_atten)
    swa = acquire.run_sweep(ri,np.linspace(100,180,64)[None,:]+np.arange(650,dtype='int')[:,None]*512./2.**18,
                            2**18,
                            verbose=True,length_seconds=.1,
                            )
    df.write(swa)
    df.close()


#for example
#170-230 MHz band, steps are (230-170)/128
#then sampling 480 times between each of these steps by stepping an additional 2**18