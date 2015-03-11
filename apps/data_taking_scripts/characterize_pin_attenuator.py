import numpy as np
import kid_readout.equipment.agilent_33220
import kid_readout.equipment.lockin_controller

import time

fg = kid_readout.equipment.agilent_33220.FunctionGenerator()
lockin = kid_readout.equipment.lockin_controller.lockinController()

print lockin.get_idn()

atten_turns = eval(raw_input("Enter mmw attenuator turns as a tuple: "))

biases = np.linspace(0.0,2.0,100)
zbd_voltages = []

for bias in biases:
    print bias
    fg.set_pulse(period=2e-3,width=1e-3,high_level=2.02,low_level=bias)
    fg.enable_output(True)
    time.sleep(2)
    x, y, r, theta = lockin.get_data()
    zbd_voltages.append(x)
zbd_voltages = np.array(zbd_voltages)

timestr = time.strftime('%Y-%m-%d_%H-%M-%S')
np.savez(('/home/data2/rtl/%s_pin_atten_character_%f_%f_turns.npz' % (timestr,atten_turns[0],atten_turns[1])),
         mmw_atten_turns=atten_turns, pin_bias_voltage=biases, zbd_voltage=zbd_voltages)

