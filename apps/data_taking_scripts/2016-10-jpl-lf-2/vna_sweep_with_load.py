import time
from kid_readout.interactive import *

from kid_readout.equipment import agilent_33220

fg = agilent_33220.FunctionGenerator()
fg.set_load_ohms(1000)
fg.set_dc_voltage(0)
fg.enable_output(False)

#setup = hardware.Hardware()

ri = Roach2Baseband()
ri.set_modulation_output('high')

for heater_voltage in np.sqrt(np.linspace(0,5**2,16))[:-1][::-1]:
    print heater_voltage
    fg.set_dc_voltage(heater_voltage)
    fg.enable_output(True)
    if heater_voltage < 4.8:
        print "waiting 60 minutes"
        time.sleep(3600)

    for dac_atten in [20]:
        ri.set_dac_atten(dac_atten)

        df = acquire.new_nc_file(suffix='vna_dac_atten_%.1f_dB_heater_%.3f_V' % (dac_atten,heater_voltage))
        swa = acquire.run_sweep(ri,np.linspace(110,170,128)[None,:]+np.arange(480,dtype='int')[:,None]*512./2.**18,
                                2**18,
                                verbose=True,length_seconds=.2,
                                )
        df.write(swa)
        df.close()
fg.enable_output(False)