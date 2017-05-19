import numpy as np

import time
#import rtlsdr
import kid_readout.equipment.rtlkid
import kid_readout.equipment.agilent_33220

fg = kid_readout.equipment.agilent_33220.FunctionGenerator()


#f_ref = 871.380e6
#f_ref = 870.436e6
f_ref=991.825e6
rtl = kid_readout.equipment.rtlkid.RtlKidReadout()
rtl.rtl.gain = 40.0
rtl.rtl.sample_rate = 256e3
rtl.hittite.set_power(10.0)
rtl.hittite.on()
rtl.adjust_freq_correction()
error = rtl.measure_freq_error()
if abs(error/1e9) > 5e-6:
    print "adjusting freq correction failed!"
biases = np.linspace(0.2,1.0,20)
atten_turns = eval(raw_input("Enter mmw attenuator turns as a tuple: "))
suffix='_pin_atten'
hittite_power_level = 0.0
freq,data = rtl.do_scan(freqs=np.linspace(-8e5,3e5,500)+f_ref,level=hittite_power_level)
peak = freq[data.argmin()]#+1e3
print "peak at",peak
rtl.hittite.set_freq(peak)
rtl.rtl.center_freq = peak + 10e3
rtl.hittite.on()
time.sleep(2)

pulse_period = 2e-3
pulse_width = 2e-3-2e-6

for bias in biases:
    fg.set_pulse(period=pulse_period,width=pulse_width,high_level=bias,low_level=bias-0.2)
    time.sleep(0.1)
    fg.enable_output(True)
    start_time = time.time()
    d = rtl.rtl.read_samples(2**21)
    start_time = time.time()
    d = rtl.rtl.read_samples(2**21)
    d = d[2048:]
    filename = '/home/data2/rtl/%s' % (time.strftime('%Y-%m-%d_%H-%M-%S'))
    filename += suffix
    np.savez(filename,data=d, time= time.time(), sample_rate=rtl.rtl.sample_rate, gain= rtl.rtl.gain,
             center_freq = rtl.rtl.center_freq,sweep_freq = freq, sweep_mag = data, start_time = start_time,
             hittite_power_level= hittite_power_level, mmw_atten_turns = atten_turns, pulse_period=pulse_period,
             pulse_width=pulse_width,high_level=bias,low_level=bias-0.1)
    print "saved in ", filename
