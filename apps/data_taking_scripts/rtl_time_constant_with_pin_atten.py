import numpy as np

import time
#import rtlsdr
import kid_readout.equipment.rtlkid
import kid_readout.equipment.agilent_33220
import kid_readout.equipment.lockin_controller

lockin = kid_readout.equipment.lockin_controller.lockinController()

print lockin.get_idn()

fg = kid_readout.equipment.agilent_33220.FunctionGenerator()


on_bias = 0.4
off_bias = 1.5
pulse_bias = 0.0
pulse_period = 2e-3
pulse_width = 2e-3-2e-6

hittite_power_level = 10.0

fg.set_pulse(period=pulse_period,width=pulse_width,high_level=on_bias,low_level=pulse_bias)
fg.enable_output(True)

#f_ref = 871.380e6
#f_ref = 870.436e6
f_ref=991.825e6
rtl = kid_readout.equipment.rtlkid.RtlKidReadout()
rtl.rtl.gain = 40.0
rtl.rtl.sample_rate = 256e3
rtl.hittite.set_power(hittite_power_level)
rtl.hittite.on()
rtl.adjust_freq_correction()
error = rtl.measure_freq_error()
if abs(error/1e9) > 5e-6:
    print "adjusting freq correction failed!"

atten_turns = eval(raw_input("Enter mmw attenuator turns as a tuple: "))
suffix='_pin_atten'
freq,data = rtl.do_scan(freqs=np.linspace(-8e5,3e5,500)+f_ref,level=hittite_power_level)
peak = freq[data.argmin()]#+1e3
print "peak at",peak
rtl.hittite.set_freq(peak)
rtl.rtl.center_freq = peak + 10e3
rtl.hittite.on()
time.sleep(2)

print "measuring pulses from on state"
d = rtl.rtl.read_samples(2**21)
start_time = time.time()
d = rtl.rtl.read_samples(2**21)
d = d[2048:]

print "measuring on state zbd voltage"
fg.set_pulse(period=pulse_period,width=pulse_period/2,high_level=off_bias,low_level=on_bias)
fg.enable_output(True)
time.sleep(2)
x,y,r,theta = lockin.get_data()

filename = '/home/data2/rtl/%s' % (time.strftime('%Y-%m-%d_%H-%M-%S'))
filename += suffix
np.savez(filename,data=d, time= time.time(), sample_rate=rtl.rtl.sample_rate, gain= rtl.rtl.gain,
         center_freq = rtl.rtl.center_freq,sweep_freq = freq, sweep_mag = data, start_time = start_time,
         hittite_power_level= hittite_power_level, mmw_atten_turns = atten_turns, pulse_period=pulse_period,
         pulse_width=pulse_width,high_level=on_bias,low_level=pulse_bias,zbd_voltage=x)
print "saved on measurement in ", filename

print "measuring pulses from off state"
fg.set_pulse(period=pulse_period,width=pulse_width,high_level=off_bias,low_level=pulse_bias)
fg.enable_output(True)
time.sleep(2)

d = rtl.rtl.read_samples(2**21)
start_time = time.time()
d = rtl.rtl.read_samples(2**21)
d = d[2048:]

print "measuring pulse state zbd voltage"
fg.set_pulse(period=pulse_period,width=pulse_period/2,high_level=off_bias,low_level=pulse_bias)
fg.enable_output(True)
time.sleep(2)
x,y,r,theta = lockin.get_data()

filename = '/home/data2/rtl/%s' % (time.strftime('%Y-%m-%d_%H-%M-%S'))
filename += suffix
np.savez(filename,data=d, time= time.time(), sample_rate=rtl.rtl.sample_rate, gain= rtl.rtl.gain,
         center_freq = rtl.rtl.center_freq,sweep_freq = freq, sweep_mag = data, start_time = start_time,
         hittite_power_level= hittite_power_level, mmw_atten_turns = atten_turns, pulse_period=pulse_period,
         pulse_width=pulse_width,high_level=off_bias,low_level=pulse_bias,zbd_voltage=x)
print "saved baseline measurement in ", filename