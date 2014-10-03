import numpy as np

import time
#import rtlsdr
import kid_readout.equipment.rtlkid

#rtl = rtlsdr.RtlSdr()
#rtl.gain = 40.2
#rtl.center_freq = 870.840e6
#rtl.sample_rate = 1024e3

f_ref = 871.380e6
rtl = kid_readout.equipment.rtlkid.RtlKidReadout()
rtl.rtl.gain = 40.0
rtl.rtl.sample_rate = 1024e3
while True:
    start_time = time.time()
    freq,data = rtl.do_scan(freqs=np.linspace(-8e5,3e5,500)+f_ref,level=2.0)
    peak = freq[data.argmin()]
    print "peak at",peak
    rtl.hittite.set_freq(peak)
    rtl.rtl.center_freq = peak + 10e3
    rtl.hittite.on()
    d = rtl.rtl.read_samples(2**21)
    d = rtl.rtl.read_samples(2**21)
    filename = '/home/data2/rtl/%s' % (time.strftime('%Y-%m-%d_%H-%M-%S'))
    np.savez(filename,data=d, time= time.time(), sample_rate=rtl.rtl.sample_rate, gain= rtl.rtl.gain,
             center_freq = rtl.rtl.center_freq,sweep_freq = freq, sweep_mag = data, start_time = start_time)
    print "saved in ", filename

    time.sleep(120.0)