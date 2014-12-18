__author__ = 'gjones'
import numpy as np
from matplotlib import pyplot as plt
from kid_readout.equipment.hittite_controller import hittiteController

import rtlsdr

class RtlKidReadout(object):
    def __init__(self,hittite_addr='192.168.1.70'):
        self.hittite = hittiteController(addr=hittite_addr)
        self.rtl = rtlsdr.RtlSdr()
        self.rtl.sample_rate = 256e3
        self.rtl.gain = 30.0
    def set_level_dbm(self,level):
        self.hittite.set_power(level)

    def read_freq(self,freq,nsamp=2048):
        self.hittite.set_freq(freq)
        self.rtl.center_freq = freq+50e3
        self.rtl.read_samples(2048)
        d = self.rtl.read_samples(nsamp)
        pxx,fr = plt.mlab.psd(d,NFFT=1024,Fs=256e3)
        loc = abs(fr+50e3).argmin()
        x = pxx.max()
        loc = pxx.argmax()
        print "peak at", (fr[loc])
#        fr = np.fft.fftfreq(nsamp)*256e3
#        bins = abs(fr+50e3) < 10e3
#        x = np.abs(np.fft.fft(d)).max()
        return x

    def do_scan(self,freqs,level=-50.0,nsamp=2048):
        self.set_level_dbm(level)
        self.hittite.on()
        data = []
        for freq in freqs:
            data.append(self.read_freq(freq,nsamp=nsamp))
        self.hittite.off()
        return freqs,np.array(data)

