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

    def measure_freq_error(self,ref_freq=991e6,offset=50e3):
        measured_freq,peak = self.read_freq_amplitude_and_peak(freq=ref_freq,nsamp=2**16,offset=offset,NFFT=8192)
        error = measured_freq-ref_freq
        ppm = 1e6*error/ref_freq
        print "ref: %f measured %f error %f ppm %f" % (ref_freq,measured_freq,error,ppm)
        return error
    def adjust_freq_correction(self,ref_freq=991e6,offset=50e3):
        self.rtl.freq_correction = 1
        measured_freq,peak = self.read_freq_amplitude_and_peak(freq=ref_freq,nsamp=2**16,offset=offset,NFFT=8192)
        error = ref_freq- measured_freq
        ppm = 1e6*error/ref_freq
        print "ref: %f measured %f error %f ppm %f" % (ref_freq,measured_freq,error,ppm)
        self.rtl.freq_correction = int(ppm)
        d = self.rtl.read_samples(2**16)
        measured_freq,peak = self.read_freq_amplitude_and_peak(freq=ref_freq,nsamp=2**16,offset=offset,NFFT=8192)
        error = measured_freq-ref_freq
        ppm = 1e6*error/ref_freq
        print "ref: %f measured %f error %f ppm %f" % (ref_freq,measured_freq,error,ppm)


    def read_freq_amplitude_and_peak(self,freq,nsamp=2048,offset=50e3,NFFT=1024):
        self.hittite.set_freq(freq)
        self.rtl.center_freq = freq+offset
        self.rtl.read_samples(2048)
        d = self.rtl.read_samples(nsamp)
        pxx,fr = plt.mlab.psd(d,NFFT=NFFT,Fs=256e3)
        x = pxx.max()
        loc = pxx.argmax()
        return fr[loc]+freq-offset,x

    def read_freq_amplitude(self,freq,nsamp=2048):
        peak_freq,peak = self.read_freq_amplitude_and_peak(freq=freq,nsamp=nsamp)
        print "peak offset", (peak_freq-freq)
#        fr = np.fft.fftfreq(nsamp)*256e3
#        bins = abs(fr+50e3) < 10e3
#        x = np.abs(np.fft.fft(d)).max()
        return peak

    def do_scan(self,freqs,level=-50.0,nsamp=2048):
        self.set_level_dbm(level)
        self.hittite.on()
        data = []
        for freq in freqs:
            data.append(self.read_freq_amplitude(freq,nsamp=nsamp))
        self.hittite.off()
        return freqs,np.array(data)

