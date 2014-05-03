import numpy as np
import scipy.signal
from kid_readout.utils.fftfilt import fftfilt

def low_pass_fir(data, num_taps=256, cutoff=1/256.,nyquist_freq=1.0,decimate_by=1):
    taps = scipy.signal.firwin(num_taps,cutoff/nyquist_freq)
    result = fftfilt(taps,data)[num_taps:]
    result = result[::decimate_by]
    return result

lpf = low_pass_fir

