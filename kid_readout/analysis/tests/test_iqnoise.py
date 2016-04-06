__author__ = 'gjones'
import numpy as np
from matplotlib import mlab
from kid_readout.analysis.iqnoise import full_spectral_helper

def test_full_spectral_helper():
    x = np.random.randn(2 ** 20)
    y = np.random.randn(2 ** 20)
    mlabPxx, fr = mlab.psd(x, NFFT=2 ** 16, Fs=512e6 / 2 ** 14)
    mlabPyy, fr = mlab.psd(y, NFFT=2 ** 16, Fs=512e6 / 2 ** 14)
    mlabPxy, fr = mlab.csd(x, y, NFFT=2 ** 16, Fs=512e6 / 2 ** 14)

    fullPxx, fullPyy, fullPxy, freqs, t = full_spectral_helper(x, y, NFFT=2 ** 16, Fs=512e6 / 2 ** 14)
    assert (np.allclose(mlabPxx, fullPxx.mean(1)))
    assert (np.allclose(mlabPyy, fullPyy.mean(1)))
    assert (np.allclose(mlabPxy, fullPxy.mean(1)))