import numpy as np
from kid_readout.analysis.timeseries import fftfilt

def test_fftfilt_nd():
    np.random.seed(123)
    x = np.random.randn(2**16) + 1j*np.random.randn(2**16)
    b = np.random.randn(128)
    oned = fftfilt.fftfilt(b,x)
    nd = fftfilt.fftfilt_nd(b,x[:,None]).squeeze()
    assert np.allclose(oned,nd)

    x = x.reshape((2**12,2**4))
    oned = np.zeros_like(x)
    for k in range(16):
        oned[:,k] = fftfilt.fftfilt(b,x[:,k])
    nd = fftfilt.fftfilt_nd(b,x)
    assert np.allclose(oned,nd)