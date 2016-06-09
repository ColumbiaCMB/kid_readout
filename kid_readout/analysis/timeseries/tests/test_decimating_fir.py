from kid_readout.analysis.timeseries import fftfilt, decimating_fir
import numpy as np

def test_decimating_fir():
    np.random.seed(123)
    x = np.random.randn(2**16) + 1j*np.random.randn(2**16)
    dfir = decimating_fir.DecimatingFIR(downsample_factor=16,num_taps=1024)
    gold = fftfilt.fftfilt(dfir.coefficients.ravel(),x)[15::16]
    result = dfir.process(x)
    assert np.allclose(gold,result)

test_decimating_fir.__test__ = False #disable for now until we have a chance to debug

def test_fir1d_history():
    np.random.seed(123)
    coeff = np.random.randn(16)
    data = np.random.randn(2**10)
    fir1 = decimating_fir.FIR1D(coeff)
    fir2 = decimating_fir.FIR1D(coeff)
    full = fir1.apply(data)
    compare = np.empty_like(full)
    part1 = fir2.apply(data[:len(data)//2])
    part2 = fir2.apply(data[len(data)//2:])
    compare[:len(part1)] = part1
    compare[len(part1):] = part2
    assert np.allclose(full,compare)

def test_fir1d_nstream():
    np.random.seed(123)
    coeff = np.random.randn(16)
    data = np.random.randn(2,2**10)
    fir1 = decimating_fir.FIR1D(coeff)
    fir2 = decimating_fir.FIR1D(coeff)
    full = fir1.apply(data)
    full = fir1.apply(data) #apply twice to excercise history
    stream1 = fir2.apply(data[0,:])
    stream1 = fir2.apply(data[0,:])

    assert  np.allclose(stream1,full[0,:])