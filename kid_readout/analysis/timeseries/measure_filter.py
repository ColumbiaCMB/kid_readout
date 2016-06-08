import numpy as np

def measure_frequency_response(filter_function, freqs=np.logspace(-3,0,1024), input_length=2**18):
    """
    Empericaly measure frequency response of a filtering function using sine waves.

    Parameters
    ----------
    filter_function : callable
    freqs : array of floats
        normalized frequencies at which to measure response (0 = DC, 1.0 = Nyquist)
    input_length : int
        number of samples in each sinewave

    Returns
    -------
    freqs : array of floats
        the frequency array (for convenience)
    mags : array of floats
        the measured output power at each frequency
    """
    mags = []
    tt = np.arange(input_length)
    for k,freq in enumerate(freqs):
        x = np.sin(np.pi*tt*freq+np.random.uniform(0,2*np.pi)).astype('float32')
        result = filter_function(x)
        nout = result.shape[0]
        result = result[nout//4:3*nout//4]
        mags.append(np.sum(np.abs(result)**2))
    return freqs, np.array(mags)


