import numpy as np

def dB(x, as_power=True):
    if as_power:
        factor = 20
    else:
        factor = 10
    return factor*np.log10(np.abs(x))