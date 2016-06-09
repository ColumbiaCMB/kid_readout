import numpy as np
from kid_readout.roach import demodulator

def test_wave_period_zero():
    assert(demodulator.get_foffs_period(np.zeros((1,)))==1)