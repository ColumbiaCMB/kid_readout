import numpy as np

import kid_readout.roach.calculate
from kid_readout.roach import demodulator

def test_wave_period_zero():
    assert(kid_readout.roach.calculate.get_offset_frequencies_period(np.zeros((1,))) == 1)