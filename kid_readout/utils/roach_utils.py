"""
Misc utils related to the ROACH hardware
"""

import numpy as np

def ntone_power_correction(ntones):
    """
    Power correction in dB relative to a single tone
    
    *ntones* : number of tones simultaneously output
    """
    if ntones < 10:
        return 20*np.log10(ntones)
    else:
        return 10*np.log10(ntones)+10    
