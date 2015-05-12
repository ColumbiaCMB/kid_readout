"""
Misc utils related to the ROACH hardware
"""

import numpy as np
import scipy.signal


def ntone_power_correction(ntones):
    """
    Power correction in dB relative to a single tone
    
    *ntones* : number of tones simultaneously output
    """
    if ntones < 10:
        return 20 * np.log10(ntones)
    else:
        return 10 * np.log10(ntones) + 10


def get_delay_estimate_for_boffile(boffile):
    if boffile in boffile_delay_estimates:
        return boffile_delay_estimates[boffile]
    print "No delay estimate found for %s, using 0 seconds" % boffile
    return 0

def get_delay_estimate_for_nfft(nfft):
    if nfft in nfft_delay_estimates:
        return nfft_delay_estimates[nfft]
    print "No delay estimate found for nfft=%d, using 0 seconds" % nfft
    return 0



boffile_delay_estimates = {  # 'bb2xpfb10mcr11_2014_Jan_20_1049.bof',
                             #'bb2xpfb10mcr8_2013_Nov_18_0706.bof',
                             'bb2xpfb11mcr7_2013_Nov_04_1309.bof' : -7.29e-6,
                             'bb2xpfb11mcr8_2013_Nov_04_2151.bof' : -7.29e-6,
                             'bb2xpfb11mcr11_2014_Feb_01_1106.bof' : -7.29e-6,
                             'bb2xpfb11mcr12_2014_Feb_26_1028.bof' : -7.29e-6,
                             'bb2xpfb11mcr13_2014_Mar_09_1719.bof' : -7.29e-6,
                             'bb2xpfb11mcr14_2014_Aug_09_1203.bof' : -7.29e-6,
                             #'bb2xpfb12mcr5_2013_Oct_29_1658.bof',
                             #'bb2xpfb12mcr6_2013_Oct_30_1519.bof',
                             #'bb2xpfb12mcr7_2013_Oct_30_1842.bof',
                             'bb2xpfb14mcr1_2013_Jul_24_1138.bof' : -63.3e-6,
                             'bb2xpfb14mcr5_2013_Jul_31_1021.bof' : -63.3e-6,
                             'bb2xpfb14mcr5_2013_Jul_31_1301.bof' : -63.3e-6,
                             'bb2xpfb14mcr7_2013_Oct_31_1332.bof' : -63.3e-6,
                             'bb2xpfb14mcr9_2013_Dec_05_1226.bof' : -63.3e-6,
                             'bb2xpfb14mcr10_2013_Dec_25_2005.bof' : -63.3e-6,
                             'bb2xpfb14mcr11_2014_Jan_17_1721.bof' : -63.3e-6,
                             'bb2xpfb14mcr16_2014_Sep_23_1157.bof' : -63.3e-6,
                             'bb2xpfb14mcr17_2014_Oct_12_1745.bof' : -63.3e-6,
                             'bb2xpfb14mcr17b_2015_Apr_21_1159.bof' : -63.3e-6,
                             #'bb2xpfb15mcr1_2013_Jul_24_1504.bof',
                             #'bb2xpfb15mcr1_2013_Jul_27_2342.bof',
                             #'bb2xpfb15mcr4_2013_Jul_31_0004.bof'
                              }

nfft_delay_estimates = {2**11 : -7.29e-6,
                        2**14 : -63.3e-6}


def compute_window(npfb=2 ** 15, taps=2, wfunc=scipy.signal.flattop):
    wv = wfunc(npfb * taps)
    sc = np.sinc(np.arange(npfb * taps) / float(npfb) - taps / 2.0)
    coeff = wv * sc
    mag = np.abs(np.fft.fft(coeff, npfb * taps * 2 ** 5)[:2 ** 7])
    mag = mag / mag.max()
    return mag