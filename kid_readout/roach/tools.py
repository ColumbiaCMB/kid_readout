"""
Misc utils related to the ROACH hardware
"""

import numpy as np
import scipy.signal
from matplotlib import pyplot as plt

from kid_readout.utils.misc import dB
from kid_readout.measurement import acquire
import logging

logger = logging.getLogger(__name__)

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

def get_delay_estimate_for_nfft(nfft, heterodyne=False):
    try:
        if not heterodyne:
            return baseband_nfft_delay_estimates[nfft]
        else:
            return heterodyne_nfft_delay_estimates[nfft]
    except KeyError:
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
                             'iq2xpfb14mcr6_2015_May_11_2241.bof' : -31.3e-6,
                             'iq2xpfb14mcr7_2015_Nov_25_0907.bof' : -31.3e-6,
                             'r2bb2xpfb14mcr21_2015_Oct_08_1422.bof' : -200.1e-6,
                             'r2bb2xpfb14mcr23_2015_Oct_27_1357.bof' : -63.68e-6,

                              }

baseband_nfft_delay_estimates = {2**11 : -7.29e-6,
                        2**13 : -31.3e-6,
                        2**14 : -63.3e-6}

heterodyne_nfft_delay_estimates = {
                        2**14 : -31.3e-6,
                        2**15 : -63.3e-6}



def compute_window(npfb=2 ** 15, taps=2, wfunc=scipy.signal.flattop):
    wv = wfunc(npfb * taps)
    sc = np.sinc(np.arange(npfb * taps) / float(npfb) - taps / 2.0)
    coeff = wv * sc
    mag = np.abs(np.fft.fft(coeff, npfb * taps * 2 ** 5)[:2 ** 7])
    mag = mag / mag.max()
    return mag


def calc_wavenorm(ntones, nsamp, baseband=False):
    # these are the max of the max(abs(waveform)) for various ntones
    # found empirically for heterodyne.  Need to divide by 2 for baseband
    numtones = 2**np.arange(9)
    maxvals = np.array([1., 2., 4., 8., 14., 24., 36., 52., 70.])
    lookupdict = dict(zip(numtones, maxvals))
    if ntones not in numtones:
        z = np.interp(np.log2(ntones), np.log2(numtones), np.log2(maxvals))
        wavenorm = 2**z / nsamp
    else:
        wavenorm = lookupdict[ntones] / nsamp
    if baseband:
        wavenorm *= 2.
    return wavenorm

def find_best_iq_delay_adc(ri,iq_delay_range=np.arange(-4,5), make_plot=False):
    tone_baseband_frequencies = ri.tone_baseband_frequencies
    total_rejections = []
    for iq_delay in iq_delay_range:
        ri.iq_delay = iq_delay
        ri.set_tone_baseband_freqs(tone_baseband_frequencies,ri.tone_nsamp)
#        time.sleep(0.1)
        x,y = ri.get_raw_adc()
        pxx,fr = plt.mlab.psd(x+1j*y,Fs=ri.fs,NFFT=1024)
        rejections = []
        for idx,tone in enumerate(tone_baseband_frequencies[0,:]):
            signal = dB(pxx[np.argmin(np.abs(fr - tone))], as_power=False)
            image = dB(pxx[np.argmin(np.abs(fr + tone))], as_power=False)
            rejection = signal-image
            logger.debug("\nAt iq_delay: %d, tone: %.1f MHz, signal: %.1f dB, image %.1f dB, rejection %.1f dB" %
                       (iq_delay, tone, signal, image, rejection))
            rejections.append(rejection)
        total_rejections.append(np.median(rejections))
    best = np.argmax(total_rejections)
    logger.info("Best delay is %d with median rejection %.1f dB" % (iq_delay_range[best],total_rejections[best]))
    if make_plot:
        plt.plot(iq_delay_range,total_rejections)
    return iq_delay_range[best], total_rejections[best]

def measure_hardware_delay(ri,frequencies=np.arange(1,9)*24,num_tone_samples=2**16,num_points=16,make_plots=False,
                           verbose=False):
    offsets = np.arange(-num_points//2,num_points//2+1)*ri.fs/float(num_tone_samples)
    if ri.is_roach2:
        sa = acquire.run_sweep(ri,ri.lo_frequency+frequencies[None,:]+offsets[:,None],
                               num_tone_samples=num_tone_samples,verbose=verbose)
    else:
        acquire.load_heterodyne_sweep_tones(ri,ri.lo_frequency+frequencies[None,:]+offsets[:,None],
                                            num_tone_samples=num_tone_samples)
        sa = acquire.run_loaded_sweep(ri,verbose=verbose)

    print np.median(np.abs(sa.s21_point))
    local_delays = []
    for k in range(frequencies.shape[0]):
        swp = sa.sweep(k)
        deltaf = swp.frequency-swp.frequency.min()
        phase = np.unwrap(np.angle(swp.s21_point))
        rad_per_Hz,offset = np.polyfit(deltaf,phase,1)
        local_delays.append(rad_per_Hz/(2*np.pi))
        if make_plots:
            plt.plot(deltaf,phase-offset,'.')
            plt.plot(deltaf,rad_per_Hz*deltaf)
            plt.xlabel('Offset Frequency (Hz)')
            plt.ylabel('Phase (rad)')
    local_delays = np.array(local_delays)
    if make_plots:
        plt.figure()
        plt.plot(frequencies,local_delays*1e9,'o')
        plt.axhline(np.median(local_delays*1e9),linestyle='--',color='r')
        plt.xlabel('Measurement Frequency (MHz)')
        plt.ylabel('Delay (ns)')
    logger.debug("median local delay: %.1f ns" % (np.median(local_delays)*1e9))
    frequency = sa.frequency
    deltaf = frequency - frequency.min()
    phase = np.unwrap(np.angle(sa.s21_point*np.exp(-1j*np.median(local_delays)*2*np.pi*deltaf)))
    rad_per_Hz,offset = np.polyfit(deltaf,phase,1)
    if make_plots:
        plt.figure()
        plt.plot(frequency/1e6,phase,'.')
        plt.plot(frequency/1e6,offset+rad_per_Hz*deltaf)
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Phase (rad)')
    total_delay = np.median(local_delays) + rad_per_Hz/(2*np.pi)
    logger.debug("residual delay %.1f ns global delay = %.1f ns" % (1e9*rad_per_Hz/(2*np.pi), 1e9*total_delay))
    return total_delay
