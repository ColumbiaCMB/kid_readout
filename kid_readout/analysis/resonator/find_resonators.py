__author__ = 'gjones'

import numpy as np
from matplotlib import pyplot as plt
import logging

from kid_readout.analysis import detect_peaks
from kid_readout.analysis.resonator import lmfit_resonator

logger = logging.getLogger(__name__)

def find_resonators(frequency, s21, s21_error, frequency_span=1e6, detect_peaks_threshold=3, detect_peaks_kwargs=dict(),
                    make_plot=False,annotate=False):
    unique_mask = np.flatnonzero(np.diff(frequency)!=0)
    frequency = frequency[unique_mask]
    s21 = s21[unique_mask]
    s21_error = s21_error[unique_mask]
    mag_s21 = np.abs(s21)
    peak_indexes = detect_peaks.detect_peaks(-mag_s21,threshold=detect_peaks_threshold,**detect_peaks_kwargs)
    peak_indexes2 = detect_peaks.detect_peaks(-mag_s21[::2],threshold=detect_peaks_threshold,**detect_peaks_kwargs)
    peak_indexes = np.sort(np.array(list(set(peak_indexes) | set(peak_indexes2*2))))
    logger.debug("Found %d peaks",peak_indexes.shape[0])
    resonators = []
    for peak_index in peak_indexes:
        peak_f = frequency[peak_index]
        sub_mask = np.abs(peak_f-frequency) < frequency_span
        res = lmfit_resonator.LinearResonatorWithCable(frequency[sub_mask], s21[sub_mask],s21_error[sub_mask])
        if res.current_result.redchi > 1000:
            res2 = lmfit_resonator.CollidingLinearResonatorsWithCable(frequency[sub_mask], s21[sub_mask],s21_error[sub_mask])
            if res2.current_result.redchi < 1000:
                logger.info("Found possible resonator collision at %.1f  %.1f" % (res2.bg_f_0,res2.fg_f_0))
#
#        else:
        resonators.append(res)
    if make_plot:
        plot_results(frequency,s21,resonators,peak_indexes,frequency_span,annotate=annotate)
    return resonators

def plot_results(frequency, s21, resonators, peak_indexes=[], frequency_span=250e3,annotate=True):
    plt.plot(frequency/1e6,20*np.log10(np.abs(s21)))
    for peak_index in peak_indexes:
        peak_f = frequency[peak_index]
        plt.plot(peak_f/1e6,20*np.log10(np.abs(s21[peak_index])),'r+',mew=2,markersize=10)

    for res in resonators:
        if res.Q_i < 2000:
            color='gray'
        elif res.Q_i < 10e3:
            color='k'
        elif res.Q_i < 100e3:
            color='m'
        else:
            color='r'
        dB_s21_at_f_0 = 20*np.log10(np.abs(res.model.eval(params=res.current_params,f=res.f_0)))
        sub_mask = np.abs(res.f_0-frequency) < frequency_span
        plt.plot(frequency[sub_mask]/1e6,20*np.log10(np.abs(res.model.eval(params=res.current_params,f=frequency[sub_mask]))))
        plt.plot(res.f_0/1e6,dB_s21_at_f_0,'o')
        if annotate:
            plt.annotate(xy=(res.f_0/1e6, dB_s21_at_f_0), s=('Q: %.1f\nQc: %.1f\nQi: %.1f\nchi2: %.1f' % (res.Q,1/np.real(1/res.Q_e),res.Q_i,res.current_result.redchi)),
            size=8,textcoords='offset points',xytext=(10,-10),color=color)

def remove_duplicates(resonators,tolerance=50e3):
    clean = []
    f0s = []
    for p in resonators:
        f0 = p.f_0
#        f0 = p['f_0'].value
        if len(clean) == 0:
            clean.append(p)
            f0s.append(f0)
            continue
        distance = np.abs(f0-np.array(f0s))
        if not np.any(distance < tolerance):
            clean.append(p)
            f0s.append(f0)
        else:
            print "found duplicate of ", f0, "distance", distance.min()

    return clean

def validate_resonator(res):
    print res.f_0,
    Qe = 1/np.real(1/res.Q_e)
    if res.Q / np.abs(Qe) < 0.01:
        print "failed shallow"
        return False
    if Qe < 1000:
        print "failed low Qe"
        return False
    if res.Q_e_real > 1e6:
        print "failed high Qer"
        return False
    if np.abs(res.Q_e_imag) > 1e6:
        print "failed high Qei"
        return False
    if Qe > 1e6:
        print "failed high Qe"
    if res.Q < 1000:
        print "failed low Q"
        return False
#    if res.current_result.redchi > 1000:
#        print "failed redchi"
#        return False
    if np.abs(res.f_0 - res.frequency.min()) < res.frequency.ptp()*0.01:
        print "failed low range"
        return False
    if np.abs(res.f_0 - res.frequency.max()) < res.frequency.ptp()*0.01:
        print "failed high range"
        return False
    if np.abs(np.mod(res.f_0,10e6)) < 10e3  or np.abs(np.mod(res.f_0,10e6)-10e6) < 10e3:
        print "failed multiple of 10 MHz"
        return False
    return True