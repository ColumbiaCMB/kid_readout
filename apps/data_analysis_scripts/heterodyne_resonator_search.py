__author__ = 'gjones'
import numpy as np
from matplotlib import pyplot as plt
import joblib
import itertools
import os
import glob

from kid_readout import *

def validate_resonator(res):
    if res.Q / np.abs(res.Q_e) < 0.01:
        print "failed shallow"
        return False
    if res.Q_e_real < 1000:
        print "failed low Qe"
        return False
    if res.Q_e_real > 1e6:
        print "failed high Qer"
        return False
    if np.abs(res.Q_e_imag) > 1e6:
        print "failed high Qei"
        return False
    if res.Q < 1000:
        print "failed low Q"
        return False
    if res.current_result.redchi > 1000:
        print "failed redchi"
        return False
    if np.abs(res.f_0 - res.frequency.min()) < res.frequency.ptp()*0.01:
        print "failed low range"
        return False
    if np.abs(res.f_0 - res.frequency.max()) < res.frequency.ptp()*0.01:
        print "failed high range"
        return False
    return True

def process_sweep(frequency,s21,errors):
    print "checking",frequency[0]/1e6,frequency[-1]/1e6
    validated = []
    for k in range(16):
        start = k*128/2
        stop = start+128
        try:
            res = lmfit_resonator.LinearResonatorWithCable(s21=s21[start:stop],frequency=frequency[start:stop],
                                                           errors=errors[start:stop])
            res.fit()
            res.fit()
            #print res.f_0,res.Q,res.Q_e,res.Q_i,res.current_result.redchi,res.f_0-res.frequency.min(),
            # validate_resonator(res)
            if validate_resonator(res):
                print res.f_0,res.Q,res.Q_e,res.Q_i,res.current_result.redchi,(res.Q / np.abs(res.Q_e))
                validated.append(res)
        except Exception, e:
            print frequency[0],k,e
    return [res.current_params for res in validated]

fns = glob.glob('/artemis/readout/2016-04-09_1[123]*_scan_lo_*_MHz_mmw_modulated_7_7_turns.nc')
fns.sort()
for fn in fns[1:]:
    ncname = os.path.split(fn)[1]
    print "loading...",fn
    ncf = NCFile(fn, cache_s21_raw=True)
    print "accessing sweep..."
    swa = ncf.SweepArray0


    print "extracting data..."
    data = []
    for k in range(32):
        swp = swa.sweep(k)
        data.append((swp.frequency, swp.s21_point, swp.s21_point_error))
    print "starting parallel jobs..."
    pp = joblib.Parallel(n_jobs=16,verbose=5)
    results = pp([joblib.delayed(process_sweep)(*args) for args in data])

    results = list(itertools.chain.from_iterable(results))
    print "saving results..."
    print joblib.dump(results,('/home/gjones/%s_resonators.pkl' % ncname),compress=True)

    print "plotting..."

    fig,ax = plt.subplots()
    for k in range(32):
        swp = swa.sweep(k)
        ax.plot(swp.frequency, 20 * np.log10(np.abs(swp.s21_point)))
        for params in results:
            f0 = params['f_0'].value
            if f0 > swp.frequency.min() and f0 < swp.frequency.max():
                idx = np.abs(swp.frequency - f0).argmin()
                ax.plot(swp.frequency[idx], 20 * np.log10(np.abs(swp.s21_point[idx])), 'o')


    fig.savefig(('/home/gjones/%s_resonators_plot.pdf' % ncname),bbox_inches='tight')

    ncf.close()