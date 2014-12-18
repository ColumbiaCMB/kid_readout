import numpy as np
from matplotlib import pyplot as plt

import kid_readout.analysis.demodulate_rtl
reload(kid_readout.analysis.demodulate_rtl)
import kid_readout.analysis.fit_pulses
import kid_readout.analysis.fitter
import copy

def get_time_constant_from_file(filename,pulse_period=10e-3, debug=False):
    d = np.load(filename)
    demod = kid_readout.analysis.demodulate_rtl.demodulate(d['data'],debug=False)
    pulse_period_samples = int(d['sample_rate'][()]*pulse_period)
    folded = kid_readout.analysis.demodulate_rtl.fold(demod,pulse_period_samples)
    deprojected,deprojection_angle = kid_readout.analysis.demodulate_rtl.deproject_rtl(folded,samples_around_peak=100)
    t = np.arange(pulse_period_samples)/d['sample_rate']
    y = np.abs(deprojected)
    fit = kid_readout.analysis.fitter.Fitter(t,y, model=kid_readout.analysis.fit_pulses.fred_model,
                                             guess = kid_readout.analysis.fit_pulses.fred_guess,
                                             )

    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(221)
        ax.plot(t,y)
        ax.plot(t,fit.model(x=t),'r--',lw=2)
        peakt = t[np.abs(y-y.mean()).argmax()]
        ax.set_xlim(peakt-.1e-3,peakt+.5e-3)
        ax= fig.add_subplot(222)
        ax.plot(deprojected.real,deprojected.imag,'.')
        ax = fig.add_subplot(223)
        ax.plot(d['sweep_freq'],10*np.log10(d['sweep_mag']))
        ax.axvline(d['center_freq'])
        ax = fig.add_subplot(224)
        pxx,fr = plt.mlab.psd(demod,Fs=d['sample_rate'],NFFT=2**18)
        ax.plot(fr,10*np.log10(pxx))
        ax.set_xscale('symlog')
    return fit.tau, fit