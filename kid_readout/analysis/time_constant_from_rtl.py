import numpy as np
from matplotlib import pyplot as plt

import kid_readout.analysis.demodulate_rtl
reload(kid_readout.analysis.demodulate_rtl)
import kid_readout.analysis.fit_pulses
import kid_readout.analysis.fitter


def get_time_constant_from_file(filename,pulse_period=10e-3, debug=False):
    d = np.load(filename)
    demod = kid_readout.analysis.demodulate_rtl.demodulate(d['data'],debug=False)
    pulse_period_samples = int(d['sample_rate'][()]*pulse_period)
    folded = kid_readout.analysis.demodulate_rtl.fold(demod,pulse_period_samples)
    t = np.arange(pulse_period_samples)/d['sample_rate']
    y = np.abs(folded).mean(0)
    fit = kid_readout.analysis.fitter.Fitter(t,y, model=kid_readout.analysis.fit_pulses.fred_model,
                                             guess = kid_readout.analysis.fit_pulses.fred_guess)

    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(t,y)
        ax.plot(t,fit.model(x=t),'r',lw=2)
        peakt = t[np.abs(y-y.mean()).argmax()]
        ax.set_xlim(peakt-.1e-3,peakt+.5e-3)
    return fit.tau, fit