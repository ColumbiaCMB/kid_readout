import numpy as np
from matplotlib import pyplot as plt

import kid_readout.analysis.demodulate_rtl
from kid_readout.analysis.timeseries import fftfilt


reload(kid_readout.analysis.demodulate_rtl)
import kid_readout.analysis.fit_pulses
import kid_readout.analysis.fitter
import scipy.signal
import pandas as pd
import kid_readout.equipment.starcryo_temps
try:
    import kid_readout.equipment.hpd_temps
except ImportError:
    print "no temperatures available"
import kid_readout.analysis.resources.experiments

def process_time_constant_rtl_file(filename,pulse_period=10e-3,debug=False,filter_cutoff=200e3,fine_fold=False,
                                   cryostat=None):
    result = {}
    tau,fit,t,folded = get_time_constant_from_file(filename,pulse_period=pulse_period, debug=debug,filter_cutoff=filter_cutoff,
                                                   fine_fold=fine_fold)
    d = np.load(filename)
    for k in d.keys():
        if k not in ['data']:
            if len(d[k].shape) == 0:
                result[k] = [d[k][()]] # unpack zero length arrays
            else:
                result[k] = [d[k]]
    d.close()
    for k,v in fit.result.params.items():
        result[k] = [v.value]
        result[k+'_err'] = [v.stderr]
    result['folded_time'] = [t]
    result['folded_data'] = [folded]
    result['folded_model'] = [fit.model(x=t)]
    result['residuals'] = [fit.model(x=t)-np.abs(folded)]
    result['noise_rms'] = [folded[:len(folded)//5].std()]
    result['folded_peak_mag'] = [np.abs(folded).max()]
    if cryostat is None:
        cryostat = kid_readout.analysis.resources.experiments.default_cryostat
    if cryostat.lower() == 'hpd':
        primary_package_temperature, secondary_package_temperature, primary_load_temperature, secondary_load_temperature = kid_readout.equipment.hpd_temps.get_temperatures_at(result['time'][0])
    else:
        primary_package_temperature, secondary_package_temperature, primary_load_temperature, secondary_load_temperature = kid_readout.equipment.starcryo_temps.get_temperatures_at(result['time'][0])
    result['primary_package_temperature'] = [primary_package_temperature]
    result['secondary_package_temperature'] = [secondary_package_temperature]
    result['primary_load_temperature'] = [primary_load_temperature]
    result['secondary_load_temperature'] = [secondary_load_temperature]
    return pd.DataFrame(result,index=[0])

def get_time_constant_from_file(filename,pulse_period=10e-3, debug=False,filter_cutoff=100e3,
                                fine_fold=False):
    d = np.load(filename)
    sample_rate = d['sample_rate']
    demod = kid_readout.analysis.demodulate_rtl.demodulate(d['data'],debug=debug)
    lpf = scipy.signal.firwin(1024,filter_cutoff/sample_rate)
    filtered = fftfilt(lpf,demod)[512:]
    if 'pulse_period' in d:
        pulse_period = d['pulse_period'][()]
    pulse_period_samples = int(d['sample_rate'][()]*pulse_period)
    if fine_fold:
        folded = kid_readout.analysis.demodulate_rtl.fold(filtered,pulse_period_samples)
    else:
        folded = filtered[:(filtered.shape[0]//pulse_period_samples)*pulse_period_samples].reshape((-1,
                                                                                                  pulse_period_samples))
    folded = folded/folded.mean(1)[:,None]
    deprojected,deprojection_angle = kid_readout.analysis.demodulate_rtl.deproject_rtl(folded,
                                                                                       samples_around_peak=1000,
                                                                                       debug=debug)
    print deprojection_angle
    deprojected,deprojection_angle = kid_readout.analysis.demodulate_rtl.deproject_rtl(deprojected,
                                                                                       samples_around_peak=1000,
                                                                                       debug=debug)
    print deprojection_angle
    t = np.arange(pulse_period_samples)/d['sample_rate']
    y = np.abs(deprojected)
    fit = kid_readout.analysis.fitter.Fitter(t,y, model=kid_readout.analysis.fit_pulses.fred_model,
                                             guess = kid_readout.analysis.fit_pulses.fred_guess,
                                             )

    if debug:
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(231)
        ax.plot(t,y)
        ax.plot(t,fit.model(x=t),'r--',lw=2)
        peakt = t[np.abs(y-y.mean()).argmax()]
        ax.set_xlim(peakt-.5e-3,peakt+.5e-3)
        ax= fig.add_subplot(232)
        ax.plot(deprojected.real,deprojected.imag,'.')
        ax = fig.add_subplot(233)
        ax.plot(d['sweep_freq'],10*np.log10(d['sweep_mag']))
        ax.axvline(d['center_freq'])
        ax = fig.add_subplot(234)
        pxx,fr = plt.mlab.psd(demod-demod.mean(),Fs=d['sample_rate'],NFFT=2**18)
        ax.plot(fr,10*np.log10(pxx))
        pxx,fr = plt.mlab.psd(filtered-filtered.mean(),Fs=d['sample_rate'],NFFT=2**18)
        ax.plot(fr,10*np.log10(pxx))
        ax.set_ylim(-120,-40)
        ax.set_xscale('symlog')
        ax = fig.add_subplot(235)
        ax.plot(t,y)
    return fit.tau, fit,t, deprojected