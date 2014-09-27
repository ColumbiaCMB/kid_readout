import os
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
matplotlib.rcParams['font.size'] = 16.0
import lmfit

from kid_readout.analysis import kid_eqns

default_settings = dict(fractional_f_0_err_limit = 1e-6,
                        fractional_Q_err_limit = 0.06,
                        valid_Q_range = (5000,1e6),
                        max_package_temp_deviation = None,
                        )
settings = dict(valid_load_temp_range = (0,8.0),
                f_0_max_temp_limit = 5.0,
                )

all_settings = dict()
all_settings.update(default_settings)
all_settings.update(settings)

from kid_readout.analysis import kid_eqns
reload(kid_eqns)
import itertools
def plot_mattis_bardeen(data,axs=None,min_fit_temp=0,error_factor=100.0):
    if axs is None:
        fig,(ax1,ax2,ax3) = plt.subplots(ncols=3,figsize=(20,6),squeeze=True)
    else:
        ax1,ax2,ax3 = axs
        
    temps = np.array(data.sweep_primary_package_temperature)
    Qis = np.array(data.Q_i)
#    Qis = 1/(1/Qis-.99/Qis.max())
    print Qis
    fractional_freq = np.array(data.fractional_delta_f_0)
#    fractional_freq = data.f_0/data.f_0.iat[temps.argmin()]-1
    Qi_err = np.array(data.Q_err)
    Qi_err[Qi_err <= 0] = np.median(Qi_err)
    Qi_err = Qi_err/error_factor
    print Qi_err/Qis
    f0_err = np.array(data.f_0_err/data.f_0)
    f0_err[f0_err <= 0] = np.median(f0_err)
    print f0_err/np.abs(fractional_freq)
    ax1.plot(temps,fractional_freq,'o',label='data')
    ax2.plot(temps,fractional_freq,'o')    
    #ax3.errorbar(temps,1/Qis,yerr=100*Qi_err/Qis**2 ,linestyle='none',marker='.')
    ax3.plot(temps,1/Qis,'o')
    #ax4.plot(temps,1/Qis,'o')
    
    T = np.linspace(0.1,0.38,1000)
    
    temp_mask = temps>min_fit_temp
    fit_temps = temps[temp_mask]
    fit_fractional_freq = fractional_freq[temp_mask]
    fit_f0_err = f0_err[temp_mask]
    fit_Qi_err = Qi_err[temp_mask]
    fit_Qis = Qis[temp_mask]
#    fit_Qi_err = fit_Qis*error_factor
    ax1.plot(fit_temps,fit_fractional_freq,'ro',label='fit data')
    ax2.plot(fit_temps,fit_fractional_freq,'ro')    
    ax3.plot(fit_temps,1/fit_Qis,'ro')
    #ax4.plot(fit_temps,1/fit_Qis,'ro')    
    
    optional_params = ['delta_0','delta_loss']
    param_sets = [[]]
    for k in range(len(optional_params)):
        param_sets.extend(itertools.combinations(optional_params,k+1))
    colors = plt.cm.spectral(np.linspace(0.1,.9,len(param_sets)))
    for color,param_set in zip(colors,param_sets):
        km = kid_eqns.DarkKIDModelFractional()
        for name in optional_params:
            km.params[name].vary = False
        for name in param_set:
            km.params[name].vary = True
        km.fit_f0(fit_temps,fit_fractional_freq,fit_f0_err)
        print "\nfitting only f0 and varying", param_set
        lmfit.report_fit(km.params)
        print km.result.message
        ax1.plot(T,km.total_fres(T),color=color,label=('f0 %s' % ','.join(param_set)))
        ax2.plot(T,km.total_fres(T),color=color,label=('f0 %s' % ','.join(param_set)))
        ax3.plot(T,1/km.total_Qi(T),color=color,label=('f0 %s' % ','.join(param_set)))
        #ax4.plot(T,1/km.total_Qi(T),color=color,label=('f0 %s' % ','.join(param_set)))
        
        if 0:
            km = kid_eqns.DarkKIDModelFractional()
            for name in optional_params:
                km.params[name].vary = False
            for name in param_set:
                km.params[name].vary = True
            km.fit_qi(fit_temps,fit_Qis,fit_Qi_err)
            print "\nfitting only qi and varying", param_set
            lmfit.report_fit(km.params)
            print km.result.message
            ax1.plot(T,km.total_fres(T),':',color=color,lw=4,label=('qi %s' % ','.join(param_set)))
            ax2.plot(T,km.total_fres(T),':',color=color,lw=4,label=('qi %s' % ','.join(param_set)))
            ax3.plot(T,1/km.total_Qi(T),':',color=color,lw=4,label=('qi %s' % ','.join(param_set)))
            #ax4.plot(T,1/km.total_Qi(T),':',color=color,lw=4,label=('qi %s' % ','.join(param_set)))
                
        km = kid_eqns.DarkKIDModelFractional()
        for name in optional_params:
            km.params[name].vary = False
        for name in param_set:
            km.params[name].vary = True
        km.fit_f0_qi(fit_temps,fit_fractional_freq,fit_Qis,f0_err=fit_f0_err,Qi_err=fit_Qi_err)
        print "\nitting f0 and Qi and varying", param_set
        lmfit.report_fit(km.params)
        print km.result.message
        ax1.plot(T,km.total_fres(T),'--',color=color,lw=2.5,label=('f0qi %s' % ','.join(param_set)))
        ax2.plot(T,km.total_fres(T),'--',color=color,lw=2.5,label=('f0qi %s' % ','.join(param_set)))
        ax3.plot(T,1/km.total_Qi(T),'--',color=color,lw=2.5,label=('%s' % ','.join(param_set)))
        #ax4.plot(T,1/km.total_Qi(T),'--',color=color,lw=1.5,label=('%s' % ','.join(param_set)))
    
    ax1.set_ylim(fractional_freq.min()*1.1,0)
    ax2.set_ylim(-2e-5,1e-5)
    #ax3.set_ylim(1/Qis.max(),1/Qis.min())
    ax3.set_ylim(0,2e-5)
    #ax5.set_visible(False)
    fig.legend(ax1.lines,[x.get_label() for x in ax1.lines],loc='upper right',prop=dict(size='xx-small'))
#    ax2.legend(loc='lower left',prop=dict(size='xx-small'),bbox_to_anchor=ax5.get_position())
    
def apply_limits(data,limits_dict):
    for name,limits in limits_dict.items():
        if name in data.columns:
            try:
                low,high = limits
            except (TypeError, ValueError) as e:
                raise ValueError("Invalid limits specified for parameter %s. Error was %s" % (name,str(e)))
            if low is not None:
                data = data[data[name]>=low]
            if high is not None:
                data = data[data[name]<=high]
    return data
                

def refine_dataset(original_data,settings):
    """
    Refine a data set based on data cuts specified in the settings dictionary
    """
    data = original_data[original_data.sweep_primary_load_temperature >= settings['valid_load_temp_range'][0]]
    data = data[data.sweep_primary_load_temperature <= settings['valid_load_temp_range'][1]]
    data = data[data.f_0_err/data.f_0 < settings['fractional_f_0_err_limit']]
    data = data[data.Q_err/data.Q < settings['fractional_Q_err_limit']]
    data = data[data.Q >= settings['valid_Q_range'][0]]
    data = data[data.Q <= settings['valid_Q_range'][1]]
    if settings['max_package_temp_deviation'] is not None:
        median_temp = np.median(data.sweep_primary_package_temperature)
        temp_deviations = np.abs(data.sweep_primary_package_temperature - median_temp)
        data = data[temp_deviations < settings['max_package_temp_deviation']]
    #data = data.sort(["f_0"])
    data['f_0_max'] = np.zeros((data.shape[0],))#data.groupby("resonator_index")["f_0"].transform(lambda x: x.max())
    data['Q_i_max'] = np.zeros((data.shape[0],))
    data['responsivity_Hz_per_K'] = np.zeros((data.shape[0],))
    data['responsivity_err'] = np.zeros((data.shape[0],))
    data['responsivity_offset'] = np.zeros((data.shape[0],))
    for index in np.unique(data.resonator_index):
        group = data[data.resonator_index == index]
        max = group[group.sweep_primary_load_temperature < settings['f_0_max_temp_limit']].f_0.max()
        data.f_0_max[data.resonator_index == index] = max
        max = group[group.sweep_primary_load_temperature < settings['f_0_max_temp_limit']].Q_i.max()
        data.Q_i_max[data.resonator_index == index] = max
        
    data['delta_f_0_Hz'] = (data.f_0-data.f_0_max)*1e6
    data['fractional_delta_f_0'] = data.delta_f_0_Hz/(1e6*data.f_0_max)#(1e6*data.noise_measurement_freq_MHz)
    data['fractional_delta_Q_i'] = data.Q_i/data.Q_i_max - 1

    for index in np.unique(data.resonator_index):
        group = data[data.resonator_index == index]
        try:
            (slope,offset),cov = np.polyfit(group.sweep_primary_load_temperature,group.delta_f_0_Hz,1,cov=True)
            print slope
            data.responsivity_Hz_per_K[data.resonator_index == index] = slope
            data.responsivity_offset[data.resonator_index == index] = offset
            data.responsivity_err[data.resonator_index == index] = np.sqrt(cov[1,1])
        except ValueError:
            continue
        except np.linalg.LinAlgError:
            continue
    eigvals_Hz = []
    nets = []
    for eigvals,freq,responsivity in zip(data.pca_eigvals,data.noise_measurement_freq_MHz,data.responsivity_Hz_per_K):
        # Convert eigvals spectra from 1/Hz units to Hz/sqrt(Hz)
        spectrum_Hz = np.sqrt(eigvals)*freq*1e6
        eigvals_Hz.append(spectrum_Hz)
        # Calculate net in muK sqrt(s). In the following, 1e6 is K -> uK factor, and sqrt(2) is 1/sqrt(Hz) -> sqrt(s) factor
        net = (1e6*spectrum_Hz/abs(responsivity))/np.sqrt(2)
        nets.append(net)
    data['pca_eigvals_Hz_per_rootHz'] = eigvals_Hz 
    data['net_uK_rootsec'] = nets
    return data
    
