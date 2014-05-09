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

from kid_readout.analysis import kid_eqns

default_settings = dict(fractional_f_0_err_limit = 1e-6,
                        fractional_Q_err_limit = 0.06,
                        valid_Q_range = (5000,1e6),
                        max_package_temp_deviation = None,
                        )
settings = dict(valid_load_temp_range = (4,8.0),
                f_0_max_temp_limit = 5.0,
                )

all_settings = dict()
all_settings.update(default_settings)
all_settings.update(settings)

def plot_mattis_bardeen(data,axs=None):
    if axs is None:
        fig,(ax1,ax2,ax3,ax4) = plt.subplots(nrows=4,figsize=(8,12),sharex=True,squeeze=True)
    else:
        ax1,ax2,ax3,ax4 = axs
        
    temps = np.array(data.sweep_primary_package_temperature)
    Qis = np.array(data.Q_i)
    fractional_freq = np.array(data.fractional_delta_f_0)
    
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
    data['responsivity_Hz_per_K'] = np.zeros((data.shape[0],))
    data['responsivity_err'] = np.zeros((data.shape[0],))
    data['responsivity_offset'] = np.zeros((data.shape[0],))
    for index in np.unique(data.resonator_index):
        group = data[data.resonator_index == index]
        max = group[group.sweep_primary_load_temperature < settings['f_0_max_temp_limit']].f_0.max()
        data.f_0_max[data.resonator_index == index] = max
    data['delta_f_0_Hz'] = (data.f_0-data.f_0_max)*1e6
    data['fractional_delta_f_0'] = data.delta_f_0_Hz/(1e6*data.f_0_max)#(1e6*data.noise_measurement_freq_MHz)

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
    
