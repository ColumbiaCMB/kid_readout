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

import kid_readout.analysis.noise_archive

default_settings = dict(fractional_f_0_err_limit = 1e-6,
                        fractional_Q_err_limit = 0.06,
                        valid_Q_range = (5000,1e6),
                        max_package_temp_deviation = 0.01,
                        )
settings = dict(valid_load_temp_range = (4,8.0),
                f_0_max_temp_limit = 5.0,
                )

all_settings = dict()
all_settings.update(default_settings)
all_settings.update(settings)

def plot_s21(data,ax=None,min_load_temp=4,max_load_temp=8):
    """
    Plot IQ resonance circles at different load temperatures
    
    all sweeps will be plotted, so may want to down select data before
    """
    if ax is None:
        fig,ax = plt.subplots(figsize=(8,8))
    data = data.sort(['sweep_primary_load_temperature'])
    for k,row in data.iterrows():
        s21 = row['sweep_normalized_s21']
        load_temp = row['sweep_primary_load_temperature']
        color = plt.cm.spectral((load_temp-min_load_temp+.1)/(max_load_temp-min_load_temp+.1))
        s21m = row['sweep_model_normalized_s21']
        ax.plot(s21m.real,s21m.imag,color=color)
        ax.plot(s21.real,s21.imag,'.',color=color)
        #lpf = row['low_pass_normalized_timeseries'].mean()
        #ax.plot(lpf.real,lpf.imag,'x',mew=2,markersize=20,color=color)
        s0 = row['normalized_model_s21_at_resonance']
        ax.plot(s0.real,s0.imag,'+',mew=2,markersize=20,color='k')
    ax.grid()
    ax.set_xlim(0,1.1)
    ax.set_ylim(-0.5,0.5)
    
def plot_load_vs_freq_shift(all_data,axs=None,min_load_temp=4,max_load_temp=8,anchor_range = (30,100)):       
    """
    Plot shift in resonance frequency as a function of load temperature
    """
    if axs is None:
        fig,(ax1,ax2,ax3,ax4) = plt.subplots(nrows=4,figsize=(8,12),sharex=True,squeeze=True)
    else:
        ax1,ax2,ax3,ax4 = axs
    x = np.linspace(min_load_temp,max_load_temp,100)
    slope = all_data.responsivity_Hz_per_K.iat[0]
    offset = all_data.responsivity_offset.iat[0]
    polynomial = np.array([slope,offset])
    y = np.polyval(polynomial,x)
    fractional = slope/(all_data.f_0_max.iat[0])

    anchors = all_data[(all_data.timestream_duration>anchor_range[0]) & 
                    (all_data.timestream_duration<anchor_range[1])]

    for name,marker,color,data in zip(['in transition','steady'],
                                ['.','o'],
                                ['b','r'],
                                [all_data,anchors]):
        load_temp = data.sweep_primary_load_temperature #data.timestream_primary_load_temperature.apply(np.mean) 
        ax1.errorbar(load_temp,data.delta_f_0_Hz,yerr=data.f_0_err*1e6,marker=marker,linestyle='none',label=name,color=color)
        ax2.errorbar(load_temp,data.delta_f_0_Hz-np.polyval(polynomial,data.sweep_primary_load_temperature),
                     yerr=data.f_0_err*1e6,linestyle='none',marker=marker,color=color,label=name)
        ax3.plot(load_temp,1e6/data.Q_i,marker,label=name,color=color)
        ax4.plot(load_temp,1e6/data.Q,marker,label=name,color=color)

    ax1.plot(x,y,label=('%.0f+/-%.0f Hz/K\n %.1fppm/K' % (slope,all_data.responsivity_err.iat[0], fractional)))
    ax2.set_ylim(-200,200)
    ax1.legend(loc='lower left',prop=dict(size='small'))
    ax1.grid()
    ax1.set_xlabel('Load Temperature (K)')
    ax2.grid()
    ax2.set_ylabel('Residual (Hz)')
    ax2.set_xlabel('Load Temperature (K)')
    ax1.set_ylabel('Frequency Shift (Hz)')
    ax3.grid()
    ax3.set_xlabel('Load_Temperature(K)')
    ax3.set_ylabel('$10^6/Q_i$')
    ax4.grid()
    ax4.set_xlabel('Load_Temperature(K)')
    ax4.set_ylabel('$10^6/Q_r$')
    
    
def plot_load_vs_fractional_freq_shift(all_data,ax=None):
    """
    Plot fractional frequency shift as a function of load temperature for all resonators
    """
    if ax is None:
        fig,ax = plt.subplots(figsize=(8,8))
    for name, group in all_data.groupby('resonator_index'):
        ax.plot(group.sweep_primary_load_temperature,group.fractional_delta_f_0,'.')
    ax.grid()
    ax.set_ylim(-2e-4,1e-5)
    ax.set_ylabel('Fractional Frequency Shift')
    ax.set_xlabel('Load Temperature (K)')
    return fig
    
def plot_noise(data,axs=None,min_load_temp=4,max_load_temp=8,max_uK=100):
    """
    Plot noise spectra and noise levels vs load temperature
    """
    if axs is None:
        fig,(ax1,ax2) = plt.subplots(nrows=2,figsize=(8,8),squeeze=True)
    else:
        ax1,ax2 = axs
    label1 = '150 Hz'
    label2 = '15 kHz'
    for k,row in data.iterrows():  # .sort(['sweep_primary_load_temperature'])
        load_temp = row['sweep_primary_load_temperature']
        color = plt.cm.spectral((load_temp-min_load_temp+.1)/(max_load_temp-min_load_temp+.1))
        freq = row['pca_freq']
        net = row['net_uK_rootsec']
        ax1.semilogx(freq,net[1,:],color=color,lw=2,label=('%.2f K' % load_temp))
        noise_150 = net[1,:][(freq>100) & (freq<200)].mean()
        noise_10k = net[1,:][(freq>1e4) & (freq<2e4)].mean()
        ax2.plot(load_temp,noise_150,'o',color=color,mew=2,markersize=10,label=label1)
        ax2.plot(load_temp,noise_10k,'d',color=color,mew=2,markersize=10,label=label2)
        if label1:
            label1 = None
            label2 = None
    ax1.grid()
    ax1b = ax1.twinx()
    ax1.set_xlim(10,2e4)
    ax1.set_ylim(0,max_uK)
    ax1b.set_ylim(0,max_uK*(1e-6*np.sqrt(2)*abs(data.responsivity_Hz_per_K.iat[0])))
    ax1b.grid(color='r')
    ax1.axvspan(100,200,hatch='/',fc='none',alpha=0.2)
    ax1.axvspan(1e4,2e4,hatch='/',fc='none',alpha=0.2)
    ax1.set_xlabel('Hz')
    ax1.set_ylabel(r'$\mu$K$\sqrt{s}$')
    ax1b.set_ylabel(r'$Hz/\sqrt{Hz}$')
    ax2.grid()
    ax2b = ax2.twinx()
    ax2.set_ylim(0,max_uK)
    ax2b.set_ylim(0,max_uK*(1e-6*np.sqrt(2)*abs(data.responsivity_Hz_per_K.iat[0])))
    ax2b.grid(color='r')
    ax2.legend(loc='lower right',prop=dict(size='small'))
    ax2.set_ylabel(r'$\mu$K$\sqrt{s}$')
    ax2b.set_ylabel(r'$Hz/\sqrt{Hz}$')
    ax2.set_xlabel('Load Temperature (K)')
    
    
def plot_resonator_net(data,resonator_index=0,fig = None,axs=None,anchor_range=(30,100),min_load_temp=4, max_load_temp=8,max_uK=70):
    """
    Make complete plot for a given resonator (including S21 sweep, responsivity, Q's, and noise)
    """
    if fig is None:
        fig,axs = plt.subplots(ncols=4,nrows=2,figsize=(20,10))
        fig.subplots_adjust(wspace=.3)
    data = data[data.resonator_index == resonator_index]
    anchors = data[(data.timestream_duration>anchor_range[0]) & 
                    (data.timestream_duration<anchor_range[1])]
    plot_s21(anchors, ax=axs[1,0], min_load_temp=min_load_temp, max_load_temp=max_load_temp)
    plot_load_vs_freq_shift(data, axs=[axs[0,1],axs[1,1],axs[0,2],axs[1,2]], min_load_temp=min_load_temp, max_load_temp=max_load_temp, 
                            anchor_range=anchor_range)
    plot_noise(anchors, axs=[axs[0,3],axs[1,3]], min_load_temp=min_load_temp, max_load_temp=max_load_temp, max_uK=max_uK)
    axs[0,0].set_visible(False)
    info = data.chip_name.iat[0].replace(', ','\n')
    info += ('\nResonator: %d @ %.6f MHz' % (resonator_index,data.f_0_max.iat[0]))
    files = np.unique(data.sweep_filename)
    files.sort()
    files = files.tolist()
    median_temp = np.median(data.sweep_primary_package_temperature)
    temp_rms = np.std(data.sweep_primary_package_temperature)
    info += ('\nfirst file: %s' % files[0][:30])
    info += ('\nlast file: %s' % files[-1][:30])
    info += ('\nPackage Temperature: %.1f$\pm$%.1f mK' % (median_temp*1000,temp_rms*1000))
    info += ('\nPower ~ %.1f dBm\n    (%.1f dB atten, %.1f dB cold)' % (data.power_dbm.iat[0],data.atten.iat[0],data.dac_chain_gain.iat[0]))
    
    fig.text(0.1,0.9,info,ha='left',va='top',size='x-small',bbox=dict(facecolor='w',pad=8))
    return fig

def plot_net_dataset(data,pdfname=None,plotdir='/home/data/plots',**kwargs):
    """
    Make PDF of plots for all resonators in a dataset.
    """
    chip_name = data.chip_name.iat[0]
    chipfname = chip_name.replace(' ','_').replace(',','')
    files = np.unique(data.sweep_filename)
    files.sort()
    files = files.tolist()
    first = os.path.splitext(os.path.split(files[0])[1])[0]
    last = os.path.splitext(os.path.split(files[-1])[1])[0]
    
    if pdfname is None:
        pdfname = '/home/data/plots/net_summary_%s_%s_to_%s.pdf' % (chipfname,first,last)
    pdf = PdfPages(pdfname)
    try:
        os.chmod(pdfname,0666)
    except OSError:
        print "could not change permissions of",pdfname
        
    indexes = np.unique(data.resonator_index)
    for index in indexes:
        try:
            fig = plot_resonator_net(data, resonator_index=index, **kwargs)
        except Exception, e:
            print index,e
            continue
        #title = ('%s\nResonator: %d @ %.6f MHz' % (chip_name,index,data[data.resonator_index==index].f_0_max.iat[0]))
        #fig.suptitle(title)
        canvas = FigureCanvasAgg(fig)
        fig.set_canvas(canvas)
        pdf.savefig(fig,bbox_inches='tight')
        
    fig = plot_load_vs_fractional_freq_shift(data)
    canvas = FigureCanvasAgg(fig)
    fig.set_canvas(canvas)
    pdf.savefig(fig,bbox_inches='tight')

    pdf.close()



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
    
