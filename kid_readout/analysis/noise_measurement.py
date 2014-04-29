import numpy as np
import matplotlib
matplotlib.use('agg')
#matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 16.0
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
mlab = plt.mlab

from kid_readout.utils.easync import EasyNetCDF4
from kid_readout.analysis.resonator import Resonator,fit_best_resonator
from kid_readout.analysis import khalil
from kid_readout.analysis import iqnoise
import scipy.signal

#from kid_readout.utils.fftfilt import fftfilt
from kid_readout.utils.filters import low_pass_fir
from kid_readout.utils.roach_utils import ntone_power_correction

from kid_readout.utils.despike import deglitch_window

import socket
if socket.gethostname() == 'detectors':
    from kid_readout.utils.hpd_temps import get_temperature_at
else:
    from kid_readout.utils.parse_srs import get_temperature_at
import bisect
import time
import os
import glob

import cPickle


class SweepNoiseMeasurement(object):
    def __init__(self,sweep,timestream,readoutnc,chip_name,index=0,low_pass_cutoff_Hz=4.0,
                 dac_chain_gain = -52, ntones=None, use_bifurcation=False, delay_estimate=-7.29,
                 deglitch_threshold=5):
        self.sweep_epoch = sweep.start_epoch
        self.start_temp = get_temperature_at(self.sweep_epoch)
        self.ts_epoch = timestream.epoch[index]
        self.end_temp = get_temperature_at(self.ts_epoch)
        self.index = index
        self.chip_name = chip_name
        self.dac_chain_gain = dac_chain_gain
        
        try:
            self.atten, self.total_dac_atten = readoutnc.get_effective_dac_atten_at(self.sweep_epoch)
            self.power_dbm = dac_chain_gain - self.total_dac_atten
        except:
            print "failed to find attenuator settings"
            self.atten = np.nan
            self.total_dac_atten = np.nan
            self.power_dbm = np.nan
            
        self.sweep_freqs_MHz, self.sweep_s21, self.sweep_errors = sweep.select_by_index(index)
        
        # find the time series that was measured closest to the sweep frequencies
        # this is a bit sloppy...
        timestream_index = np.argmin(abs(timestream.measurement_freq-self.sweep_freqs_MHz.mean()))
        
        original_timeseries = timestream.get_data_index(timestream_index)
        self.adc_sampling_freq_MHz = timestream.adc_sampling_freq[timestream_index]
        self.noise_measurement_freq_MHz = timestream.measurement_freq[timestream_index]
        self.nfft = timestream.nfft[timestream_index]
        self.timeseries_sample_rate_Hz = timestream.sample_rate[timestream_index]
        
        # We can use the timestream measurement as an additional sweep point.
        # We average only the first 2048 points of the timeseries to avoid any drift. 
        self.sweep_freqs_MHz = np.hstack((self.sweep_freqs_MHz,[self.noise_measurement_freq_MHz]))
        self.sweep_s21 = np.hstack((self.sweep_s21,[original_timeseries[:2048].mean()]))
        self.sweep_errors = np.hstack((self.sweep_errors,
                                           [original_timeseries[:2048].real.std()/np.sqrt(2048)
                                            +original_timeseries[:2048].imag.std()/np.sqrt(2048)]))
        
        # Now put all the sweep data in increasing frequency order so it plots nicely
        order = self.sweep_freqs_MHz.argsort()
        self.sweep_freqs_MHz = self.sweep_freqs_MHz[order]
        self.sweep_s21 = self.sweep_s21[order]
        self.sweep_errors = self.sweep_errors[order]
        
        rr = fit_best_resonator(self.sweep_freqs_MHz,self.sweep_s21,errors=self.sweep_errors,delay_estimate=delay_estimate)
        self.Q_i = rr.Q_i
        self.fit_params = rr.result.params
        
        decimation_factor = self.timeseries_sample_rate_Hz/low_pass_cutoff_Hz
        normalized_timeseries = rr.normalize(self.noise_measurement_freq_MHz,original_timeseries)
        self.low_pass_normalized_timeseries = low_pass_fir(normalized_timeseries, num_taps=1024, cutoff=low_pass_cutoff_Hz, 
                                                          nyquist_freq=self.timeseries_sample_rate_Hz, decimate_by=decimation_factor)
        self.normalized_timeseries_mean = normalized_timeseries.mean()

        projected_timeseries = rr.project_s21_to_delta_freq(self.noise_measurement_freq_MHz,normalized_timeseries,
                                                            s21_already_normalized=True)
        
        # calculate the number of samples for the deglitching window.
        # the following will be the next power of two above 1 second worth of samples
        window = int(2**np.ceil(np.log2(self.timeseries_sample_rate_Hz)))
        # reduce the deglitching window if we don't have enough samples
        if window > projected_timeseries.shape[0]:
            window = projected_timeseries.shape[0]//2
        deglitched_timeseries = deglitch_window(projected_timeseries,window,thresh=5)
        
        
        self.low_pass_projected_timeseries = low_pass_fir(deglitched_timeseries, num_taps=1024, cutoff=low_pass_cutoff_Hz, 
                                                nyquist_freq=self.timeseries_sample_rate_Hz, decimate_by=decimation_factor)
        self.low_pass_timestep = decimation_factor/self.timeseries_sample_rate_Hz
        
        self.normalized_model_s21_at_meas_freq = rr.normalized_model(self.noise_measurement_freq_MHz)
        self.normalized_model_s21_at_resonance = rr.normalized_model(rr.f_0)
        self.normalized_ds21_df_at_meas_freq = rr.approx_normalized_gradient(self.noise_measurement_freq_MHz)
        
        self.sweep_normalized_s21 = rr.normalize(self.sweep_freqs_MHz,self.sweep_s21)

        self.sweep_model_freqs_MHz = np.linspace(self.sweep_freqs_MHz.min(),self.sweep_freqs_MHz.max(),1000)
        self.sweep_model_normalized_s21 = rr.normalized_model(self.sweep_model_freqs_MHz) 
        self.sweep_model_normalized_s21_centered = self.sweep_model_normalized_s21 - self.normalized_timeseries_mean
        
        fractional_fluctuation_timeseries = deglitched_timeseries / (self.noise_measurement_freq_MHz*1e6)
        fr,S,evals,evects,angles,piq = iqnoise.pca_noise(fractional_fluctuation_timeseries, 
                                                         NFFT=None, Fs=self.timeseries_sample_rate_Hz)
        
        self.pca_freq = fr
        self.pca_S = S
        self.pca_evals = evals
        self.pca_evects = evects
        self.pca_angles = angles
        self.pca_piq = piq
        
        self.prr_fine,self.sweep_freqs_MHz_fine = mlab.psd(fractional_fluctuation_timeseries.real,NFFT=2**18,window=mlab.window_none,Fs=self.timeseries_sample_rate_Hz)
        self.pii_fine,fr = mlab.psd(fractional_fluctuation_timeseries.imag,NFFT=2**18,window=mlab.window_none,Fs=self.timeseries_sample_rate_Hz)
        self.prr_coarse,self.sweep_freqs_MHz_coarse = mlab.psd(fractional_fluctuation_timeseries.real,NFFT=2**12,window=mlab.window_none,Fs=self.timeseries_sample_rate_Hz)
        self.pii_coarse,fr = mlab.psd(fractional_fluctuation_timeseries.imag,NFFT=2**12,window=mlab.window_none,Fs=self.timeseries_sample_rate_Hz)
        
        self.normalized_timeseries = normalized_timeseries[:2048].copy()

        
    def plot(self,fig=None,title=''):
        if fig is None:
            f1 = plt.figure(figsize=(16,8))
        else:
            f1 = fig
        ax1 = f1.add_subplot(121)
        ax2 = f1.add_subplot(222)
        ax2b = ax2.twinx()
        ax2b.set_yscale('log')
        ax3 = f1.add_subplot(224)
        f1.subplots_adjust(hspace=0.25)
        
        ax1.plot((self.sweep_normalized_s21).real,(self.sweep_normalized_s21).imag,'.-',lw=2,label='measured frequency sweep')
        ax1.plot(self.sweep_model_normalized_s21.real,self.sweep_model_normalized_s21.imag,'.-',markersize=2,label='model frequency sweep')
        ax1.plot([self.normalized_model_s21_at_resonance.real],[self.normalized_model_s21_at_resonance.imag],'kx',mew=2,markersize=20,label='model f0')
        ax1.plot([self.normalized_timeseries_mean.real],[self.normalized_timeseries_mean.imag],'m+',mew=2,markersize=20,label='timeseries mean')
        ax1.plot(self.normalized_timeseries.real[:128],self.normalized_timeseries.imag[:128],'k,',alpha=1,label='timeseries samples')
        ax1.plot(self.low_pass_normalized_timeseries.real,self.low_pass_normalized_timeseries.imag,'r,') #uses proxy for label
        #ax1.plot(self.pca_evects[0,0,:100]*100,self.pca_evects[1,0,:100]*100,'y.')
        #ax1.plot(self.pca_evects[0,1,:100]*100,self.pca_evects[1,1,:100]*100,'k.')
        x1 = self.normalized_model_s21_at_meas_freq.real
        y1 = self.normalized_model_s21_at_meas_freq.imag
        x2 = x1 + self.normalized_ds21_df_at_meas_freq.real*100
        y2 = y1 + self.normalized_ds21_df_at_meas_freq.imag*100
        ax1.annotate("",xytext=(x1,y1),xy=(x2,y2),arrowprops=dict(lw=2,color='orange',arrowstyle='->'),zorder=0)
        #proxies
        l = plt.Line2D([0,0.1],[0,0.1],color='orange',lw=2)
        l2 = plt.Line2D([0,0.1],[0,0.1],color='r',lw=2)
        ax1.text((self.sweep_normalized_s21).real[0],(self.sweep_normalized_s21).imag[0],('%.3f kHz' % ((self.sweep_freqs_MHz[0]-self.noise_measurement_freq_MHz)*1000)))
        ax1.text((self.sweep_normalized_s21).real[-1],(self.sweep_normalized_s21).imag[-1],('%.3f kHz' % ((self.sweep_freqs_MHz[-1]-self.noise_measurement_freq_MHz)*1000)))
        ax1.set_xlim(0,1.1)
        ax1.set_ylim(-.55,.55)
        ax1.grid()
        handles,labels = ax1.get_legend_handles_labels()
        handles.append(l)
        labels.append('dS21/(100Hz)')
        handles.append(l2)
        labels.append('LPF timeseries')
        ax1.legend(handles,labels,prop=dict(size='xx-small'))
        
        ax1b = inset_axes(parent_axes=ax1, width="20%", height="20%", loc=4)
        ax1b.plot(self.sweep_freqs_MHz,20*np.log10(abs(self.sweep_normalized_s21)),'.-')
        frm = np.linspace(self.sweep_freqs_MHz.min(),self.sweep_freqs_MHz.max(),1000)
        ax1b.plot(frm,20*np.log10(abs(self.sweep_model_normalized_s21)))
                
        ax2.loglog(self.sweep_freqs_MHz_fine[1:],self.prr_fine[1:],'b',label='Srr')
        ax2.loglog(self.sweep_freqs_MHz_fine[1:],self.pii_fine[1:],'g',label='Sii')
        ax2.loglog(self.sweep_freqs_MHz_coarse[1:],self.prr_coarse[1:],'y',lw=2)
        ax2.loglog(self.sweep_freqs_MHz_coarse[1:],self.pii_coarse[1:],'m',lw=2)
        ax2.loglog(self.pca_freq[1:],self.pca_evals[:,1:].T,'k',lw=2)
        ax2.set_title(title,fontdict=dict(size='small'))
        
        n500 = self.prr_coarse[np.abs(self.sweep_freqs_MHz_coarse-500).argmin()]
        ax2.annotate(("%.2g Hz$^2$/Hz @ 500 Hz" % n500),xy=(500,n500),xycoords='data',xytext=(5,20),textcoords='offset points',
                     arrowprops=dict(arrowstyle='->'))
        
        ax2b.set_xscale('log')
    #    ax2b.set_xlim(ax2.get_xlim())
        ax2.grid()
        ax2b.grid(color='r')
        ax2.set_xlim(self.pca_freq[1],self.pca_freq[-1])
        ax2.set_ylabel('1/Hz')
        ax2.set_xlabel('Hz')
        ax2.legend(prop=dict(size='small'))
        
        tsl = self.low_pass_projected_timeseries
        tsl = tsl - tsl.mean()
        dtl = self.low_pass_timestep
        t = dtl*np.arange(len(tsl))
        ax3.plot(t,tsl.real,'b',lw=2,label = 'LPF timeseries real')
        ax3.plot(t,tsl.imag,'g',lw=2,label = 'LPF timeseries imag')
        ax3.set_ylabel('Hz')
        ax3.set_xlabel('seconds')
        ax3.legend(prop=dict(size='xx-small'))
        
        params = self.fit_params
        amp_noise_voltsrthz = np.sqrt(4*1.38e-23*4.0*50)
        vread = np.sqrt(50*10**(self.power_dbm/10.0)*1e-3)
        alpha = 1.0
        Qe = abs(params['Q_e_real'].value+1j*params['Q_e_imag'].value)
        f0_dVdf = 4*vread*alpha*params['Q'].value**2/Qe
        expected_amp_noise = (amp_noise_voltsrthz/f0_dVdf)**2 
        text = (("measured at: %.6f MHz\n" % self.noise_measurement_freq_MHz)
                + ("temperature: %.1f - %.1f mK\n" %(self.start_temp*1000, self.end_temp*1000))
                + ("power: ~%.1f dBm (%.1f dB att)\n" %(self.power_dbm,self.atten))
                + ("fit f0: %.6f +/- %.6f MHz\n" % (params['f_0'].value,params['f_0'].stderr))
                + ("Q: %.1f +/- %.1f\n" % (params['Q'].value,params['Q'].stderr))
                + ("Re(Qe): %.1f +/- %.1f\n" % (params['Q_e_real'].value,params['Q_e_real'].stderr))
                + ("|Qe|: %.1f\n" % (Qe))
                + ("Qi: %.1f\n" % (self.Q_i))
                + ("Eamp: %.2g 1/Hz" % expected_amp_noise)
                )
        if expected_amp_noise > 0:
            ax2.axhline(expected_amp_noise,linewidth=2,color='m')
            ax2.text(10,expected_amp_noise,r"expected amp noise",va='top',ha='left',fontdict=dict(size='small'))
#        ax2.axhline(expected_amp_noise*4,linewidth=2,color='m')
#        ax2.text(10,expected_amp_noise*4,r"$\alpha = 0.5$",va='top',ha='left',fontdict=dict(size='small'))
        if params.has_key('a'):
            text += ("\na: %.3g +/- %.3g" % (params['a'].value,params['a'].stderr))

        ylim = ax2.get_ylim()
        ax2b.set_ylim(ylim[0]*(self.noise_measurement_freq_MHz*1e6)**2,ylim[1]*(self.noise_measurement_freq_MHz*1e6)**2)
        ax2b.set_ylabel('$Hz^2/Hz$')

        
        ax1.text(0.02,0.95,text,ha='left',va='top',bbox=dict(fc='white',alpha=0.6),transform = ax1.transAxes,
                 fontdict=dict(size='x-small'))
        
        title = ("%s\nmeasured %s\nplotted %s" % (self.chip_name,time.ctime(self.sweep_epoch),time.ctime()))
        ax1.set_title(title,size='small')
        return f1
        
def load_noise_pkl(pklname):
    fh = open(pklname,'r')
    pkl = cPickle.load(fh)
    fh.close()
    return pkl
