__author__ = 'gjones'

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import mlab
from kid_readout.analysis.timeseries import filters

class CrossSpectralAnalysis(object):
    def __init__(self,snm1=None,snm2=None):

        if snm1 is not None:
            self.get_data_from_snms(snm1,snm2)
    def get_data_from_snms(self,snm1,snm2,deglitch_thresh=5):
        self.snm1 = snm1
        self.snm2 = snm2
        self.snm1._fractional_fluctuation_timeseries = None
        self.snm2._fractional_fluctuation_timeseries = None
        self.snm1.deglitch_threshold=deglitch_thresh
        self.snm2.deglitch_threshold=deglitch_thresh

        self.snm1_phase = self.snm1.pca_angles[0,:10].mean()
        self.snm2_phase = self.snm2.pca_angles[0,:10].mean()
        self.corrected_timeseries1 = self.snm1.fractional_fluctuation_timeseries*np.exp(1j*self.snm1_phase)
        self.corrected_timeseries2 = self.snm2.fractional_fluctuation_timeseries*np.exp(1j*self.snm2_phase)
        self.timeseries_sample_rate = self.snm1.timeseries_sample_rate
        self.calculate()

    def get_data_directly(self,ts1,ts2,sample_rate=256e6/2**14):
        self.corrected_timeseries1 = ts1
        self.corrected_timeseries2 = ts2
        self.timeseries_sample_rate = sample_rate
        self.calculate()

    def calculate(self,NFFT=2**16,thresh=0.95):
        self.pxx1,self.freq = mlab.psd(self.corrected_timeseries1.real,NFFT=NFFT,Fs=self.timeseries_sample_rate)
        self.pii1,self.freq = mlab.psd(self.corrected_timeseries1.imag,NFFT=NFFT,Fs=self.timeseries_sample_rate)
        self.pxx2,self.freq = mlab.psd(self.corrected_timeseries2.real,NFFT=NFFT,Fs=self.timeseries_sample_rate)
        self.pii2,self.freq = mlab.psd(self.corrected_timeseries2.imag,NFFT=NFFT,Fs=self.timeseries_sample_rate)

        self.coherence,self.coh_freq = mlab.cohere(self.corrected_timeseries1.real,self.corrected_timeseries2.real,
                                     NFFT=NFFT,Fs=self.timeseries_sample_rate)
        #print self.coherence.mean()
        self.effective_dof = self.coherence.mean()*len(self.corrected_timeseries1)/(NFFT/2.)
        self.effective_dof = 0.25*(len(self.corrected_timeseries1)/(NFFT/2.))
        self.gamma95 = 1-(1-thresh)**(1./(self.effective_dof-1.))
        self.mask95 = self.coherence > self.gamma95

        self.csd,self.csd_freq = mlab.csd(self.corrected_timeseries1.real,self.corrected_timeseries2.real,
                            NFFT=NFFT,Fs=self.timeseries_sample_rate)

        self.angle = np.angle(self.csd)

        self.lpts1 = filters.low_pass_fir(self.corrected_timeseries1.real,cutoff=100.0,nyquist_freq=self.timeseries_sample_rate)
        self.lpts2 = filters.low_pass_fir(self.corrected_timeseries2.real,cutoff=100.0,
                                          nyquist_freq=self.timeseries_sample_rate)


    def plot(self):
        fig,(ax1) = plt.subplots(1,1,figsize=(8,6))
        fig.subplots_adjust(hspace=.3)
        #self.mask_95 = [i for i,v in enumerate(self.coherence) if v > self.gamma95]
        #self.mask_5 = [i for i, v in enumerate(self.coherence) if v < self.gamma95]

        self.interp_freq = np.linspace(min(self.coh_freq), max(self.coh_freq), len(self.coh_freq)*100)

        self.interp_coh = np.interp(self.interp_freq, self.coh_freq, self.coherence)

        self.mask_95 = self.interp_coh > self.gamma95

        ax1.semilogx(self.coh_freq,self.coherence, alpha = .7, color = 'k')
        ax1.semilogx(self.interp_freq[self.mask_95], self.interp_coh[self.mask_95], '.', markersize = 1, color='r')
        #ax2b = plt.twinx(ax2)
        ax1.axhline(self.gamma95, color = 'k', linestyle = '--')
        #ax2b.semilogx(self.coh_freq[self.mask95],self.angle[self.mask95], '.')
        #ax1.grid()
        #ax1.set_xlim(ax1.get_xlim())
        #ax2.set_title('Coherence/Phase')
        #ax1.set_xlabel('Frequency', fontsize = 24)
        ax1.set_xlabel('Hz', fontsize=24)
        ax1.set_ylabel('Coherence', fontsize = 24)
        ax1.set_xlim(.4, 10e3)
        ax1.set_ylim(0, 1)
        ax1.tick_params(labelsize = 24, size = 10)
        fig.tight_layout()
        #ax2b.set_ylabel('Phase Difference (rad)',color='red')
        #ax2b.set_ylim(-np.pi,np.pi)
