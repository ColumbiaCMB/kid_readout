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
        fig,(ax3,ax1,ax2) = plt.subplots(3,1,figsize=(18,18))
        fig.subplots_adjust(hspace=.3)
        ax1.loglog(self.freq,self.pxx1,'k')
        ax1.loglog(self.freq,self.pxx2,'r')
        ax1.loglog(self.freq,self.pii1,'k',alpha=0.5)
        ax1.loglog(self.freq,self.pii2,'r',alpha=0.5)
        ax1.loglog(self.freq[self.mask95],self.pxx1[self.mask95],'b.')
        ax1.loglog(self.freq[self.mask95],self.pxx2[self.mask95],'b.')
        ax1.grid()
        ax1.set_title('Spectra')
        ax1.set_xlabel('Frequency')


        ax2.semilogx(self.coh_freq,self.coherence)
        ax2b = plt.twinx(ax2)
        ax2.axhline(self.gamma95)
        ax2b.semilogx(self.coh_freq[self.mask95],self.angle[self.mask95],'r.')
        ax2.grid()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_title('Coherence/Phase')
        ax2.set_xlabel('Frequency')
        ax2.set_ylabel('Coherence',color='blue')
        ax2b.set_ylabel('Phase Difference (rad)',color='red')
        ax2b.set_ylim(-np.pi,np.pi)



#        ax3.plot(1e3*np.arange(1000)/self.timeseries_sample_rate,self.corrected_timeseries1[:1000].real,'k')
#        ax3.plot(1e3*np.arange(1000)/self.timeseries_sample_rate,self.corrected_timeseries2[:1000].real,'r')
        #ax3.plot(np.arange(50000)/self.timeseries_sample_rate,self.lpts1[:50000],'k')
        #ax3.plot(np.arange(50000)/self.timeseries_sample_rate,self.lpts2[:50000],'r')
        ax3.plot(np.arange(self.lpts1.shape[0])/self.timeseries_sample_rate,self.lpts1,'k')
        ax3.plot(np.arange(self.lpts1.shape[0])/self.timeseries_sample_rate,self.lpts2,'r')
        ax3.set_title('Timeseries')
        ax3.set_xlabel('s')


