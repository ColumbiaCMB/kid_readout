import numpy as np
from matplotlib import pyplot as plt

import kid_readout.analysis.fit_pulses
import kid_readout.analysis.resonator
import kid_readout.utils.readoutnc

def normalized_s21_to_detuning(s21,resonator):
    if 'a' in resonator.result.params:
        print "warning: inverse not yet defined for bifurcation model, proceeding anyway"
    Q = resonator.Q
    Qe = resonator.Q_e
    x = 1j*(Qe*(s21-1)+Q) / (2*Qe*Q*(s21-1))
    return x.real

class MmwResponse(object):
    def __init__(self,ncfilename,resonator_index):
        rnc = kid_readout.utils.readoutnc.ReadoutNetCDF(ncfilename)
        self.resonator_index=resonator_index
        sweep = rnc.sweeps[0]
        num_resonators = len(np.unique(sweep.index))
        self.sweep_freq, self.sweep_s21, self.sweep_s21_error = sweep.select_by_index(self.resonator_index)
        self.resonator = kid_readout.analysis.resonator.fit_best_resonator(self.sweep_freq,self.sweep_s21,
                                                                           errors=self.sweep_s21_error)

        self.mmw_atten_turns = rnc.ncroot.mmw_atten_turns[:]
        self.dac_atten = rnc.dac_atten[0]
        timestream = rnc.timestreams[0]
        modulation_freq = timestream.mmw_source_modulation_freq[0]
        self.measurement_freq = timestream.measurement_freq[self.resonator_index]
        sample_rate = timestream.sample_rate[0]
        samples_per_period = int(np.round(sample_rate/modulation_freq))
        total_num_timestreams = timestream.epoch.shape[0]
        timestream_indexes = range(self.resonator_index,total_num_timestreams,num_resonators)
        num_timestreams = len(timestream_indexes)
        self.raw_high = np.zeros((num_timestreams,),dtype='complex')
        self.raw_low = np.zeros((num_timestreams,),dtype='complex')
        self.mmw_freq = np.zeros((num_timestreams,))
        #self.s0 = self.resonator.model(x=self.measurement_freq)
        self.s0 = self.resonator.s21_data[np.abs(self.resonator.freq_data-self.measurement_freq).argmin()]
        self.ns0 = self.resonator.normalized_model(self.measurement_freq)
        for k,index in enumerate(timestream_indexes):
            data = timestream.get_data_index(index)
            num_periods = data.shape[0]//samples_per_period
            data = data[:samples_per_period*num_periods].reshape((num_periods,samples_per_period)).mean(0)
            high,low = kid_readout.analysis.fit_pulses.find_high_low(data-self.s0)
            self.raw_high[k] = high+self.s0
            self.raw_low[k] = low+self.s0
            self.mmw_freq[k] = timestream.mmw_source_freq[index]

        self.normalized_high = self.resonator.normalize(self.measurement_freq,self.raw_high)
        self.normalized_low = self.resonator.normalize(self.measurement_freq,self.raw_low)

        self.mmw_on_frac_freq = normalized_s21_to_detuning(self.normalized_high,self.resonator)
        self.mmw_off_frac_freq = normalized_s21_to_detuning(self.normalized_low,self.resonator)

        if (self.mmw_on_frac_freq-self.mmw_off_frac_freq).sum() > 0:
            self.mmw_frac_response = self.mmw_on_frac_freq - self.mmw_off_frac_freq
        else:
            self.mmw_frac_response = self.mmw_off_frac_freq - self.mmw_on_frac_freq


    def plot(self):
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(121)
        ax.plot(self.mmw_freq/1e9,1e6*self.mmw_frac_response)
        ax.set_xlabel('mm-wave freq [GHz]')
        ax.set_ylabel('frac. freq. response [ppm]')
        if 'a' in self.resonator.result.params:
            non_lin_str = 'a: %.3f' % self.resonator.a
        else:
            non_lin_str = ''
        ax.set_title('Resonator %d, mm-wave attenuators: %.1f, %.1f turns\nDAC atten: %.1f %s' %
                     (self.resonator_index, self.mmw_atten_turns[0],self.mmw_atten_turns[1], self.dac_atten,
                     non_lin_str))
        ax2 = fig.add_subplot(122)
        ns21 = self.resonator.normalize(self.resonator.freq_data,self.resonator.s21_data)
        mfrq = np.linspace(self.resonator.freq_data.min(),self.resonator.freq_data.max(),1000)
        nms21 = self.resonator.normalized_model(mfrq)
        ax2.plot(ns21.real,ns21.imag,'kx',mew=2)
        ax2.plot(nms21.real,nms21.imag,'y')
        ax2.plot(self.normalized_high.real,self.normalized_high.imag,'r.')
        ax2.plot(self.normalized_low.real,self.normalized_low.imag,'g.')
        ax2.set_title('%.6f MHz' % (self.measurement_freq))
        ax2.plot(self.ns0.real,self.ns0.imag,'b+',mew=2,markersize=20)


