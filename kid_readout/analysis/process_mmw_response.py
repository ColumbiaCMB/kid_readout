import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import kid_readout.analysis.fit_pulses
import kid_readout.analysis.resonator
from kid_readout.analysis.resonator import normalized_s21_to_detuning
import kid_readout.utils.readoutnc
import kid_readout.analysis.resources.skip5x4
from kid_readout.analysis.resources.local_settings import BASE_DATA_DIR

file_id_to_res_id = [0,1,2,3,4,5,6,7,8,17,16,15,14,13,10,9]

def plot_file_with_air_spacer_reference(filename):
    d1 = np.load(os.path.join(BASE_DATA_DIR,'mmw_sweeps/2014-09-03_150009_140.000_161.000_10.0.npz'))
    d2 = np.load(os.path.join(BASE_DATA_DIR,'mmw_sweeps/2014-09-11_105355_135.000_165.000_10.0.npz'))
    waveguide_freq = d1['mmw_freq']
    waveguide_response = d1['power_watts']/d1['power_watts'].max()
    optics_freq = d2['mmw_freq']
    optics_response = d2['power_watts']/d2['power_watts'].max()
    total_response = np.interp(waveguide_freq,optics_freq,optics_response)*waveguide_response
    total_freq = waveguide_freq

    plot_file(filename,reference_spectrum=(total_freq,total_response*9))


def plot_file(filename,num_resonators=16,reference_spectrum=None):
    plt.rcParams['font.size'] = 16
    mmws = [MmwResponse(filename,k) for k in range(num_resonators)]
    blah, fbase = os.path.split(filename)
    fbase,ext = os.path.splitext(fbase)
    pdfname = os.path.join(BASE_DATA_DIR,'plots/%s.pdf') % (fbase,)
    pdf = PdfPages(pdfname)
    try:
        os.chmod(pdfname,0666)
    except OSError:
        print "could not change permissions of",pdfname

    fig = Figure(figsize=(12,12))
    ax = fig.add_subplot(221)
    for mmw in mmws:
        ax.plot(mmw.mmw_freq/1e9,1e6*np.abs(mmw.mmw_frac_response))
    if reference_spectrum is not None:
        ax.plot(reference_spectrum[0]/1e9,reference_spectrum[1])
    ax.set_title('Response of all detectors')
    ax.set_ylabel('Frac. freq response [ppm]')
    ax.set_xlabel('mm-wave source freq [GHz]')

    ax = fig.add_subplot(222)
    ref = mmws[3]
    for mmw in mmws:
        ax.plot(mmw.mmw_freq/1e9,np.abs(mmw.mmw_frac_response/ref.mmw_frac_response))
        ax.set_ylim(0,3)
        ax.set_title('Relative response of all detectors versus #3')
        ax.set_ylabel('response / (det. #3 response)')
        ax.set_xlabel('mm-wave source freq [GHz]')

    ax = fig.add_subplot(223)
    for mmw in mmws:
        ax.plot(mmw.mmw_freq/1e9,10*np.log10(np.abs(mmw.mmw_frac_response)))
    ax.set_title('Response of all detectors [dB]')
    ax.set_ylabel('Frac. freq response [dB]')
    ax.set_xlabel('mm-wave source freq [GHz]')

    canvas = FigureCanvasAgg(fig)
    pdf.savefig(fig,bbox_inches='tight')

    fig = Figure(figsize=(16,8))
    for n,mmw in enumerate(mmws):
        file_id = mmw.resonator_index
        resonator_id = file_id_to_res_id[file_id]
        x,y = kid_readout.analysis.resources.skip5x4.id_to_coordinate[resonator_id]
        ax = fig.add_axes([x/5.0,y/4.0,0.8/5.0,0.8/4.0])
#        ax = fig.add_subplot(4,5,n+1)
        ax.plot(mmw.mmw_freq/1e9,1e6*np.abs(mmw.mmw_frac_response))
        if reference_spectrum is not None:
            ax.plot(reference_spectrum[0]/1e9,reference_spectrum[1],color='gray')
        ax.text(0.1,0.9,("%.6f MHz" % (mmw.resonator.f_0)),transform=ax.transAxes,ha='left',va='top',size='small')
        ax.set_ylim(0,8)
    canvas = FigureCanvasAgg(fig)
    pdf.savefig(fig,bbox_inches='tight')

    fig = Figure(figsize=(16,8))
    for n,mmw in enumerate(mmws):
        file_id = mmw.resonator_index
        resonator_id = file_id_to_res_id[file_id]
        x,y = kid_readout.analysis.resources.skip5x4.id_to_coordinate[resonator_id]
        ax = fig.add_axes([x/5.0,y/4.0,0.8/5.0,0.8/4.0])
#        ax = fig.add_subplot(4,5,n+1)
        ax.plot(1e3*np.arange(mmw.aligned_data.shape[1])/(256e6/2**14), np.abs(mmw.aligned_data[::40,:].T-mmw.s0))
#        ax.plot(mmw.mmw_freq/1e9,1e6*(mmw.mmw_frac_response))
        ax.text(0.1,0.9,("%.6f MHz" % (mmw.resonator.f_0)),transform=ax.transAxes,ha='left',va='top',size='small')
        #ax.set_ylim(0,8)
        ax.set_xlabel('ms')
    canvas = FigureCanvasAgg(fig)
    pdf.savefig(fig,bbox_inches='tight')

    for mmw in mmws:
        fig = Figure(figsize=(16,8))
        mmw.plot(fig=fig)
        canvas = FigureCanvasAgg(fig)
        #fig.set_canvas(canvas)
        pdf.savefig(fig,bbox_inches='tight')
    pdf.close()

class MmwResponse(object):
    def __init__(self,ncfilename,resonator_index):
        rnc = kid_readout.utils.readoutnc.ReadoutNetCDF(ncfilename)
        self.resonator_index=resonator_index
        sweep = rnc.sweeps[0]
        num_resonators = len(np.unique(sweep.index))
        self.sweep_freq, self.sweep_s21, self.sweep_s21_error = sweep.select_by_index(self.resonator_index)
        self.resonator = kid_readout.analysis.resonator.fit_best_resonator(self.sweep_freq,self.sweep_s21,
                                                                           errors=self.sweep_s21_error)

        try:
            self.mmw_atten_turns = rnc.ncroot.mmw_atten_turns[:]
        except AttributeError:
            self.mmw_atten_turns = np.zeros((2,))
            self.mmw_atten_turns[:] = np.nan
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
        self.zbd_voltage = np.zeros((num_timestreams,))
        self.aligned_data = np.zeros((num_timestreams,samples_per_period),dtype='complex')
        #self.s0 = self.resonator.model(x=self.measurement_freq)
        self.s0 = self.resonator.s21_data[np.abs(self.resonator.freq_data-self.measurement_freq).argmin()]
        self.ns0 = self.resonator.normalized_model(self.measurement_freq)
        for k,index in enumerate(timestream_indexes):
            data = timestream.get_data_index(index)
            num_periods = data.shape[0]//samples_per_period
            data = data[:samples_per_period*num_periods].reshape((num_periods,samples_per_period)).mean(0)
            high,low,rising_edge = kid_readout.analysis.fit_pulses.find_high_low(data-self.s0)
            self.aligned_data[k,:] = np.roll(data,-rising_edge)
            self.raw_high[k] = high+self.s0
            self.raw_low[k] = low+self.s0
            self.mmw_freq[k] = timestream.mmw_source_freq[index]
            self.zbd_voltage[k] = timestream.zbd_voltage[index]

        self.normalized_high = self.resonator.normalize(self.measurement_freq,self.raw_high)
        self.normalized_low = self.resonator.normalize(self.measurement_freq,self.raw_low)

        self.mmw_on_frac_freq = normalized_s21_to_detuning(self.normalized_high,self.resonator)
        self.mmw_off_frac_freq = normalized_s21_to_detuning(self.normalized_low,self.resonator)

        if (self.mmw_on_frac_freq-self.mmw_off_frac_freq).sum() > 0:
            self.mmw_frac_response = self.mmw_on_frac_freq - self.mmw_off_frac_freq
        else:
            self.mmw_frac_response = self.mmw_off_frac_freq - self.mmw_on_frac_freq

        rnc.close()


    def plot(self,fig=None):
        if fig is None:
            fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(121)
        ax.plot(self.mmw_freq/1e9,1e6*np.abs(self.mmw_frac_response))
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
        mfrq = np.linspace(self.resonator.freq_data.min(),self.resonator.freq_data.max(),10000)
        nms21 = self.resonator.normalized_model(mfrq)
        ax2.plot(ns21.real,ns21.imag,'kx',mew=2)
        ax2.plot(nms21.real,nms21.imag,'y')
        ax2.plot(self.normalized_high.real,self.normalized_high.imag,'r.')
        ax2.plot(self.normalized_low.real,self.normalized_low.imag,'g.')
        ax2.set_title('%.6f MHz' % (self.measurement_freq))
        ax2.plot(self.ns0.real,self.ns0.imag,'b+',mew=2,markersize=20)


