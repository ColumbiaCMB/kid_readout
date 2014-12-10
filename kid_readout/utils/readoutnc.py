import numpy as np
import netCDF4
import types
import bisect
import warnings
from collections import OrderedDict
from kid_readout.utils.roach_utils import ntone_power_correction
import kid_readout.utils.fftfilt
from kid_readout.utils.data_block import lpf
import kid_readout.utils.roach_utils


class TimestreamGroup(object):
    def __init__(self,ncgroup, parent=None):
        self.parent = parent
        keys = ncgroup.variables.keys()
        keys.remove('data')
        keys.remove('dt')
        keys.remove('fs')
        keys.remove('tone')
        keys.remove('nsamp')
        for key in keys:
            setattr(self,key,ncgroup.variables[key][:])
        required_keys = ['wavenorm', 'sweep_index']
        for key in required_keys:
            if key not in keys:
                setattr(self,key,None)
#        self.epoch = ncgroup.variables['epoch'][:]
        self.tonebin = ncgroup.variables['tone'][:]
        self.tone_nsamp = ncgroup.variables['nsamp'][:]
#        self.fftbin = ncgroup.variables['fftbin'][:]
#        self.nfft = ncgroup.variables['nfft'][:]
#        self.dt = ncgroup.variables['dt'][:] # the dt property is actually misleading at this point, so leaving it out
        self.adc_sampling_freq = ncgroup.variables['fs'][:]
        self.measurement_freq = self.adc_sampling_freq*self.tonebin/(1.0*self.tone_nsamp)
        self.sample_rate = self.adc_sampling_freq*1e6/(2*self.nfft)
#        if ncgroup.variables.has_key('wavenorm'):
#            self.wavenorm = ncgroup.variables['wavenorm'][:]
#        else:
#            self.wavenorm = None
#        if ncgroup.variables.has_key('sweep_index'):
#            self.sweep_index = ncgroup.variables['sweep_index'][:]
#        else:
#            self.sweep_index = None

        if self.parent is not None:
            self.modulation_duty_cycle = np.zeros_like(self.epoch)
            self.modulation_phase = np.zeros_like(self.epoch)
            self.modulation_freq = np.zeros_like(self.epoch)
            self.modulation_period_samples = np.zeros_like(self.epoch)
            for index in range(len(self.epoch)):
                out, rate = self.parent.get_modulation_state_at(self.epoch[index])
                if out == 2:
                    self.modulation_duty_cycle[index] = 0.5
                    self.modulation_freq[index] = self.sample_rate[index]/2.**rate
                    self.modulation_period_samples[index] = 2.**rate
                else:
                    self.modulation_duty_cycle[index] = out
                    self.modulation_freq[index] = 0.0
                    self.modulation_period_samples[index] = 0.0

        self._data = ncgroup.variables['data']
        self.num_data_samples = self._data.shape[1]
        self.data_len_seconds = self.num_data_samples/self.sample_rate
        self._datacache = None
        
    @property
    def data(self):
        if self._datacache is None:
            if self.wavenorm is None:
                wavenorm = 1.0
                warnings.warn("wave normalization not found, time series will not match sweep")
            else:
                wavenorm = self.wavenorm[:,None]
            self._datacache = self._data[:].view(self._data.datatype.name)*wavenorm
        return self._datacache
    
    def get_data_index(self,index):
        if self._datacache is None:
            if self.wavenorm is None:
                wavenorm = 1.0
                warnings.warn("wave normalization not found, time series will not match sweep")
            else:
                wavenorm = self.wavenorm[index]            
            return self._data[index].view(self._data.datatype.name)*wavenorm
        else:
            return self._datacache[index]
        
class SweepGroup(object):
    def __init__(self,ncgroup, parent=None):
        self.parent = parent
        self.frequency = ncgroup.variables['frequency'][:]
        self.s21 = ncgroup.variables['s21'][:].view(ncgroup.variables['s21'].datatype.name)
        self.index = ncgroup.variables['index'][:]
        self.timestream_group = TimestreamGroup(ncgroup.groups['datablocks'], parent=parent)
        self.start_epoch = self.timestream_group.epoch.min()
        self.end_epoch = self.timestream_group.epoch.max()
    
    @property
    def errors(self):
        if self.timestream_group.wavenorm is None:
            wavenorm = 1
        else:
            wavenorm = self.timestream_group.wavenorm[0]
        errors = np.zeros(self.timestream_group.data.shape[0], dtype='complex')
        for index in range(self.timestream_group.data.shape[0]):
            filtered = kid_readout.utils.fftfilt.fftfilt(lpf, self.timestream_group.data[index,:])[len(lpf):]
            # the standard deviation is scaled by the number of independent samples
            # to compute the error on the mean.
            error_scaling = np.sqrt(float(len(filtered))/len(lpf))
            real_error = filtered.real.std()/error_scaling
            imag_error = filtered.imag.std()/error_scaling
            errors[index] = real_error + 1j*imag_error

        return errors

    def select_by_index(self,index):
        mask = self.index == index
        freq,s21,errors = self.frequency[mask], self.s21[mask], self.errors[mask]
        order = freq.argsort()
        return freq[order], s21[order], errors[order]
    
    def select_by_frequency(self,freq):
        findex = np.argmin(abs(self.frequency - freq))
        index = self.index[findex]
        return self.select_by_index(index)
    
class ReadoutNetCDF(object):
    def __init__(self,filename):
        self.filename = filename
        self.ncroot = netCDF4.Dataset(filename,mode='r')
        hwgroup = self.ncroot.groups['hw_state']
        self.hardware_state_epoch = hwgroup.variables['epoch'][:]
        self.adc_atten = hwgroup.variables['adc_atten'][:]
        self.dac_atten = hwgroup.variables['dac_atten'][:]
        if 'ntones' in hwgroup.variables:
            self.num_tones = hwgroup.variables['ntones'][:]
        else:
            self.num_tones = None
        for key in ['modulation_rate', 'modulation_output']:
            if key in hwgroup.variables:
                self.__setattr__(key,hwgroup.variables[key][:])
            else:
                self.__setattr__(key,None)

        try:
            self.gitinfo = self.ncroot.gitinfo
        except AttributeError:
            self.gitinfo = ''

        try:
            self.boffile = self.ncroot.boffile
        except AttributeError:
            self.boffile = ''

        try:
            self.mmw_atten_turns = self.ncroot.mmw_atten_turns
        except AttributeError:
            self.mmw_atten_turns = (np.nan,np.nan)
            
        self.sweeps_dict = OrderedDict()
        self.timestreams_dict = OrderedDict()
        for name,group in self.ncroot.groups['sweeps'].groups.items():
            self.sweeps_dict[name] = SweepGroup(group, parent=self)
            self.__setattr__(name,self.sweeps_dict[name])
        self.sweeps = self.sweeps_dict.values()
        for name,group in self.ncroot.groups['timestreams'].groups.items():
            self.timestreams_dict[name] = TimestreamGroup(group, parent=self)
            self.__setattr__(name,self.timestreams_dict[name])
        self.timestreams = self.timestreams_dict.values()
    def close(self):
        self.ncroot.close()

    def get_delay_estimate(self):
        if self.boffile == '':
            try:
                nfft = self.sweeps[0].timestream_group.nfft[0]
            except IndexError:
                raise Exception("could not find any means to estimate the delay for %s" % self.filename)
            return kid_readout.utils.roach_utils.get_delay_estimate_for_nfft(nfft)
        else:
            return kid_readout.utils.roach_utils.get_delay_estimate_for_boffile(self.boffile)

    def _get_hwstate_index_at(self,epoch):
        """
        Find the index of the hardware state arrays corresponding to the hardware state at a given epoch
        :param epoch: unix timestamp
        :return:
        """
        index = bisect.bisect_left(self.hardware_state_epoch, epoch) # find the index of the epoch immediately preceding the desired epoch
        index = index - 1
        if index < 0:
            index = 0
        return index

    def get_effective_dac_atten_at(self,epoch):
        """
        Get the dac attenuator value and total signal attenuation at a given time
        :param epoch: unix timestamp
        :return: dac attenuator in dB, total attenuation in dB
        """
        index = self._get_hwstate_index_at(epoch)
        dac_atten = self.dac_atten[index]
        if self.num_tones is not None:
            ntones = self.num_tones[index]
        else:
            ntones = 1
            warnings.warn("ntones parameter not found in data file %s, assuming 1. The effective power level may be wrong" % self.filename)
        total = dac_atten + ntone_power_correction(ntones)
        return dac_atten, total

    def get_modulation_state_at(self,epoch):
        """
        Get the source modulation TTL output state at a given time
        :param epoch: unix timestamp
        :return: modulation output state: 0 -> low, 1 -> high, 2 -> modulated
         modulation rate parameter: FIXME
        """
        if self.modulation_rate is None:
            return 0,0
        index = self._get_hwstate_index_at(epoch)
        modulation_rate = self.modulation_rate[index]
        modulation_output = self.modulation_output[index]
        return modulation_output, modulation_rate