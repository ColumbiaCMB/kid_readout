import numpy as np
import scipy.signal
from kid_readout.analysis.timeseries import fftfilt

class DecimatingFIR(object):
    def __init__(self,downsample_factor=16, num_taps=1024):
        self.downsample_factor = downsample_factor
        self.num_taps = num_taps
        self.prepare()

    def prepare(self):
        self.coefficients = scipy.signal.firwin(self.num_taps,cutoff=1./self.downsample_factor).reshape((
            self.num_taps/self.downsample_factor,self.downsample_factor))[:,::-1]

    def process(self,data,continuation=True, use_fft=True):
        data = data.reshape((data.shape[0]//self.downsample_factor,self.downsample_factor))
        if use_fft:
            result = fftfilt.fftfilt_nd(self.coefficients,data)
        else:
            result = np.empty_like(data)
            for k in range(self.downsample_factor):
                result[:,k] = scipy.signal.lfilter(self.coefficients[:,k],1,data[:,k])
                #result[:,k] = np.convolve(self.coefficients[:,k],data[:,k],mode='same')


        return result.sum(1)

class FIR1D(object):
    def __init__(self,coeff):
        self.coeff = coeff
        self.num_taps = coeff.size
        self._history = None
    def apply(self,data):
        data = np.atleast_2d(data)
        result = np.apply_along_axis(np.convolve,1,data,self.coeff,mode='full')
        if self._history is not None:
            result[:,:self.num_taps-1] += self._history
        self._history = result[:,-(self.num_taps-1):]
        result = result[:,:-(self.num_taps-1)]
        return result.squeeze()

class Delay1D(object):
    def __init__(self,coeff,num_taps):
        self.coeff = coeff
        self.num_taps = num_taps
        self._history = None
    def apply(self,data):
        data = np.atleast_2d(data)
        result = np.empty((data.shape[0],data.shape[1]+self.num_taps-1),dtype=data.dtype)
        result[:,:self.num_taps//2] = 0
        result[:,self.num_taps//2:-(self.num_taps//2-1)] = data*self.coeff
        if self._history is not None:
            result[:,:self.num_taps-1] += self._history
        self._history = result[:,-(self.num_taps-1):]
        result = result[:,:-(self.num_taps-1)]
        return result.squeeze()


class HalfBandFilter(object):
    def __init__(self,num_taps=16,window_param=('chebwin',80),coeff_dtype=np.float32):
        self.num_taps = num_taps
        self.window_param=window_param
        self.coeff = scipy.signal.firwin(num_taps+1,0.5,window=window_param).astype(coeff_dtype)[:-1]
        self.coeff[np.abs(self.coeff)<1e-14]=0.0
        self._naive_filter = FIR1D(self.coeff)
        self._polyphase_component = FIR1D(self.coeff[1::2])
        self._polyphase_even_coeff = self.coeff[num_taps//2]
        self._polyphase_component_even = FIR1D(self.coeff[::2])
        #self._polyphase_component_even = Delay1D(self._polyphase_even_coeff,num_taps//2)
    def naive(self,data):
        return self._naive_filter.apply(data)
    def polyphase_explicit_even_component(self,data):
        result = self._polyphase_component.apply(data[1::2])
        result += self._polyphase_component_even.apply(data[::2])
        return result
    def polyphase(self,data):
        result = self._polyphase_component.apply(data[1::2])
        result += self._polyphase_component_even.apply(data[::2])
#        result[self.num_taps-1:,...] += self._polyphase_even_coeff*data[::2][:-(self.num_taps-1)]
        return result

class MultistageHalfBandDecimationFilter(object):
    def __init__(self,taps,window_param=('hamming',)):
        self.taps = taps
        self.num_stages = taps.size
        self.filters = [HalfBandFilter(num_taps=num_taps,window_param=window_param) for num_taps in taps]
    def naive(self,data,decimate=True):
        for filter_ in self.filters:
            data = filter_.naive(data)
            if decimate:
                data = data[::2]
        return data
    def polyphase(self,data):
        for filter_ in self.filters:
            data = filter_.polyphase(data)
        return data