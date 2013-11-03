import numpy as np
import bisect

import scipy.signal

import fftfilt

lpf = scipy.signal.firwin(256,1/256.)

class DataBlock():
    def __init__(self, data, tone, fftbin, 
                     nsamp, nfft, t0 = 0, fs = 512e6,
                     sweep_index = 0):
        self.data = data
        self.tone = tone
        self.fftbin = fftbin
        self.nsamp = nsamp
        self.nfft = nfft
        self.dt = 1/(fs/nfft)
        self.fs = fs
        self.t0 = t0
        self.sweep_index = sweep_index
        self._mean = None
        self._lpf_data = None
        
    def mean(self):
        if self._mean is None:
            self._lpf_data = fftfilt.fftfilt(lpf,self.data)[len(lpf):] 
            self._mean = self._lpf_data.mean(0)
        return self._mean
        
class SweepData():
    def __init__(self,sweep_id=1):
        self.sweep_id = sweep_id
        self.blocks = []
        self._freqs = []
        self._sweep_indexes = []
    def add_block(self,block):
        f = (block.fs*block.tone)/block.nsamp
        idx = bisect.bisect(self._freqs, f)
        self._freqs.insert(idx,f)
        self._sweep_indexes.insert(idx,block.sweep_index)
        self.blocks.insert(idx,block)
        
    def select_index(self,index):
        msk = self.sweep_indexes == index
        freqs = self.freqs[msk]
        data = self.data[msk]
        return freqs,data
    
    @property
    def freqs(self):
        return np.array(self._freqs)
    @property
    def data(self):
        return np.array([x.mean() for x in self.blocks])
    @property
    def sweep_indexes(self):
        return np.array(self._sweep_indexes)
