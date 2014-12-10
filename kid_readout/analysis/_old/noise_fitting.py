"""
va,vb,vi = True: values of alpha, beta, i params are variable respectively
k=1: the module fits the noise 1 time. va,vb,vi will not be use.
For all k!=1, the noise is fitted with va=True, vb=True, vi=True
              second fit picks up values of params returned by first fit with 
              value of va,vb,vi depending on choices of users.
       
plot whole pkls: k = plot_pkls_noise(fn,k=1)
plot single file: k = plot_single_noise(f,pxx)
"""

from __future__ import division
import cPickle

import numpy as np
import matplotlib
from lmfit import minimize

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 16.0
import os
    
from kid_readout.analysis._old.noise_function import pkl_mask as default_mask
from kid_readout.analysis._old.noise_function import guess as default_guess
from kid_readout.analysis._old.noise_function import model as default_model
from kid_readout.analysis._old.noise_function import plot_single_pkl_noise as plot_single_pkl_noise
from kid_readout.analysis._old.noise_function import crop_data as crop_data
class Noise(object):
    def __init__(self,f,p,A=4709632.149,B=1,alpha=-3,beta=-0.1,fc=2e3,i=3,N_white=1e-3,
                 guess_func=default_guess,model_func=default_model,mask_func=None,
                 k=1,va=True,vb=True,vi=True): 
        if mask_func is None:
            self.f = f
            self.p = p
        else:
            f_masked,p_masked = mask_func(f,p)
            self.f = f_masked
            self.p = p_masked
        self._model = model_func
        self._guess = guess_func
        self.fit(self.guess(A,B,alpha,beta,fc,i,N_white=N_white),k,va,vb,vi)
           
    def __getattr__(self,attr):
        try:
            return self.result.params[attr].value
        except KeyError:
            pass

    def __dir__(self):
        return (dir(super(Noise, self)) +
                self.__dict__.keys() +
                self.result.params.keys())
        
    def fit(self,initial,k,va,vb,vi):
        if k==1:
            self.result = minimize(self.residual,initial) #ftol=1e-6
        else:
            minimize(self.residual,initial)
            A = initial['A'].value
            B = initial['B'].value
            alpha = initial['alpha'].value
            beta = initial['beta'].value
            N_white = initial['N_white'].value
            fc = initial['fc'].value
            i = initial['i'].value
            p2 = self.guess(A,B,alpha,beta,fc,i,N_white=N_white,va=va,vb=vb,vi=vi)
            self.result = minimize(self.residual,p2)
            
    def guess(self,A,B,alpha,beta,fc,i,N_white=1e-3,va=True,vb=True,vi=True):
        return self._guess(A,B,alpha,beta,fc,i,N_white=N_white,va=va,vb=vb,vi=vi)
        
    def residual(self,params):
        return (self.p - self.model(params))
        
    def model(self,params):
        return self._model(self.f,params)
        
def plot_pkls_noise(fns,A=4709632.149,B=1,alpha=-3,beta=-0.1,fc=2e3,i=3,N_white=1e-3,
                    front_crop=1,end_crop=2e4,
                    guess_func=default_guess,model_func=default_model,
                    mask_func=default_mask,
                    k=1,va=True,vb=True,vi=True):

    params_dict = dict([('A',[]),('alpha',[]),('B',[]),('beta',[]),('fc',[]),('i',[]),('N_white',[])])
    fdir,fbase = os.path.split(fns)
    fh = open(fns)
    try:
        pkls = cPickle.load(fh)
    except EOFError:
        print "End of File Error"
    else:
        fh.close()            
        for el in range(np.size(pkls)):
            nm=pkls[el]
            f = nm.fr_coarse
            pxx = nm.prr_coarse
            f,pxx = crop_data(f,pxx,front_crop=front_crop,end_crop=end_crop)
            
            n = Noise(f,pxx,A,B,alpha,beta,fc,i,N_white=N_white,
                      guess_func=guess_func,model_func=model_func,
                      mask_func=mask_func,
                      k=k,va=va,vb=vb,vi=vi)
                      
            plot_single_pkl_noise(n,pkls_name=fbase,single_pkl_no=el)
        
            params_dict['A'].append((el,n.A))
            params_dict['alpha'].append((el,n.alpha))
            params_dict['B'].append((el,n.B))
            params_dict['beta'].append((el,n.beta))
            params_dict['i'].append((el,n.i))
            params_dict['fc'].append((el,n.fc))
            params_dict['N_white'].append((el,n.N_white))
            
    return params_dict

def plot_single_noise(f,pxx,A=4709632.149,B=1,alpha=-3,beta=-0.1,fc=2e3,i=3,N_white=1e-3,
                      front_crop=1,end_crop=2e4,
                      guess_func=default_guess,model_func=default_model,
                      mask_func=default_mask,
                      k=1,va=True,vb=True,vi=True):
                          
    f,pxx = crop_data(f,pxx,front_crop=front_crop,end_crop=end_crop)
    n = Noise(f,pxx,A,B,alpha,beta,fc,i,N_white=N_white,
              guess_func=guess_func,model_func=model_func,
              mask_func=mask_func,
              k=k,va=va,vb=vb,vi=vi)
              
    plot_single_pkl_noise(n)
    
    params_dict = dict([('A',n.A),('alpha',n.alpha),('B',n.B),('beta',n.beta),('fc',n.fc),('i',n.i),('N_white',n.N_white)])
    return params_dict