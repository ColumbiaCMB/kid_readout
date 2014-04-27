import numpy as np
from scipy.misc import logsumexp
import emcee
import triangle
import lmfit
from kid_readout.analysis.fitter import Fitter
from kid_readout.analysis.resonator import Resonator


parameter_list = ['f_0',
                 'A_mag',
#                 'A_phase',
                 'Q',
                 'Q_e_real',
                 'Q_e_imag',
                 'delay',
                 'phi',
                 'a']

def convert_to_lmfit_params(raw_params,base_params):
    lm_params = lmfit.Parameters()
    for index,value in enumerate(raw_params):
        lm_params.add(parameter_list[index],value=value)
    # Now add parameters from the base list.
    for name,param in base_params.items():
        if name not in lm_params:
            lm_params.add(name,value=param.value)
    return lm_params


class MCMCFitter(Fitter):
    def get_param_bounds_by_index(self,index):
        name = parameter_list[index]
        if not self.result.params[name].vary:
            raise Exception("Non varying parameter found %s" % name) 
        err = self.result.params[name].stderr
        value = self.result.params[name].value
        min = value - 10*err
        max = value + 10*err
        if min < self.result.params[name].min:
            min = self.result.params[name].min
        if max > self.result.params[name].max:
            max = self.result.params[name].max
        return min,max
    def uniform_logprior(self,params):
        for k in range(len(params)):
            value = params[k]
            min,max = self.get_param_bounds_by_index(k)
            if value > max or value < min:
                return -np.inf
        return 0.
    
    def basic_loglikelihood(self,params):
        model = self.model(params = convert_to_lmfit_params(params,self.result.params))
        return (-np.log(self.errors)
                - 0.5*abs((self.y_data-model)/self.errors)**2)
    def basic_logprob(self,params):
        if np.isinf(self.uniform_logprior(params)):
            return -np.inf
        return np.sum(self.basic_loglikelihood(params))

class MCMCResonator(Resonator,MCMCFitter):
    def setup_sampler(self,nwalkers=32):
        ndim = len(parameter_list)
        if not 'a' in self.result.params:
            ndim -= 1
        self.initial = np.zeros((nwalkers,ndim))
        for dim in range(ndim):
            min,max = self.get_param_bounds_by_index(dim)
            self.initial[:,dim] = np.random.uniform(min,max,size=nwalkers)
        self.sampler = emcee.EnsembleSampler(nwalkers,ndim,self.basic_logprob)
        
