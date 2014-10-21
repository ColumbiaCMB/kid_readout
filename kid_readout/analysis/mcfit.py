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

def convert_to_lmfit_params(raw_params,base_params,raw_param_list=parameter_list):
    lm_params = lmfit.Parameters()
    for index,value in enumerate(raw_params):
        lm_params.add(raw_param_list[index],value=value)
    # Now add parameters from the base list.
    for name,param in base_params.items():
        if name not in lm_params:
            lm_params.add(name,value=param.value)
    return lm_params


class MCMCFitter(Fitter):
    def get_param_bounds_by_index(self,index,error_factor=None):
        parameter_list = self.parameter_list # need to set this up, something like self.reseult.params.keys() but ommitting non varying params
        name = parameter_list[index]
        if not self.result.params[name].vary:
            raise Exception("Non varying parameter found %s" % name)
        min = self.result.params[name].min
        max = self.result.params[name].max
        if error_factor is not None:
            err = self.result.params[name].stderr
            value = self.result.params[name].value
            min = value - error_factor*err
            max = value + error_factor*err
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
        model = self.model(params = convert_to_lmfit_params(params,self.result.params,self.result.params.keys()))
        return (-np.log(np.abs(self.errors))
                - 0.5*abs((self.y_data-model)/self.errors)**2)
    def basic_logprob(self,params):
        if np.isinf(self.uniform_logprior(params)):
            return -np.inf
        return np.sum(self.basic_loglikelihood(params))

    def setup_sampler(self,nwalkers=32,error_factor=None):
        ndim = len(self.result.params)
        self.initial = np.zeros((nwalkers,ndim))
        for dim in range(ndim):
            min,max = self.get_param_bounds_by_index(dim,error_factor=error_factor)
            self.initial[:,dim] = np.random.uniform(min,max,size=nwalkers)
        self.sampler = emcee.EnsembleSampler(nwalkers,ndim,self.basic_logprob)


class MCMCResonator(Resonator,MCMCFitter):
    def setup_sampler(self,nwalkers=32):
        self.parameter_list = parameter_list
        ndim = len(self.parameter_list)
        if not 'a' in self.result.params:
            ndim -= 1
#        for name,param in self.result.params.items():
#            if name not in ['phi', 'delay','Q_e_imag']:
#                if param.value*1.2 <= param.max:
#                    param.max = param.value*1.2
#                if param.value*0.8 >= param.min:
#                    param.min = param.value*0.8
#            else:
#                if name == 'Q_e_imag':
#                    param.max = param.value + 1000
#                    param.min = param.value - 1000
        self.ndim = ndim
        self.initial = np.zeros((nwalkers,ndim))
        for dim in range(ndim):
            min,max = self.get_param_bounds_by_index(dim,error_factor=10)
            vals = self.result.params[self.parameter_list[dim]].value +\
                                  self.result.params[self.parameter_list[dim]].stderr*np.random.randn(nwalkers)
            vals[vals > max] = max
            vals[vals < min] = min
            self.initial[:,dim] = vals
        self.sampler = emcee.EnsembleSampler(nwalkers,ndim,self.basic_logprob)

    def basic_loglikelihood(self,params):
        model = self.model(params = convert_to_lmfit_params(params,self.result.params))
        return (-np.log(np.abs(self.errors))
                - 0.5*abs((self.y_data-model)/self.errors)**2)

    def run(self,length=500,burn_in=100,nwalkers=32):
        self.setup_sampler(nwalkers=nwalkers)
        self.sampler.run_mcmc(self.initial,length)
        self.samples = self.sampler.chain[:,burn_in:,:].reshape((-1,self.ndim))

    def triangle(self,*args,**kwargs):
        import triangle
        kwargs['labels'] = self.parameter_list
        triangle.corner(self.samples,*args,**kwargs)

    def uniform_logprior(self,params):
        for k in range(len(params)):
            value = params[k]
            min,max = self.get_param_bounds_by_index(k)

            if value > max or value < min:
                return -np.inf
        return 0.
