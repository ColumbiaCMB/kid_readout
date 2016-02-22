import numpy as np
from scipy.misc import logsumexp
import emcee
import lmfit
from kid_readout.analysis.fitter import Fitter
from kid_readout.analysis.resonator.resonator import Resonator


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

class GeneralMCMC():
    def __init__(self, fitter):
        self.fitter = fitter
        self.parameter_list = []
        self.parameter_mins = []
        self.parameter_maxs = []
        self.fixed_parameters = []
        self.update_parameter_list()
    def update_parameter_list(self):
        self.parameter_list = []
        self.parameter_mins = []
        self.parameter_maxs = []
        self.fixed_parameters = []
        for name,param in self.fitter.result.params.items():
            if param.vary:
                self.parameter_list.append(name)
                self.parameter_mins.append(param.min)
                self.parameter_maxs.append(param.max)
            else:
                self.fixed_parameters.append(name)
    def set_parameter_values(self,args):
        if len(args) != len(self.parameter_list):
            raise ValueError("Got %d values but have %d parameters. values were %s, parameters are %s" %
                             (len(args), len(self.parameter_list), str(args), str(self.parameter_list)))
        for name,value in zip(self.parameter_list,args):
            self.fitter.result.params[name].value = value

    def uniform_logprior(self,args):
        if len(args) != len(self.parameter_list):
            raise ValueError("Got %d values but have %d parameters. values were %s, parameters are %s" %
                             (len(args), len(self.parameter_list), str(args), str(self.parameter_list)))

        for value,min,max in zip(args,self.parameter_mins,self.parameter_maxs):
            if value > max or value < min:
                return -np.inf
        return 0.

    def basic_loglikelihood(self,params):
        self.set_parameter_values(params)
        residual, errors = self.fitter.residual()
        if errors is None:
            errors = np.ones(residual.shape[0])
        return (-np.log(np.abs(errors))
                - 0.5*abs(residual/errors)**2)
    def basic_logprob(self,params):
        if np.isinf(self.uniform_logprior(params)):
            return -np.inf
        return np.sum(self.basic_loglikelihood(params))

    def setup_sampler(self,nwalkers=32,error_factor=None):
        self.update_parameter_list()
        ndim = len(self.parameter_list)
        self.ndim = ndim
        self.initial = np.zeros((nwalkers,ndim))
        for dim in range(ndim):
            min,max = self.parameter_mins[dim], self.parameter_maxs[dim]
            self.initial[:,dim] = np.random.uniform(min,max,size=nwalkers)
        self.sampler = emcee.EnsembleSampler(nwalkers,ndim,self.basic_logprob,threads=4)

    def run(self,length=500,burn_in=100,nwalkers=32):
        self.setup_sampler(nwalkers=nwalkers)
        self.sampler.run_mcmc(self.initial,length)
        self.samples = self.sampler.chain[:,burn_in:,:].reshape((-1,self.ndim))
        for dim,name in enumerate(self.parameter_list):
            self.fitter.params[name].value = self.samples[:,dim].mean()
            self.fitter.params[name].stderr = self.samples[:,dim].std()

    def triangle(self,*args,**kwargs):
        import triangle
        kwargs['labels'] = self.parameter_list
        triangle.corner(self.samples,*args,**kwargs)


class MCMCFitter(Fitter):
    def __init__(self,*args,**kwargs):
        super(MCMCFitter,self).__init__(*args,**kwargs)
        self.parameter_list = self.result.params.keys()
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

    def run(self,length=500,burn_in=100,nwalkers=32):
        self.setup_sampler(nwalkers=nwalkers)
        self.sampler.run_mcmc(self.initial,length)
        self.samples = self.sampler.chain[:,burn_in:,:].reshape((-1,self.ndim))
        self.mcmc_params = self.result.params.copy()
        for dim in range(self.ndim):
            self.mcmc_params[self.mcmc_params.keys()[dim]].value = self.samples[:,dim].mean()
            self.mcmc_params[self.mcmc_params.keys()[dim]].stderr = self.samples[:,dim].std()

    def triangle(self,*args,**kwargs):
        import triangle
        kwargs['labels'] = self.parameter_list
        triangle.corner(self.samples,*args,**kwargs)

    def setup_sampler(self,nwalkers=32,error_factor=None):
        ndim = len(self.result.params)
        self.ndim = ndim
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


    def uniform_logprior(self,params):
        for k in range(len(params)):
            value = params[k]
            min,max = self.get_param_bounds_by_index(k)

            if value > max or value < min:
                return -np.inf
        return 0.
