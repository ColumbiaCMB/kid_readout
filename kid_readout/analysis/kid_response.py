import numpy as np
import lmfit
import kid_readout.analysis.fitter
import kid_readout.analysis.mcfit

def fractional_freq_to_power(x,break_point,scale):
    return (scale*((x/break_point+1)**2-1))

def d_power_d_fractional_freq(frac_freq,break_point,scale):
    return 2*scale*(break_point+frac_freq)/break_point**2

def fractional_freq_to_power_model(params,x):
    break_point = params['break_point'].value
    scale = params['scale'].value
    return fractional_freq_to_power(x,break_point,scale)

def fractional_freq_to_power_guess(x,y):
    mid = 10**np.log10(x).mean()
    scale = y.max()
    params = lmfit.Parameters()
    params.add('break_point',1/mid,min=x.min(),max=x.max())
    params.add('scale',scale,min=0,max=1)
    return params

class MCMCKidResponseFitter(kid_readout.analysis.mcfit.MCMCFitter):
    def __init__(self, x_data, y_data,
                 model=fractional_freq_to_power_model, guess=fractional_freq_to_power_guess, functions={},
                 mask=None, errors=None, weight_by_errors=True, method='leastsq', nwalkers=32):
        super(MCMCKidResponseFitter,self).__init__(x_data, y_data,
                 model=model, guess=guess, functions=functions,
                 mask=mask, errors=errors, weight_by_errors=weight_by_errors, method=method)

        self.setup_sampler(nwalkers=nwalkers)

    def to_power(self,frac_freq):
        return self.model(self.mcmc_params,x=frac_freq)

    def dpdf(self,frac_freq):
        scale = self.mcmc_params['scale'].value
        break_point = self.mcmc_params['break_point'].value
        return d_power_d_fractional_freq(frac_freq,break_point,scale)
