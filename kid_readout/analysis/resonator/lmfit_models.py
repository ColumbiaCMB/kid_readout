import numpy as np
import lmfit
import equations
from distutils.version import StrictVersion

def update_param_values_and_limits(pars, prefix, **kwargs):
    for key, val in kwargs.items():
        if key.endswith('_max'):
            key_name = key[:-len('_max')]
            attr = 'max'
        elif key.endswith('_min'):
            key_name = key[:-len('_min')]
            attr = 'min'
        else:
            key_name = key
            attr = 'value'
        pname = "%s%s" % (prefix, key_name)
        if pname in pars:
            setattr(pars[pname],attr,val)
    return pars

# Version 0.9.3 of lmfit incorporates changes that allow models that return complex values.
# For earlier versions, we need the following work around
if StrictVersion(lmfit.__version__) < StrictVersion('0.9.3'):
    class ComplexModel(lmfit.model.Model):
        def _residual(self,params,data,weights,**kwargs):
            diff = self.eval(params, **kwargs) - data
            diff_as_ri = diff.astype('complex').view('float')
            if weights is not None:
                weights_as_ri = weights.astype('complex').view('float')
                diff_as_ri *= weights_as_ri
            retval = np.asarray(diff_as_ri).ravel()
            return retval
else:
    ComplexModel = lmfit.model.Model

class GeneralCableModel(ComplexModel):
    def __init__(self, *args, **kwargs):
        super(GeneralCableModel, self).__init__(equations.general_cable, *args, **kwargs)
    def guess(self,data=None, **kwargs):
        pass

class LinearResonatorModel(ComplexModel):
    def __init__(self, *args, **kwargs):
        super(LinearResonatorModel, self).__init__(equations.linear_resonator, *args, **kwargs)
        self.set_param_hint('Q', min = 0)  # Enforce Q is positive
    def guess(self, data, f=None, **kwargs):
        verbose = kwargs.pop('verbose',None)
        if f is None:
            return
        argmin_s21 = np.abs(data).argmin()
        fmin = f.min()
        fmax = f.max()
        f_0_guess = f[argmin_s21] # guess that the resonance is the lowest point
        Q_min = 0.1 * (f_0_guess/(fmax-fmin)) # assume the user isn't trying to fit just a small part of a resonance curve.
        delta_f = np.diff(f) #assume f is sorted
        min_delta_f = delta_f[delta_f>0].min()
        Q_max = f_0_guess/min_delta_f # assume data actually samples the resonance reasonably
        Q_guess = np.sqrt(Q_min*Q_max) # geometric mean, why not?
        Q_e_real_guess = Q_guess/(1-np.abs(data[argmin_s21]))
        if verbose:
            print "fmin=",fmin,"fmax=",fmax,"f_0_guess=",f_0_guess
            print "Qmin=",Q_min,"Q_max=",Q_max,"Q_guess=",Q_guess,"Q_e_real_guess=",Q_e_real_guess
        params = self.make_params(Q=Q_guess, Q_e_real=Q_e_real_guess, Q_e_imag=0, f_0=f_0_guess)
        params['%sQ' % self.prefix].set(min=Q_min, max=Q_max)
        params['%sf_0' % self.prefix].set(min=fmin, max=fmax)
        return update_param_values_and_limits(params,self.prefix,**kwargs)


class InverseLinearResonatorModel(ComplexModel):
    def __init__(self, *args, **kwargs):
        super(InverseLinearResonatorModel, self).__init__(equations.inverse_linear_resonator, *args, **kwargs)

class NonlinearResonatorModel(ComplexModel):
    def __init__(self, *args, **kwargs):
        super(NonlinearResonatorModel, self).__init__(equations.nonlinear_resonator, *args, **kwargs)


