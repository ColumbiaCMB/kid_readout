from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import lmfit
import scipy.stats
import scipy.optimize
minimize = lmfit.minimize

from fitter import Fitter

# To use different defaults, change these three import statements.
from kid_readout.analysis.khalil import delayed_generic_s21 as default_model
from kid_readout.analysis.khalil import delayed_generic_guess as default_guess
from kid_readout.analysis.khalil import generic_functions as default_functions
from kid_readout.analysis.khalil import bifurcation_s21, bifurcation_guess

def fit_resonator(freq, s21, mask= None, errors=None, weight_by_errors=True, min_a = 0.08, fstat_thresh = 0.999):
    rr = Resonator(freq, s21, mask=mask, errors=errors, weight_by_errors=weight_by_errors)
    bif = Resonator(freq, s21, mask=mask, errors=errors, weight_by_errors=weight_by_errors, 
                    guess = bifurcation_guess, model = bifurcation_s21)
    fval = scipy.stats.f_value(np.sum(np.abs(rr.residual())**2),
                                np.sum(np.abs(bif.residual())**2),
                                rr.result.nfree, bif.result.nfree)
    fstat = scipy.stats.distributions.f.cdf(fval,rr.result.nfree,bif.result.nfree)
    aval = bif.result.params['a'].value
    aerr = bif.result.params['a'].stderr
    reasons = []
    if aval <= aerr:
        prefer_bif = False
        reasons.append("Error on bifurcation parameter exceeds fitted value")
    else:
        if aval < min_a:
            prefer_bif = False
            reasons.append("Bifurcation parameter %f is less than minimum required %f" % (aval,min_a))
        else:
            #not sure this is working right, so leave it out for now.
            if False:#fstat < fstat_thresh:
                prefer_bif = False
                reasons.append("F-statistic %f is less than threshold %f" % (fstat,fstat_thresh))
            else:
                prefer_bif = True
    if not prefer_bif:
        print "Not using bifurcation model because:",(','.join(reasons))
    return rr,bif,prefer_bif
    
def fit_best_resonator(*args,**kwargs):
    rr,bif,prefer_bif = fit_resonator(*args,**kwargs)
    return (rr,bif)[prefer_bif]

class Resonator(Fitter):
    """
    This class represents a single resonator. All of the
    model-dependent behavior is contained in functions that are
    supplied to the class. There is a little bit of Python magic that
    allows for easy access to the fit parameters and functions of only
    the fit parameters.

    The idea is that, given sweep data f and s21,
    r = Resonator(f, s21)
    should just work. Modify the import statements to change the
    defaults.
    """
   
    def __init__(self, f, data, model=default_model, guess=default_guess, functions=default_functions, 
                 mask=None, errors=None, weight_by_errors=True):
        """
        Instantiate a resonator using our current best model.
        Parameter model is a function S_21(params, f) that returns the
        modeled values of S_21.
        Parameter guess is a function guess(f, data) that returns a
        good-enough initial guess at all of the fit parameters.
        Parameter functions is a dictionary that maps keys that are
        valid Python variables to functions that take a Parameters
        object as their only argument.
        Parameter mask is a boolean array of the same length as f and
        data; only points f[mask] and data[mask] are used to fit the
        data. The default is to use all data. Use this to exclude
        glitches or resonances other than the desired one.
        """
        super(Resonator,self).__init__(f,data,model=model,guess=guess,functions=functions,mask=mask,
                                       errors=errors,weight_by_errors=weight_by_errors)

