from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import lmfit
minimize = lmfit.minimize

# To use different defaults, change these three import statements.
from kid_readout.analysis.khalil import delayed_generic_s21 as default_model
from kid_readout.analysis.khalil import delayed_generic_guess as default_guess
from kid_readout.analysis.khalil import generic_functions as default_functions

class Resonator(object):
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
        self.f = f
        self.data = data
        self._model = model
        self._functions = functions
        if mask is None:
            if errors is None:
                self.mask = np.ones_like(data).astype(np.bool)
            else:
                self.mask = abs(errors) < np.median(abs(errors))*3
        else:
            self.mask = mask
        self.errors = errors
        self.weight_by_errors = weight_by_errors
        self.fit(guess(f[self.mask], data[self.mask]))

    def __getattr__(self, attr):
        """
        Return a fit parameter or value derived from the fit
        parameters. This allows syntax like r.Q_i after a fit has been
        performed.
        """
        try:
            return self.result.params[attr].value
        except KeyError:
            pass
        try:
            return self._functions[attr](self.result.params)
        except KeyError:
            raise AttributeError("'{0}' object has no attribute '{1}'".format(self.__class__.__name__, attr))

    def __dir__(self):
        return (dir(super(Resonator, self)) +
                self.__dict__.keys() +
                self.result.params.keys() +
                self._functions.keys())
    
    def fit(self, initial):
        """
        Fit S_21 using the data and model given at
        instantiation. Parameter initial is a Parameters object
        containing initial values. It is modified by lmfit.
        """
        self.result = minimize(self.residual, initial,ftol=1e-6)
                               
    def residual(self, params):
        """
        This is the residual function used by lmfit. Only data where
        mask is True is used for the fit.
        
        Note that the residual needs to be purely real, and should *not* include abs.
        The minimizer needs the signs of the residuals to properly evaluate the gradients.
        """
        # in the following, .view('float') will take a length N complex array 
        # and turn it into a length 2*N float array.
        
        if self.errors is None or not self.weight_by_errors:
            return ((self.data[self.mask] - self.model(params)[self.mask]).view('float'))
        else:
            errors = self.errors[self.mask]
            if not np.iscomplexobj(errors):
                errors = errors + 1j*errors
            return ((self.data[self.mask] - self.model(params)[self.mask]).view('float'))/errors.view('float')
                

    def model(self, params=None, f=None):
        """
        Return the model evaluated with the given parameters at the
        given frequencies. Defaults are the fit-derived params and the
        frequencies corresponding to the data.
        """
        if params is None:
            params = self.result.params
        if f is None:
            f = self.f
        return self._model(params, f)
