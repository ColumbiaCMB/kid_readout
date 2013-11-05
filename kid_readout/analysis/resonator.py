from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize

from kid_readout.analysis.generic_resonator import GenericResonator
# To use different defaults, change these three import statements.
# Note that the current model doesn't include a cable delay, so feel
# free to upgrade it to something better.
from kid_readout.analysis.khalil import generic_s21 as default_model
from kid_readout.analysis.khalil import generic_guess as default_guess
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
   
    def __init__(self, f, data, model=default_model, guess=default_guess, functions=default_functions):
        """
        Instantiate a resonator using our current best model.
        Parameter model is a function S_21(params, f) that returns the modeled values of S_21.
        Parameter guess is a function guess(f, data) that returns a good-enough initial guess at all of the fit parameters.
        Parameter functions is a dictionary that maps keys that are
        valid Python variables to functions that take a Parameters
        object as their only argument. This is used 

        """
        self.f = f
        self.data = data
        self.model = model
        self._functions = functions
        self.fit(guess(f, data))

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
            raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, attr))

    def __dir__(self):
        return (dir(super(Resonator, self)) +
                self.result.params.keys() +
                self._functions.keys())
    
    def fit(self, initial):
        """
        Fit S_21 using the data and model given at
        instantiation. Parameter initial is a Parameters object
        containing initial values. It is modified by lmfit.
        """
        self.result = minimize(self.residual, initial)
                               
    def residual(self, params):
        """
        This is the residual function used by lmfit.
        """
        return np.abs(self.data - self.model(params, self.f))

    def plot(self):
        """
        Plot the data, fit, and f_0.
        """
        model = self.model(self.result.params, self.f)
        model_0 = model[np.argmin(abs(self.f_0 - self.f))]
        interactive = plt.isinteractive()
        plt.ioff()
        fig = plt.figure()
        plt.plot(self.f, 20*np.log10(abs(self.data)), '.b', label='data')
        plt.plot(self.f, 20*np.log10(abs(model)), '-g', label='fit')
        plt.plot(self.f_0, 20*np.log10(abs(model_0)), '.r', label='f_0')
        plt.xlabel('frequency [Hz]')
        plt.ylabel('|S_21| [dB]')
        plt.legend(loc='lower right')
        if interactive:
            plt.ion()
            plt.show()
        return fig
