from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize

class GenericResonator(object):
    """
    This class represents a single resonator.

    The idea is that all of the model-dependent behavior is contained
    in functions that are supplied to the class. There is a little bit
    of Python magic that allows for easy access to the fit parameters
    and functions of only the fit parameters.
    """

    def __init__(self, model, functions={}):
        """
        Parameter model is a function S_21(params, f)
        Parameter functions is a dictionary that maps keys that are
        valid Python variables to functions that take a Parameters
        object as their only argument.
        """
        self.model = model
        self._functions = functions

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
    
    def fit(self, f, data, initial):
        """
        Fit the S_21 data using the model given at instantiation.
        Parameter initial is a Parameters object containing initial
        values. It is modified by lmfit.
        """
        self.f = f
        self.data = data
        self.result = minimize(self.residual,
                               initial,
                               args=(f, data))
                               
    def residual(self, params, f, data):
        """
        This is the residual function used in by lmfit.
        """
        return np.abs(data - self.model(params, f))

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
