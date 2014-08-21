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

def fit_resonator(freq, s21, mask= None, errors=None, weight_by_errors=True, min_a = 0.08, fstat_thresh = 0.999,
                  delay_estimate = None, verbose=False):
    if delay_estimate is not None:
        def my_default_guess(f,data):
            params = default_guess(f,data)
            params['delay'].value = delay_estimate
            return params
    else:
        my_default_guess = default_guess
    rr = Resonator(freq, s21, mask=mask, errors=errors, weight_by_errors=weight_by_errors,guess=my_default_guess)
    
    if delay_estimate is not None:
        def my_bifurcation_guess(f,data):
            params = bifurcation_guess(f,data)
            params['delay'].value = delay_estimate
            return params
    else:
        my_bifurcation_guess = bifurcation_guess
    
    bif = Resonator(freq, s21, mask=mask, errors=errors, weight_by_errors=weight_by_errors, 
                    guess = my_bifurcation_guess, model = bifurcation_s21)
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
    if verbose and not prefer_bif:
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
        if not np.iscomplexobj(data):
            raise TypeError("Resonator data should always be complex, but got real values")
        if errors is not None:
            if not np.iscomplexobj(errors):
                errors = errors*(1+1j)  # ensure errors is complex
        super(Resonator,self).__init__(f,data,model=model,guess=guess,functions=functions,mask=mask,
                                       errors=errors,weight_by_errors=weight_by_errors)
        if self.x_data.max() < 1e6:
            self.freq_units_MHz = True
        else:
            self.freq_units_MHz = False
        self.freq_data = self.x_data
        self.s21_data = self.y_data

    def get_normalization(self, freq, remove_amplitude = True, remove_delay = True, remove_phase = True):
        """
        return the complex factor that removes the arbitrary amplitude, cable delay, and phase from the resonator fit
        
        freq : float or array of floats
            frequency in same units as the model was built with, at which normalization should be computed
            
        remove_amplitude : bool, default True
            include arbitrary amplitude correction
            
        remove_delay : bool, default True
            include cable delay correction
        
        remove_phase : bool, default True
            include arbitrary phase offset correction
        """
        normalization = 1.0
        if remove_amplitude:
            normalization *= 1.0/self.A_mag
        if remove_phase:
            phi = self.phi + self.A_phase
        else:
            phi = 0
        if remove_delay:
            delay = self.delay
        else:
            delay = 0
        normalization *= np.exp(1j*(2*np.pi*(freq-self.f_phi)*delay - phi))
        return normalization
    
    def normalize(self, freq, s21_raw, remove_amplitude = True, remove_delay = True, remove_phase = True):
        """
        Normalize s21 data, removing arbitrary ampltude, delay, and phase terms
        
        freq : float or array of floats
            frequency in same units as the model was built with, at which normalization should be computed
            
        s21_raw : complex or array of complex
            raw s21 data which should be normalized
        """
        normalization = self.get_normalization(freq, remove_amplitude=remove_amplitude, remove_delay=remove_delay, 
                                               remove_phase= remove_phase)
        return s21_raw*normalization
        
    def normalized_model(self,freq,remove_amplitude = True, remove_delay = True, remove_phase = True):
        """
        Evaluate the model, removing arbitrary ampltude, delay, and phase terms
        
        freq : float or array of floats
            frequency in same units as the model was built with, at which normalized model should be evaluated
        """
        return self.normalize(freq, self.model(x=freq),remove_amplitude=remove_amplitude, remove_delay=remove_delay, 
                                               remove_phase= remove_phase)
        
    def approx_normalized_gradient(self,freq):
        """
        Calculate the approximate gradient of the normalized model dS21/df at the given frequency. 
        
        The units will be S21 / Hz
        
        freq : float or array of floats
            frequency in same units as the model was built with, at which normalized gradient should be evaluated
        """
        if self.freq_units_MHz:
            df = 1e-6  # 1 Hz
        else:
            df = 1.0
        f1 = freq+df
        y = self.normalized_model(freq)
        y1 = self.normalized_model(f1)
        gradient = y1-y  # division by 1 Hz is implied.
        return gradient

    def project_s21_to_delta_freq(self,freq,s21,use_data_mean=True,s21_already_normalized=False):
        """
        Project s21 data onto the orthogonal vectors tangent and perpendicular to the resonance circle at the 
        measurement frequency
        
        This results in complex data with the real part projected along the frequency direction (in Hz) and the 
        imaginary part projected along the dissipation direction (also in pseudo-Hz).
        
        freq : float
            frequency in same units as the model was built with, at which the S21 data was measured.
        
        s21 : complex or array of complex
            Raw S21 data measured at the indicated frequency
        
        use_data_mean : bool, default True
            if true, center the data on the mean of the data before projecting.
            if false, center the data on the value of the model evaluated at the measurement frequency.
            
        s21_already_normalized : bool, default False
            if true, the s21 data has already been normalized
            if false, first normalize the s21 data
        """
        if s21_already_normalized:
            normalized_s21 = s21
        else:
            normalized_s21 = self.normalize(freq,s21)
        if use_data_mean:
            mean_ = normalized_s21.mean()
        else:
            mean_ = self.normalized_model(freq)
        gradient = self.approx_normalized_gradient(freq)
        delta_freq = (normalized_s21-mean_)/gradient
        return delta_freq
        
    def convert_s21_to_freq_fluctuation(self,freq,s21):
        """
        Use formula in Phil's LTD paper to convert S21 data to frequency fluctuations.
        
        The result of this is the same as Re(S21/(dS21/df)), so the same as self.project_s21_to_delta_freq().real
        
        freq : float
            frequency in same units as the model was built with, at which the S21 data was measured.
        
        s21 : complex or array of complex
            Raw S21 data measured at the indicated frequency         
        """
        normalized_s21 = self.normalize(freq,s21)
        gradient = self.approx_normalized_gradient(freq)
        # using notation from Phil's LTD paper
        I = normalized_s21.real
        Q = normalized_s21.imag
        dIdf = gradient.real
        dQdf = gradient.imag
        ef = (I*dIdf + Q*dQdf)/(dIdf**2 + dQdf**2)
        return ef
