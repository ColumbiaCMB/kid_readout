from __future__ import division
import warnings
import numpy as np
import scipy.stats
from lmfit.ui import fitter


class Resonator(fitter.Fitter):
    """
    This class represents a single resonator. All of the model-dependent behavior is contained in functions that are
    supplied to the class. There is a little bit of Python magic that allows for easy access to the fit parameters
    and functions of only the fit parameters.

    The idea is that, given sweep data freq and s21,
    r = Resonator(freq, s21)
    should just work. Modify the import statements to change the default model, guess, and functions of the parameters.
    """

    def __init__(self, freq, s21,
                 model=default_model, guess=default_guess, functions=default_functions,
                 mask=None, errors=None):
        """
        Fit a resonator using the given model.

        f: the frequencies used in a sweep.

        s21: the complex S_21 data taken at the given frequencies.

        model: a function S_21(params, f) that returns the modeled values of S_21.

        guess: a function guess(f, s21) that returns a good-enough initial guess at all of the fit parameters.

        functions: a dictionary that maps keys that are valid Python variables to functions that take a Parameters
        object as their only argument.

        mask: a boolean array of the same length as f and s21; only points f[mask] and s21[mask] are used to fit the
        data and the default is to use all data; use this to exclude glitches or resonances other than the desired one.
        """
        if not np.iscomplexobj(s21):
            raise TypeError("Resonator s21 must be complex.")
        if errors is not None and not np.iscomplexobj(errors):
            raise TypeError("Resonator s21 errors must be complex.")
        super(Resonator, self).__init__(freq, s21,
                                        model=model, guess=guess, functions=functions, mask=mask, errors=errors)
        self.freq_data = self.x_data
        self.s21_data = self.y_data
        self.freq_units_MHz = self.freq_data.max() < 1e6

    # todo: this should be in the same module as the functions
    # todo: make it clear whether one should multiply or divide by the return value to normalize
    def get_normalization(self, freq, remove_amplitude=True, remove_delay=True, remove_phase=True):
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
            normalization /= self.A_mag
        if remove_phase:
            phi = self.phi + self.A_phase
        else:
            phi = 0
        if remove_delay:
            delay = self.delay
        else:
            delay = 0
        normalization *= np.exp(1j * (2 * np.pi * (freq - self.f_phi) * delay - phi))
        return normalization

    def normalize(self, freq, s21_raw, remove_amplitude=True, remove_delay=True, remove_phase=True):
        """
        Normalize s21 data, removing arbitrary ampltude, delay, and phase terms
        
        freq : float or array of floats
            frequency in same units as the model was built with, at which normalization should be computed
            
        s21_raw : complex or array of complex
            raw s21 data which should be normalized
        """
        normalization = self.get_normalization(freq, remove_amplitude=remove_amplitude, remove_delay=remove_delay,
                                               remove_phase=remove_phase)
        return s21_raw * normalization

    def normalized_model(self, freq, remove_amplitude=True, remove_delay=True, remove_phase=True):
        """
        Evaluate the model, removing arbitrary ampltude, delay, and phase terms
        
        freq : float or array of floats
            frequency in same units as the model was built with, at which normalized model should be evaluated
        """
        return self.normalize(freq, self.model(x=freq), remove_amplitude=remove_amplitude, remove_delay=remove_delay,
                              remove_phase=remove_phase)

    def approx_normalized_gradient(self, freq):
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
        f1 = freq + df
        y = self.normalized_model(freq)
        y1 = self.normalized_model(f1)
        gradient = y1 - y  # division by 1 Hz is implied.
        return gradient

    def project_s21_to_delta_freq(self, freq, s21, use_data_mean=True, s21_already_normalized=False):
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
            normalized_s21 = self.normalize(freq, s21)
        if use_data_mean:
            mean_ = normalized_s21.mean()
        else:
            mean_ = self.normalized_model(freq)
        gradient = self.approx_normalized_gradient(freq)
        delta_freq = (normalized_s21 - mean_) / gradient
        return delta_freq

    # todo: I and Q are supposed to be power spectra, not sweep data. Remove? -DF
    def convert_s21_to_freq_fluctuation(self, freq, s21):
        """
        Use formula in Phil's LTD paper to convert S21 data to frequency fluctuations.

        The result of this is the same as Re(S21/(dS21/df)), so the same as self.project_s21_to_delta_freq().real

        freq : float
            frequency in same units as the model was built with, at which the S21 data was measured.

        s21 : complex or array of complex
            Raw S21 data measured at the indicated frequency
        """
        normalized_s21 = self.normalize(freq, s21)
        gradient = self.approx_normalized_gradient(freq)
        # using notation from Phil's LTD paper
        I = normalized_s21.real
        Q = normalized_s21.imag
        dIdf = gradient.real
        dQdf = gradient.imag
        ef = (I * dIdf + Q * dQdf) / (dIdf ** 2 + dQdf ** 2)
        return ef

