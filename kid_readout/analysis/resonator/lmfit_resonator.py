from __future__ import division
import warnings

import lmfit
import numpy as np
import scipy.stats
from lmfit.ui import Fitter
from kid_readout.analysis.resonator import lmfit_models

class FitterWithAttributeAccess(Fitter):
    def __getattr__(self, attr):
        """
        This allows instances to have a consistent interface while using different underlying models.

        Return a fit parameter or value derived from the fit parameters.
        """
        if attr.endswith('_error'):
            name = attr[:-len('_error')]
            try:
                return self.current_result.params[name].stderr
            except KeyError:
                print "couldnt find error for ",name,"in self.current_result"
                pass
        try:
            return self.current_params[attr].value
        except KeyError:
            raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, attr))

    def __dir__(self):
        return (dir(super(Fitter, self)) +
                self.__dict__.keys() +
                self.current_params.keys() +
                [name + '_error' for name in self.current_params.keys()])

class BaseResonator(FitterWithAttributeAccess):
    def __init__(self, frequency, s21, errors, model, **kwargs):
        """
        General resonator fitting class.

        Parameters
        ----------
        frequency: array of floats
            Frequencies at which data was measured
        s21: array of complex
            measured S21 data
        errors: None or array of complex
            errors on the real and imaginary parts of the s21 data. None means use no errors
        model: an lmfit.Model
            the model for the resonator. Common models are provided in lmfit_models. If the model is composite,
            it is assumed to be of the form background * target, where target is the model of the target resonator
            itself, and background represents any other nuisance effects (cable delay, other adjacent resonators etc.)
        kwargs:
            passed on to model.fit
        """
        if not np.iscomplexobj(s21):
            raise TypeError("Resonator s21 must be complex.")
        if errors is not None and not np.iscomplexobj(errors):
            raise TypeError("Resonator s21 errors must be complex.")
        if errors is None:
            weights = None
        else:
            weights = 1/errors.real + 1j/errors.imag
        # kwargs get passed from Fitter to Model.fit directly. Signature is:
        #    def fit(self, data, params=None, weights=None, method='leastsq',
        #            iter_cb=None, scale_covar=True, verbose=False, fit_kws=None, **kwargs):
        super(BaseResonator, self).__init__(data=s21, f=frequency,
                                        model=model, weights=weights, **kwargs)
        self.frequency = frequency

        self.fit()

    @property
    def s21(self):
        return self._data

    @property
    def target(self):
        if isinstance(self.model,lmfit.model.CompositeModel):
            return self.model.right
        else:
            return self.model

    def target_s21(self, frequency=None):
        if frequency is None:
            frequency = self.frequency
        return self.target.eval(self.current_result.params, f=frequency)

    @property
    def background(self):
        if isinstance(self.model,lmfit.model.CompositeModel):
            return self.model.left
        else:
            return None

    def background_s21(self, frequency=None):
        if frequency is None:
            frequency = self.frequency
        background = self.background
        if background is None:
            if np.isscalar(frequency):
                return 1.0
            else:
                return np.ones_like(frequency)
        else:
            return background.eval(self.current_result.params,f=frequency)

    def remove_background(self, frequency, s21_raw):
        """
        Normalize s21 data, removing arbitrary ampltude, delay, and phase terms
        
        frequency : float or array of floats
            frequency in same units as the model was built with, at which normalization should be computed
            
        s21_raw : complex or array of complex
            raw s21 data which should be normalized
        """
        normalization = self.background_s21(frequency)

        return s21_raw / normalization


    def approximate_target_gradient(self, frequency, delta_f=1.0):
        """
        Calculate the approximate gradient of the target model dS21/df at the given frequency.
        
        The units will be S21 / Hz
        
        frequency : float or array of floats
            frequency in same units as the model was built with, at which normalized gradient should be evaluated
        delta_f : float default 1
            frequency offset used to calculate gradient
        """
        f1 = frequency + delta_f
        y = self.target_s21(frequency)
        y1 = self.target_s21(f1)
        gradient = (y1 - y)/delta_f  # division by 1 Hz is implied.
        return gradient

    def project_s21_to_frequency(self, frequency, s21, use_data_mean=True, s21_already_normalized=False):
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
            normalized_s21 = self.remove_background(frequency, s21)
        if use_data_mean:
            mean_ = normalized_s21.mean()
        else:
            mean_ = self.target_s21(frequency)
        gradient = self.approximate_target_gradient(frequency)
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


class LinearResonator(BaseResonator):
    def __init__(self, frequency, s21, errors, **kwargs):
        super(LinearResonator,self).__init__(frequency=frequency, s21=s21, errors=errors,
                                             model = lmfit_models.LinearResonatorModel, **kwargs)

class LinearResonatorWithCable(BaseResonator):
    def __init__(self, frequency, s21, errors, **kwargs):
        super(LinearResonator,self).__init__(frequency=frequency, s21=s21, errors=errors,
                                             model = (lmfit_models.LinearResonatorModel() *
                                                      lmfit_models.GeneralCableModel()), **kwargs)

