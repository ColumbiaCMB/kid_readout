from __future__ import division
from collections import namedtuple

import lmfit
import numpy as np

from kid_readout.analysis.lmfit_fitter import FitterWithAttributeAccess
from kid_readout.analysis.resonator import lmfit_models


# This is a simple format for extracted resonator data, useful for plotting.
# The _data arrays are the raw frequency and s21 data;
# The _model arrays are the model frequency and evaluated s21
# The _0 points are the resonance frequency (float) and the model (complex) evaluated there.
ResonatorData = namedtuple('ResonatorData', field_names=['f_data', 's21_data',
                                                         'f_model', 's21_model',
                                                         'f_0', 's21_0'])


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
        model: lmfit.Model
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
            weights = 1 / errors.real + 1j / errors.imag
        nanmask = np.isfinite(frequency)
        nanmask = nanmask & np.isfinite(s21)
        if weights is not None:
            nanmask = nanmask & np.isfinite(weights)
            weights = weights[nanmask]
        frequency = frequency[nanmask]
        s21 = s21[nanmask]
        if s21.shape[0] < 1:
            raise ValueError("After masking NaNs, there is no data left to fit!")
        # kwargs get passed from Fitter to Model.fit directly. Signature is:
        #    def fit(self, data, params=None, weights=None, method='leastsq',
        #            iter_cb=None, scale_covar=True, verbose=False, fit_kws=None, **kwargs):
        super(BaseResonator, self).__init__(data=s21, f=frequency,
                                            model=model, weights=weights, **kwargs)
        self.frequency = frequency
        self.errors = errors
        self.weights = weights
        self.fit()

    @property
    def Q_i(self):
        return 1 / (1 / self.Q - np.real(1 / self.Q_e))

    @property
    def Q_e(self):
        return self.Q_e_real + 1j * self.Q_e_imag

    @property
    def s21(self):
        return self._data

    @property
    def target(self):
        if isinstance(self.model, lmfit.model.CompositeModel):
            return self.model.right
        else:
            return self.model

    def target_s21(self, frequency=None):
        if frequency is None:
            frequency = self.frequency
        return self.target.eval(self.current_result.params, f=frequency)

    @property
    def background(self):
        if isinstance(self.model, lmfit.model.CompositeModel):
            return self.model.left
        else:
            return None

    def eval(self, frequency=None, params=None):
        if params is None:
            params = self.current_params
        if frequency is None:
            frequency = self.frequency
        return self.model.eval(f=frequency, params=params)

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
            return background.eval(self.current_result.params, f=frequency)

    def remove_background(self, frequency, s21_raw):
        """
        Normalize s21 data, removing arbitrary amplitude, delay, and phase terms
        
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
        gradient = (y1 - y) / delta_f  # division by 1 Hz is implied.
        return gradient

    def extract(self, normalize=False, num_model_points=1000):
        """
        Extract and return a namedtuple containing three pairs of frequency and s21 values corresponding to (1) the
        input data array, (2) the evaluated model array, and (3) a single point at the resonance frequency.

        Parameters
        ----------
        normalize : bool, default False
            If True, return all s21 values with the background model removed.
        num_model_points : int, default 1000
            The number of data points to use in evaluating the model over the span of the data frequencies.

        Returns
        -------
        ResonatorData, a namedtuple defined in this module
        """
        f_data = self.frequency.copy()
        s21_data = self.data.copy()
        f_model = np.linspace(f_data.min(), f_data.max(), num_model_points)
        s21_model = self.model.eval(params=self.current_params, f=f_model)
        f_0 = self.f_0
        s21_0 = self.model.eval(params=self.current_params, f=f_0)
        if normalize:
            s21_data = self.remove_background(frequency=f_data, s21_raw=s21_data)
            s21_model = self.remove_background(frequency=f_model, s21_raw=s21_model)
            s21_0 = self.remove_background(frequency=f_0, s21_raw=s21_0)
        return ResonatorData(f_data, s21_data, f_model, s21_model, f_0, s21_0)

    def invert_raw(self, frequency, s21_raw):
        """
        Invert the resonator model and return the time-ordered resonator parameters x(t) and Q_i^[-1}(t) that correspond
        to the given time-ordered non-normalized s21 data; see invert().

        Parameters
        ----------
        frequency : float
            The tone frequency used to record the time-ordered data.
        s21_raw : ndarray (complex)
            The time-ordered raw s21 data.

        Returns
        -------
        ndarray (real)
            The time-ordered values of the fractional frequency detuning.
        ndarray (real)
            The time-ordered values of the inverse internal quality factor.

        """
        return self.invert(self.remove_background(frequency=frequency, s21_raw=s21_raw))

    # ToDo: finish
    def invert(self, s21_normalized):
        """
        Invert the resonator model and return the time-ordered resonator parameters x(t) and Q_i^[-1}(t) that correspond
        to the given time-ordered s21 data. These data should be normalized to equal 1 far from the resonance.

        Parameters
        ----------
        s21_normalized : ndarray (complex)
            The time-ordered s21 data, normalized to 1 off-resonance.

        Returns
        -------
        ndarray (real)
            The time-ordered values of the fractional frequency detuning x.
        ndarray (real)
            The time-ordered values of the inverse internal quality factor q = 1 / Q_i.
        """
        raise NotImplementedError("Subclasses should implement this using their parameters.")

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
        super(LinearResonator, self).__init__(frequency=frequency, s21=s21, errors=errors,
                                              model=lmfit_models.LinearResonatorModel, **kwargs)


_general_cable_model = lmfit_models.GeneralCableModel()
_linear_resonator_model = lmfit_models.LinearResonatorModel()
_linear_resonator_with_cable = (_general_cable_model * _linear_resonator_model)


def _linear_resonator_with_cable_guess(data, f=None, **kwargs):
    cable_params = _general_cable_model.guess(data=data, f=f, **kwargs)
    resonator_params = _linear_resonator_model.guess(data=data, f=f, **kwargs)
    cable_params.update(resonator_params)
    return cable_params


_linear_resonator_with_cable.guess = _linear_resonator_with_cable_guess


class LinearResonatorWithCable(BaseResonator):
    def __init__(self, frequency, s21, errors, **kwargs):
        super(LinearResonatorWithCable, self).__init__(frequency=frequency, s21=s21, errors=errors,
                                                       model=_linear_resonator_with_cable, **kwargs)

    def invert(self, s21_normalized):
        c = 1 / self.Q_e  # c is the inverse of the complex couping quality factor.
        z = c / (1 - s21_normalized)
        q = z.real - c.real
        x = z.imag / 2  # This factor of two means S_xx = S_qq / 4 when amplifier-noise dominated.
        return x, q


_background_resonator_model = lmfit_models.LinearResonatorModel(prefix='bg_')
_foreground_resonator_model = lmfit_models.LinearResonatorModel(prefix='fg_')
_colliding_linear_resonators_with_cable = ((_general_cable_model * _background_resonator_model)
                                           * _foreground_resonator_model)


def _colliding_linear_resonators_with_cable_guess(data, f=None, **kwargs):
    cable_params = _general_cable_model.guess(data=data, f=f, **kwargs)
    resonator_params = _foreground_resonator_model.guess(data=data, f=f, **kwargs)
    bg_resonator_params = _background_resonator_model.guess(data=data, f=f, **kwargs)
    cable_params.update(resonator_params)
    cable_params.update(bg_resonator_params)
    return cable_params


_colliding_linear_resonators_with_cable.guess = _colliding_linear_resonators_with_cable_guess


class CollidingLinearResonatorsWithCable(BaseResonator):
    def __init__(self, frequency, s21, errors, **kwargs):
        super(CollidingLinearResonatorsWithCable, self).__init__(frequency=frequency, s21=s21, errors=errors,
                                                                 model=_colliding_linear_resonators_with_cable,
                                                                 **kwargs)


class GeneralCable(FitterWithAttributeAccess):
    def __init__(self, frequency, s21, errors, **kwargs):
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
        kwargs:
            passed on to model.fit
        """
        if not np.iscomplexobj(s21):
            raise TypeError("S21 must be complex.")
        if errors is not None and not np.iscomplexobj(errors):
            raise TypeError("S21 errors must be complex.")
        if errors is None:
            weights = None
        else:
            weights = 1 / errors.real + 1j / errors.imag
        # kwargs get passed from Fitter to Model.fit directly. Signature is:
        #    def fit(self, data, params=None, weights=None, method='leastsq',
        #            iter_cb=None, scale_covar=True, verbose=False, fit_kws=None, **kwargs):
        super(GeneralCable, self).__init__(data=s21, f=frequency,
                                           model=lmfit_models.GeneralCableModel, weights=weights, **kwargs)
        self.frequency = frequency
        self.errors = errors
        self.weights = weights
        self.fit()


# This is a copy of the code surrounding LinearResonatorWithCable
_linear_loss_resonator_model = lmfit_models.LinearLossResonatorModel()
_linear_loss_resonator_with_cable = _general_cable_model * _linear_loss_resonator_model


def _linear_loss_resonator_with_cable_guess(data, f=None, **kwargs):
    cable_params = _general_cable_model.guess(data=data, f=f, **kwargs)
    resonator_params = _linear_loss_resonator_model.guess(data=data, f=f, **kwargs)
    cable_params.update(resonator_params)
    return cable_params


_linear_loss_resonator_with_cable.guess = _linear_loss_resonator_with_cable_guess


class LinearLossResonatorWithCable(BaseResonator):

    def __init__(self, frequency, s21, errors, **kwargs):
        super(LinearLossResonatorWithCable, self).__init__(frequency=frequency, s21=s21, errors=errors,
                                                           model=_linear_loss_resonator_with_cable, **kwargs)

    def invert(self, s21_normalized):
        z = self.loss_c * (1j * self.asymmetry + s21_normalized) / (1 - s21_normalized)
        x = z.imag / 2
        q = z.real
        return x, q

    # These properties improve compatibility with other models.

    @property
    def Q(self):
        return 1 / (self.loss_i + self.loss_c)

    @property
    def Q_e(self):
        return 1 / (self.loss_c * (1 + 1j * self.asymmetry))

    @property
    def Q_i(self):
        return 1 / self.loss_i
