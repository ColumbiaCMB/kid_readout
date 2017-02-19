from __future__ import division

import numpy as np
import lmfit
from distutils.version import StrictVersion

import equations


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
            setattr(pars[pname], attr, val)
    return pars


# Version 0.9.3 of lmfit incorporates changes that allow models that return complex values.
# For earlier versions, we need the following work around
if StrictVersion(lmfit.__version__) < StrictVersion('0.9.3'):
    class ComplexModel(lmfit.model.Model):
        def _residual(self, params, data, weights, **kwargs):
            diff = self.eval(params, **kwargs) - data
            diff_as_ri = diff.astype('complex').view('float')
            if weights is not None:
                weights_as_ri = weights.astype('complex').view('float')
                diff_as_ri *= weights_as_ri
            retval = np.asarray(diff_as_ri).ravel()
            return retval
else:
    class ComplexModel(lmfit.model.Model):
        def eval(self, params=None, **kwargs):
            ind_var = kwargs[self.independent_vars[0]]
            is_scalar = np.isscalar(ind_var)
            kwargs[self.independent_vars[0]] = np.atleast_1d(ind_var)
            result = super(ComplexModel, self).eval(params=params, **kwargs)
            if is_scalar:
                return result[0]
            else:
                return result


# ComplexModel = lmfit.model.Model

class GeneralCableModel(ComplexModel):
    def __init__(self, *args, **kwargs):
        super(GeneralCableModel, self).__init__(equations.general_cable, *args, **kwargs)
        self.set_param_hint('f_min', vary=False)
        self.set_param_hint('phi', min=-np.pi, max=np.pi)
        self.set_param_hint('A_mag', min=0)

    def guess(self, data, f=None, **kwargs):
        verbose = kwargs.pop('verbose', None)
        if f is None:
            return self.make_params(verbose=verbose, **kwargs)
        f_min = f.min()
        f_max = f.max()
        abs_data = np.abs(data)
        A_min = abs_data.min()
        A_max = abs_data.max()
        A_slope, A_offset = np.polyfit(f - f_min, np.abs(data), 1)
        A_mag = A_offset
        A_mag_slope = A_slope / A_mag
        phi_slope, phi_offset = np.polyfit(f - f_min, np.unwrap(np.angle(data)), 1)
        delay = -phi_slope / (2 * np.pi)
        params = self.make_params(delay=delay, phi=phi_offset, f_min=f_min, A_mag=A_mag, A_slope=A_mag_slope)
        params = update_param_values_and_limits(params, self.prefix,
                                                phi_offset_min=phi_offset - np.pi,
                                                phi_offset_max=phi_offset + np.pi,
                                                )
        return update_param_values_and_limits(params, self.prefix, **kwargs)


class LinearResonatorModel(ComplexModel):
    def __init__(self, *args, **kwargs):
        super(LinearResonatorModel, self).__init__(equations.linear_resonator, *args, **kwargs)
        self.set_param_hint('Q', min=0)  # Enforce Q is positive

    def guess(self, data, f=None, **kwargs):
        verbose = kwargs.pop('verbose', None)
        if f is None:
            return self.make_params(verbose=verbose, **kwargs)
        argmin_s21 = np.abs(data).argmin()
        fmin = f.min()
        fmax = f.max()
        f_0_guess = f[argmin_s21]  # guess that the resonance is the lowest point
        Q_min = 0.1 * (f_0_guess / (fmax - fmin))  # assume not trying to fit just a small part of a resonance curve.
        delta_f = np.diff(f)  # assume f is sorted
        min_delta_f = delta_f[delta_f > 0].min()
        Q_max = f_0_guess / min_delta_f  # assume data actually samples the resonance reasonably
        Q_guess = np.sqrt(Q_min * Q_max)  # geometric mean, why not?
        s21_min = np.abs(data[argmin_s21])
        s21_max = np.abs(data).max()
        Q_e_real_guess = Q_guess / (1 - s21_min / s21_max)
        if verbose:
            print "fmin=", fmin, "fmax=", fmax, "f_0_guess=", f_0_guess
            print "Qmin=", Q_min, "Q_max=", Q_max, "Q_guess=", Q_guess, "Q_e_real_guess=", Q_e_real_guess
        params = self.make_params(Q=Q_guess, Q_e_real=Q_e_real_guess, Q_e_imag=0, f_0=f_0_guess)
        params['%sQ' % self.prefix].set(min=Q_min, max=Q_max)
        params['%sf_0' % self.prefix].set(min=fmin, max=fmax)
        params['%sQ_e_real' % self.prefix].set(min=1, max=1e7)
        params['%sQ_e_imag' % self.prefix].set(min=-1e7, max=1e7)
        return update_param_values_and_limits(params, self.prefix, **kwargs)


class InverseLinearResonatorModel(ComplexModel):
    def __init__(self, *args, **kwargs):
        super(InverseLinearResonatorModel, self).__init__(equations.inverse_linear_resonator, *args, **kwargs)


class NonlinearResonatorModel(ComplexModel):
    def __init__(self, *args, **kwargs):
        super(NonlinearResonatorModel, self).__init__(equations.nonlinear_resonator, *args, **kwargs)


class LinearLossResonatorModel(ComplexModel):

    def __init__(self, *args, **kwargs):
        super(LinearLossResonatorModel, self).__init__(equations.linear_loss_resonator, *args, **kwargs)

    def guess(self, data=None, f=None, **kwargs):
        f_0_guess = f[np.argmin(np.abs(data))]  # guess that the resonance is the lowest point
        width = f.size // 10
        gaussian = np.exp(-np.linspace(-4, 4, width) ** 2)
        gaussian /= np.sum(gaussian)  # not necessary
        smoothed = np.convolve(gaussian, abs(data), mode='same')
        derivative = np.convolve(np.array([1, -1]), smoothed, mode='same')
        # Exclude the edges, which are affected by zero padding.
        linewidth = 1 / 2 * (f[np.argmax(derivative[width:-width])] - f[np.argmin(derivative[width:-width])])
        i_plus_c = linewidth / f_0_guess
        i_over_c = 1 / (1 / np.min(np.abs(data)) - 1)
        loss_c_guess = i_plus_c / (1 + i_over_c)
        loss_i_guess = i_plus_c * i_over_c / (1 + i_over_c)
        params = self.make_params(f_0=f_0_guess, loss_i=loss_i_guess, loss_c=loss_c_guess, asymmetry=0)
        params['{}f_0'.format(self.prefix)].set(min=f.min(), max=f.max())
        params['{}loss_i'.format(self.prefix)].set(min=0, max=1)
        params['{}loss_c'.format(self.prefix)].set(min=0, max=1)
        params['{}asymmetry'.format(self.prefix)].set(min=-10, max=10)
        return update_param_values_and_limits(params, self.prefix, **kwargs)

