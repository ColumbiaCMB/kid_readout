"""
This module uses the Khalil and Swenson models but fits directly to the inverse quality factors, since these are actually more
useful.


"""
from __future__ import division
import numpy as np
from scipy.special import cbrt
import lmfit
from kid_readout.analysis import khalil


def s21(params, f):
    A = (params['A_mag'].value *
         np.exp(1j * params['A_phase'].value))
    f_r = params['f_r'].value
    i = params['i'].value
    c = params['c'].value
    c_imag = params['c_imag'].value
    a = params['a'].value
    Q_r = (i + c)**-1
    y_0 = (f / f_r - 1) / Q_r
    y = (y_0 / 3 +
         (y_0**2 / 9 - 1 / 12) / cbrt(a / 8 + y_0 / 12 + np.sqrt(
             (y_0**3 / 27 + y_0 / 12 + a / 8)**2 - (y_0**2 / 9 - 1 / 12)**3) + y_0**3 / 27) +
         cbrt(a / 8 + y_0 / 12 + np.sqrt(
             (y_0**3 / 27 + y_0 / 12 + a / 8)**2 - (y_0**2 / 9 - 1 / 12)**3) + y_0**3 / 27))
    x = y / Q_r
    return A * (1 - ((c + 1j * c_imag) /
                     (i + c + 2j * x)))


def guess_normalized(f, s21_normalized):
    """
    Use the linewidth and the transmission ratio on and off resonance to guess the initial values.  Estimate the
    linewidth by smoothing then looking for the extrema of the first derivative. This may fail if the resonance is
    very close to the edge of the data.

    This function expects the s21 data to be normalized.
    """
    p = lmfit.Parameters()
    # Allow f_r to vary by +/- the bandwidth over which we have data
    bandwidth = f.max() - f.min()
    f_r = f[np.argmin(abs(s21_normalized))]
    p.add('f_r', value=f_r,
          min=f.min() - bandwidth, max=f.max() + bandwidth)
    width = int(f.size / 10)
    gaussian = np.exp(-np.linspace(-4, 4, width)**2)
    gaussian /= np.sum(gaussian)  # not necessary
    smoothed = np.convolve(gaussian, abs(s21_normalized), mode='same')
    derivative = np.convolve(np.array([1, -1]), smoothed, mode='same')
    # Exclude the edges, which are affected by zero padding.
    linewidth = (f[np.argmax(derivative[width:-width])] -
                 f[np.argmin(derivative[width:-width])])
    Q = f_r / linewidth
    Q_c = Q / (1 - np.min(np.abs(s21_normalized)) / off)
    c = Q_c**-1
    i = Q**-1 - c
    p.add('i', value=i, min=1e-8, max=1)
    p.add('c', value=c, min=1e-8, max=1)
    p.add('c_imag', value=0, min=-1e-6, max=1e-6)
    return p


def guess(f, s21):
    """
    The phase of A is fixed at 0 and the phase at lowest frequency is incorporated into the cable delay term.
    """
    A = np.mean((np.abs(s21[0]), np.abs(s21[-1])))
    slope, offset = np.polyfit(f, np.unwrap(np.angle(s21)), 1)
    delay = -slope / (2 * np.pi)
    phi = np.angle(s21[0])
    f_phi = f[0]
    s21_normalized = s21 / (A * np.exp(2j * np.pi * delay * f))
    p.add('A_mag', value=off, min=0, max=1e6)


    p.add('delay', value=-slope / (2 * np.pi))
    p.add('phi', value=np.angle(s21[0]), min=-np.pi, max=np.pi)
    p.add('f_phi', value=f[0], vary=False)

    return p




def delayed_s21(params, f):
    return khalil.cable_delay(params, f) * s21(params, f)


def f_0(params):
    return params['f_r'].value


def Q_i(params):
    return params['i'].value**-1


def Q_c(params):
    return params['c'].value**-1


def Q_e(params):
    return (params['c'].value + 1j * params['c_imag'].value)**-1


def Q_r(params):
    return (params['i'].value + params['c'].value)**-1


def Q(params):
    return Q_r(params)


functions = {'f_0': f_0,
             'Q_i': Q_i,
             'Q_c': Q_c,
             'Q_e': Q_e,
             'Q_r': Q_r,
             'Q': Q}
