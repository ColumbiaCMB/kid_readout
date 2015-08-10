"""
This module uses the Khalil and Swenson models but fits directly to the inverse quality factors, since these are actually more
useful.
"""
from __future__ import division
import numpy as np
from scipy.special import cbrt
import lmfit


def parameters(f_r=100, i=1e-6, c=1e-5, c_imag=0, a=0, A=1, delay=0, phi=0, f_phi=0):
    p = lmfit.Parameters()
    p.add('f_r', value=f_r, min=0)
    p.add('i', value=i, min=1e-8, max=1)
    p.add('c', value=c, min=1e-8, max=1)
    p.add('c_imag', value=c_imag, min=-1e-6, max=1e-6)
    p.add('a', value=a, min=0, max=4*3**(1/2)/9)
    p.add('A', value=A, min=0)
    p.add('delay', value=delay)
    p.add('phi', value=phi)
    p.add('f_phi', value=f_phi, vary=False)
    return p


def normalization(params, f):
    A = params['A'].value
    delay = params['delay'].value
    phi = params['phi'].value
    f_phi = params['f_phi'].value
    return A * np.exp(1j * (-2 * np.pi * (f - f_phi) * delay + phi))


def model_normalized(params, f):
    f_r = params['f_r'].value
    i = params['i'].value
    c = params['c'].value
    c_imag = params['c_imag'].value
    a = params['a'].value
    Q_r = (i + c)**-1
    y_0 = (f / f_r - 1) * Q_r
    y = (y_0 / 3 +
         (y_0**2 / 9 - 1 / 12) / cbrt(a / 8 + y_0 / 12 + np.sqrt(
             (y_0**3 / 27 + y_0 / 12 + a / 8)**2 - (y_0**2 / 9 - 1 / 12)**3) + y_0**3 / 27) +
         cbrt(a / 8 + y_0 / 12 + np.sqrt(
             (y_0**3 / 27 + y_0 / 12 + a / 8)**2 - (y_0**2 / 9 - 1 / 12)**3) + y_0**3 / 27))
    x = y / Q_r
    return (1 - ((c + 1j * c_imag) /
                 (i + c + 2j * x)))


def model(params, f):
    return normalization(params, f) * model_normalized(params, f)


def guess_normalized(f, s21):
    """
    Use the linewidth and the transmission ratio on and off resonance to guess the initial values.  Estimate the
    linewidth by smoothing then looking for the extrema of the first derivative. This may fail if the resonance is
    very close to the edge of the data.

    This function expects the s21 data to be approximately normalized to 1 off resonance, and for the cable delay to be
    removed.
    """
    p = lmfit.Parameters()
    # Allow f_r to vary by a quarter of the data bandwidth.
    bandwidth = (f.max() - f.min()) / 4
    f_r = f[np.argmin(abs(s21))]
    p.add('f_r', value=f_r,
          min=f.min() - bandwidth, max=f.max() + bandwidth)
    width = int(f.size / 10)
    gaussian = np.exp(-np.linspace(-4, 4, width)**2)
    gaussian /= np.sum(gaussian)  # not necessary
    smoothed = np.convolve(gaussian, abs(s21), mode='same')
    derivative = np.convolve(np.array([1, -1]), smoothed, mode='same')
    # Exclude the edges, which are affected by zero padding.
    linewidth = (f[np.argmax(derivative[width:-width])] -
                 f[np.argmin(derivative[width:-width])])
    Q_r = f_r / linewidth
    Q_c = Q_r / (1 - np.min(np.abs(s21)))
    c = Q_c**-1
    i = Q_r**-1 - c
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
    s21_normalized = s21 / normalization(parameters(A=A, delay=delay, phi=phi, f_phi=f_phi), f)
    p = guess_normalized(f, s21_normalized)
    p.add('A', value=A, min=0, max=1e6)
    p.add('delay', value=delay)
    p.add('phi', value=phi, min=-np.pi, max=np.pi)
    p.add('f_phi', value=f_phi, vary=False)
    p.add('a', value=0, min=0, max=4*3**(1/2)/9)  # TODO: make exact
    return p


def Q_i(params):
    return params['i'].value**-1


def Q_c(params):
    return params['c'].value**-1


def Q_r(params):
    return (params['i'].value + params['c'].value)**-1


# Functions below are for backward compatibility:

def Q(params):
    return Q_r(params)


def Q_e(params):
    return (params['c'].value + 1j * params['c_imag'].value)**-1


def f_0(params):
    return params['f_r'].value


functions = {'f_0': f_0,
             'Q_i': Q_i,
             'Q_c': Q_c,
             'Q_e': Q_e,
             'Q_r': Q_r,
             'Q': Q}
