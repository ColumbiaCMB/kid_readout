"""
Equations relevant to modeling microwave resonators
"""
from __future__ import division

import numpy as np
from scipy.special import cbrt


################
# Cable response models
################

def cable_delay(f, delay, phi, f_min):
    """
    This assumes that signals go as exp(i \omega t) so that a time
    delay corresponds to negative phase. In our sweeps the phase
    advances with frequency, so I think that currently either the
    convention is reversed in the readout or we have a time lead.
    If *f* is in MHz, *delay* will be in microseconds.
    If *f* is in Hz, *delay* will be in seconds.
    Parameter *phi* is the phase at f = f_min.
    """
    return np.exp(1j * (-2 * np.pi * (f - f_min) * delay + phi))


def general_cable(f, delay, phi, f_min, A_mag, A_slope):
    phase_term =  cable_delay(f,delay,phi,f_min)
    magnitude_term = ((f-f_min)*A_slope + 1)* A_mag
    return magnitude_term*phase_term

###############
# Linear models
###############

def linear_resonator(f, f_0, Q, Q_e_real, Q_e_imag):
    Q_e = Q_e_real + 1j*Q_e_imag
    return (1 - (Q * Q_e**-1 /
                     (1 + 2j * Q * (f - f_0) / f_0)))

def inverse_linear_resonator(f, f_0, iQ, iQ_e_real, iQ_e_imag):
    Q = 1./iQ
    Qe = 1./(iQ_e_real+1j*iQ_e_imag)
    return linear_resonator(f,f_0,Q,Q_e_real=np.real(Qe),Q_e_imag=np.imag(Qe))

######################################
# Non-linear model from Swenson et al.
######################################

def nonlinear_resonator(f,f_0,Q,Q_e_real,Q_e_imag,a):
    Q_e = Q_e_real + 1j*Q_e_imag
    if np.isscalar(f):
        fmodel = np.linspace(f * 0.9999, f * 1.0001, 1000)
        scalar = True
    else:
        fmodel = f
        scalar = False
    y_0 = ((fmodel - f_0) / f_0) * Q
    y = (y_0 / 3. +
         (y_0 ** 2 / 9 - 1 / 12) / cbrt(a / 8 + y_0 / 12 + np.sqrt(
             (y_0 ** 3 / 27 + y_0 / 12 + a / 8) ** 2 - (y_0 ** 2 / 9 - 1 / 12) ** 3) + y_0 ** 3 / 27) +
         cbrt(a / 8 + y_0 / 12 + np.sqrt(
             (y_0 ** 3 / 27 + y_0 / 12 + a / 8) ** 2 - (y_0 ** 2 / 9 - 1 / 12) ** 3) + y_0 ** 3 / 27))
    x = y / Q
    s21 = (1 - (Q / Q_e) / (1 + 2j * Q * x))
    mask = np.isfinite(s21)
    if scalar or not np.all(mask):
        s21_interp_real = np.interp(f, fmodel[mask], s21[mask].real)
        s21_interp_imag = np.interp(f, fmodel[mask], s21[mask].imag)
        s21new = s21_interp_real + 1j * s21_interp_imag
    else:
        s21new = s21
    return s21new


def inverse_nonlinear_resonator(f, f_0, iQ, iQ_e_real, iQ_e_imag,a):
    Q = 1/iQ
    Qe = 1/(iQ_e_real+1j*iQ_e_imag)
    return nonlinear_resonator(f,f_0,Q,Q_e_real=np.real(Qe),Q_e_imag=np.imag(Qe),a=a)


# A linear model that fits directly to the inverse quality factors, or losses
def linear_loss_resonator(f, f_0, loss_i, loss_c, asymmetry):
    x = f / f_0 - 1
    return 1 - ((1 + 1j * asymmetry) /
                (1 + (loss_i + 2j * x) / loss_c))
