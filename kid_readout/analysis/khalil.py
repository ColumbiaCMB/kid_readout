"""
This module uses models from the Khalil paper.
"""

from __future__ import division
import numpy as np
from scipy.special import cbrt
from lmfit import Parameters


def qi_error(Q, Q_err, Q_e_real, Q_e_real_err, Q_e_imag, Q_e_imag_err):
    """
    Compute error on Qi
    
    Khalil et al defines Qi as 1/Qi = 1/Qr - Real(1/Qe), where Qe is
    the complex coupling Q. This can be rewritten as:
    
    $$ Qi = 1/(1/Q_r - \frac{Q_{e,real}}{Q_{e,real}^2 - Q_{e,imag}^2} $$
    
    Assuming the errors are independent (which they seem to mostly be),
    the error on Qi will then be:
    
    $$ \Delta Q_i = \sqrt( (\Delta Q \diff{Qi}{Q})^2 + (\Delta Q_{e,real} \diff{Qi}{Q_{e,real}})^2 + (\Delta Q_{e,imag} \diff{Qi}{Q_{e,imag}})^2 )$$
    
    The derivatives are:
    
    $$ \diff{Qi}{Q} = \frac{(Qer^2-Qei^2)^2}{(Q Qer - Qer^2 + Qei^2)^2} $$
    
    $$ \diff{Qi}{Qer} = -\frac{Qe^2(Qer^2 + Qei^2)}{(Q Qer - Qer^2 + Qei^2)^2} $$
    
    $$ \diff{Qi}{Qei} = \frac{2 Q^2 Qer Qei}{(Q Qer - Qer^2 + Qei^2)^2} $$
    
    """

    dQ = Q_err
    Qer = Q_e_real
    dQer = Q_e_real_err
    Qei = Q_e_imag
    dQei = Q_e_imag_err
    denom = (Q * Qer - Qer ** 2 + Qei ** 2) ** 2

    dQi_dQ = (Qer ** 2 - Qei ** 2) ** 2 / denom
    dQi_dQer = (Q ** 2 * (Qer ** 2 + Qei ** 2)) / denom
    dQi_dQei = (2 * Q ** 2 * Qer * Qei) / denom

    dQi = np.sqrt((dQ * dQi_dQ) ** 2 + (dQer * dQi_dQer) ** 2 + (dQei * dQi_dQei) ** 2)
    return dQi

# todo: rewrite all functions to use params.valuesdict()

def cable_delay(params, f):
    """
    This assumes that signals go as exp(i \omega t) so that a time
    delay corresponds to negative phase. In our sweeps the phase
    advances with frequency, so I think that currently either the
    convention is reversed in the readout or we have a time lead.
    If *f* is in MHz, *delay* will be in microseconds.
    If *f* is in Hz, *delay* will be in seconds.
    Parameter *phi* is the phase at f = f_min.
    """
    delay = params['delay'].value
    phi = params['phi'].value
    f_min = params['f_phi'].value
    return np.exp(1j * (-2 * np.pi * (f - f_min) * delay + phi))


def generic_s21(params, f):
    """
    This is Equation 11, except that the parameter A is a complex
    prefactor intended to encapsulate the 1 + \hat{\epsilon} as well
    as any external gains and phase shifts.
    """
    A = (params['A_mag'].value *
         np.exp(1j * params['A_phase'].value))
    f_0 = params['f_0'].value
    Q = params['Q'].value
    Q_e = (params['Q_e_real'].value +
           1j * params['Q_e_imag'].value)
    return A * (1 - (Q * Q_e ** -1 /
                     (1 + 2j * Q * (f - f_0) / f_0)))


def create_model(f_0=100e6, Q=1e4, Q_e=2e4, A=1.0, delay=0.0, a=0.0):
    p = Parameters()
    A_mag = np.abs(A)
    phi = np.angle(A)
    Q_e_real = np.real(Q_e)
    Q_e_imag = np.imag(Q_e)
    p.add('f_0', value=f_0)
    p.add('Q', value=Q)
    p.add('Q_e_real', value=Q_e_real)
    p.add('Q_e_imag', value=Q_e_imag)
    p.add('A_mag', value=A_mag)
    p.add('A_phase', value=0)
    p.add('phi', value=phi)
    p.add('delay', value=delay)
    p.add('f_phi', value=0)
    p.add('a', value=a)
    return p


def bifurcation_s21(params, f):
    """
    Swenson paper:
        Equation: y = yo + A/(1+4*y**2)
    """
    A = (params['A_mag'].value *
         np.exp(1j * params['A_phase'].value))
    f_0 = params['f_0'].value
    Q = params['Q'].value
    Q_e = (params['Q_e_real'].value +
           1j * params['Q_e_imag'].value)

    a = params['a'].value

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
    s21 = A * (1 - (Q / Q_e) / (1 + 2j * Q * x))
    mask = np.isfinite(s21)
    if scalar or not np.all(mask):
        s21_interp_real = np.interp(f, fmodel[mask], s21[mask].real)
        s21_interp_imag = np.interp(f, fmodel[mask], s21[mask].imag)
        s21new = s21_interp_real + 1j * s21_interp_imag
    else:
        s21new = s21
    return s21new * cable_delay(params, f)


def delayed_generic_s21(params, f):
    """
    This adds a cable delay controlled by two parameters to the generic model above.
    """
    return cable_delay(params, f) * generic_s21(params, f)


def bifurcation_guess(f, s21):
    p = delayed_generic_guess(f, s21)
    p.add('a', value=0, min=0, max=0.8)
    return p


def delayed_generic_guess(f, s21):
    """
    The phase of A is fixed at 0 and the phase at lowest frequency is incorporated into the cable delay term.
    """
    p = generic_guess(f, s21)
    p['A_phase'].value = 0
    p['A_phase'].vary = False
    slope, offset = np.polyfit(f, np.unwrap(np.angle(s21)), 1)
    p.add('delay', value=-slope / (2 * np.pi))
    p.add('phi', value=np.angle(s21[0]), min=-np.pi, max=np.pi)
    p.add('f_phi', value=f[0], vary=False)
    return p


def generic_guess(f, s21):
    """
    This is deprecated in favor of auto_guess because it uses hardcoded Q values.
    """
    p = Parameters()
    bw = f.max() - f.min()
    # Allow f_0 to vary by +/- the bandwidth over which we have data
    p.add('f_0', value=f[np.argmin(abs(s21))],
          min=f.min() - bw, max=f.max() + bw)
    p.add('A_mag', value=np.mean((np.abs(s21[0]), np.abs(s21[-1]))),
          min=0, max=1e6)
    p.add('A_phase', value=np.mean(np.angle(s21)),
          min=-np.pi, max=np.pi)
    p.add('Q', value=5e4, min=0, max=1e7)
    p.add('Q_e_real', value=4e4, min=0, max=1e6)
    p.add('Q_e_imag', value=0, min=-1e6, max=1e6)
    return p


def auto_guess(f, s21):
    """
    Use the linewidth and the transmission ratio on and off resonance to guess the initial Q values.  Estimate the
    linewidth by smoothing then looking for the extrema of the first derivative. This may fail if the resonance is
    very close to the edge of the data.
    """
    p = Parameters()
    bw = f.max() - f.min()
    # Allow f_0 to vary by +/- the bandwidth over which we have data
    p.add('f_0', value=f[np.argmin(abs(s21))],
          min=f.min() - bw, max=f.max() + bw)
    off = np.mean((np.abs(s21[0]), np.abs(s21[-1])))
    p.add('A_mag', value=off,
          min=0, max=1e6)
    p.add('A_phase', value=np.mean(np.angle(s21)),
          min=-np.pi, max=np.pi)
    width = int(f.size / 10)
    gaussian = np.exp(-np.linspace(-4, 4, width) ** 2)
    gaussian /= np.sum(gaussian)  # not necessary
    smoothed = np.convolve(gaussian, abs(s21), mode='same')
    derivative = np.convolve(np.array([1, -1]), smoothed, mode='same')
    # Exclude the edges, which are affected by zero padding.
    linewidth = (f[np.argmax(derivative[width:-width])] -
                 f[np.argmin(derivative[width:-width])])
    p.add('Q', value=p['f_0'].value / linewidth,
          min=1, max=1e7)  # This seems to stop an occasional failure mode.
    p.add('Q_e_real', value=(p['Q'].value /
                             (1 - np.min(np.abs(s21)) / off)),
          min=1, max=1e6)  # As above.
    p.add('Q_e_imag', value=0, min=-1e6, max=1e6)
    return p


def delayed_auto_guess(f, s21):
    auto = auto_guess(f, s21)
    delayed = delayed_generic_guess(f, s21)
    delayed['Q'].value = auto['Q'].value
    delayed['Q_e_real'].value = auto['Q_e_real'].value
    return delayed


def Q_i(params):
    """
    Return the internal quality factor of the resonator.
    """
    Q = params['Q'].value
    Qe = Q_e(params)
    return (Q ** -1 - np.real(Qe ** -1)) ** -1


def Q_e(params):
    """
    Return the external (coupling) quality factor of the resonator.
    """
    return (params['Q_e_real'].value +
            1j * params['Q_e_imag'].value)


# Zmuidzinas doesn't say how to calculate the coupling coefficient \chi_c when Q_e (what he calls Q_c) is complex,
# and I don't know whether to use the real part or the norm of Q_e. It doesn't seem to make much difference.
def chi_c_real(params):
    """
    Calculate the coupling coefficient \chi_c using the real part of Q_e.
    """
    Qi = Q_i(params)
    Qc = params['Q_e_real'].value
    return ((4 * Qc * Qi) /
            (Qc + Qi) ** 2)


def chi_c_norm(params):
    """
    Calculate the coupling coefficient \chi_c using the norm of Q_e.
    """
    Qi = Q_i(params)
    Qc = np.abs(Q_e(params))
    return ((4 * Qc * Qi) /
            (Qc + Qi) ** 2)

# todo: write a function to calculate normalized s21 here.

generic_functions = {'Q_i': Q_i,
                     'Q_e': Q_e,
                     'chi_c_real': chi_c_real,
                     'chi_c_norm': chi_c_norm}
