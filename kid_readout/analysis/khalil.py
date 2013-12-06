"""
This module uses models from the Khalil paper.
"""

from __future__ import division
from scipy.special import cbrt
import numpy as np
from lmfit import Parameters

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
    return np.exp(1j * (-2 * np.pi * (f - np.min(f)) * delay + phi))

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
    return A * (1 - (Q * Q_e**-1 /
                     (1 + 2j * Q * (f - f_0) / f_0)))

def bifurcation_s21(params,f):
    A = (params['A_mag'].value *
         np.exp(1j * params['A_phase'].value))
    f_0 = params['f_0'].value
    Q = params['Q'].value
    Q_e = (params['Q_e_real'].value +
           1j * params['Q_e_imag'].value)
           
    a = params['a'].value
    y_0 = ((f - f_0)/f_0)*Q
    y =  (y_0/3. + 
            (y_0**2./9. - 1./12.)/cbrt(a/8 + y_0/12 + ((y_0**3/27 + y_0/12 + a/8)**2 - (y_0**2/9 - 1/12.)**3)**(1/2.) + y_0**3/27) + 
            cbrt(a/8 + y_0/12 + np.sqrt((y_0**3/27. + y_0/12 + a/8)**2 - (y_0**2/9 - 1/12.)**3) + y_0**3/27))
    x = y/Q
    s21 = A*(1 - (Q/Q_e)/(1+2j*Q*x))
    return s21

def delayed_generic_s21(params, f):
    """
    This adds a cable delay controlled by two parameters to the
    generic model above.
    """
    return cable_delay(params, f) * generic_s21(params, f)
    
def bifurcation_guess(f, data):
    p = delayed_generic_guess(f,data)
    p.add('a',value=0,min=0,max=1)
    return p

def delayed_generic_guess(f, data):
    """
    The phase of A is fixed at 0 and the phase at lowest frequency is
    incorporated into the cable delay term.
    """
    p = generic_guess(f, data)
    p['A_phase'].value = 0
    p['A_phase'].vary = False
    slope, offset = np.polyfit(f, np.unwrap(np.angle(data)), 1)
    p.add('delay', value = -slope / (2 * np.pi))
    p.add('phi', value = np.angle(data[0]), min = -np.pi, max = np.pi)
    return p

def generic_guess(f, data):
    """
    Right now these Q values are magic numbers. I suppose the
    design values are a good initial guess, but there might be a
    good way to approximate them without doing the full fit.
    """
    p = Parameters()
    bw = f.max() - f.min()
    # Allow f_0 to vary by +/- the bandwidth over which we have data
    p.add('f_0', value = f[np.argmin(abs(data))],
          min = f.min() - bw, max = f.max() + bw)
    p.add('A_mag', value = np.mean((np.abs(data[0]), np.abs(data[-1]))),
          min = 0, max = 1e6)
    p.add('A_phase', value = np.mean(np.angle(data)),
          min = -np.pi, max = np.pi)
    p.add('Q', value = 5e4, min = 0, max = 1e7)
    p.add('Q_e_real', value = 4e4, min = 0, max = 1e6)
    p.add('Q_e_imag', value = 0, min = -1e6, max =1e6)
    return p

def Q_i(params):
    """
    Return the internal Q of the resonator.
    """
    Q = params['Q'].value
    Qe = Q_e(params)
    return (Q**-1 - np.real(Qe**-1))**-1

def Q_e(params):
    """
    Return the external (coupling) Q of the resonator.
    """
    return (params['Q_e_real'].value +
            1j * params['Q_e_imag'].value)
    
# Zmuidzinas doesn't say how to calculate the coupling coefficient
# \chi_c when Q_e (what he calls Q_c) is complex, and I don't know
# whether to use the real part or the norm of Q_e. It doesn't seem to
# make much difference.
def chi_c_real(params):
    """
    Calculate the coupling coefficient \chi_c
    using the real part of Q_e.
    """
    Qi = Q_i(params)
    Qc = params['Q_e_real'].value
    return ((4 * Qc * Qi) /
            (Qc + Qi)**2)

def chi_c_norm(params):
    """
    Calculate the coupling coefficient \chi_c
    using the norm of Q_e.
    """
    Qi = Q_i(params)
    Qc = np.abs(Q_e(params))
    return ((4 * Qc * Qi) /
            (Qc + Qi)**2)

generic_functions = {'Q_i': Q_i,
                     'Q_e': Q_e,
                     'chi_c_real': chi_c_real,
                     'chi_c_norm': chi_c_norm}
