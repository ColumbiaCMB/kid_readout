"""
This module uses models from the Khalil paper.
"""

from __future__ import division

import numpy as np
from lmfit import Parameters

def cable_delay(f, delay, phi, f_low):
    """
    This assumes that signals go as exp(i \omega t) so that a time
    delay corresponds to negative phase.
    if *f* is in MHz, *delay* will be in microseconds
    if *f* is in Hz, *delay* will be in seconds 
    """
    f = f - f_low # subtract off the lowest frequency, otherwise phase at f.min() can essentially be arbitrary.
                    # doing this seems to improve covariance between delay and A_phase term.
    return np.exp(-1j * (2 * np.pi * f * delay + phi))

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

# This needs a corresponding guess function. It's fine if it includes
# magic numbers that we have measured.
def delayed_generic_s21(params, f):
    """
    This adds a cable delay controlled by two parameters to the
    generic model above.
    """
    delay = params['delay'].value
    f_low = params['f_low'].value
    phi = 0.0 #params['phi'].value # phase is already taken account in A_phase
    return cable_delay(f, delay, phi, f_low) * generic_s21(params, f)

def delayed_generic_guess(f, data):
    p = generic_guess(f,data)
    p.add('delay', value = 0.0, min = -10,max=10)
    p.add('f_low', value = f.min(), vary = False)
    return p

def generic_guess(f, data):
    """
    Right now these Q values are magic numbers. I suppose the
    design values are a good initial guess, but there might be a
    good way to approximate them without doing the full fit.
    """
    p = Parameters()
    bw = f.max() - f.min()
    p.add('f_0', value=f[np.argmin(abs(data))], min=f.min()-bw, max = f.max()+bw)  # Allow f_0 to vary by +/- the bandwidth over which we have data
    p.add('A_mag', value=np.mean((np.abs(data[0]), np.abs(data[-1]))), min=0,max=1e6)
    p.add('A_phase', value=np.mean(np.angle(data)), min=-np.pi, max=np.pi)
    p.add('Q', value=5e4, min=0,max=1e7)
    p.add('Q_e_real', value=4e4,min=0,max=1e6)
    p.add('Q_e_imag', value=0,min=-1e6,max=1e6)
    return p

def Q_i(params):
    """
    Return the internal Q of the resonator.
    """
    Q = params['Q'].value
    Q_e = (params['Q_e_real'].value +
           1j * params['Q_e_imag'].value)
    return (Q**-1 - np.real(Q_e**-1))**-1

generic_functions = {'Q_i': Q_i}
