"""
This module uses models from the Khalil paper.
"""

from __future__ import division

import numpy as np
from lmfit import Parameters

def cable_delay(f, delay, phi):
    """
    This assumes that signals go as exp(i \omega t) so that a time
    delay corresponds to negative phase.
    """
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
    phi = params['phi'].value
    return cable_delay(f, delay, phi) * generic(params, f)

def generic_guess(f, data):
    """
    Right now these Q values are magic numbers. I suppose the
    design values are a good initial guess, but there might be a
    good way to approximate them without doing the full fit.
    """
    p = Parameters()
    p.add('f_0', value=f[np.argmin(abs(data))], min=0)
    p.add('A_mag', value=np.mean((np.abs(data[0]), np.abs(data[-1]))), min=0)
    p.add('A_phase', value=np.mean(np.angle(data)), min=-np.pi, max=np.pi)
    p.add('Q', value=5e4, min=0)
    p.add('Q_e_real', value=5e4)
    p.add('Q_e_imag', value=0)
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
