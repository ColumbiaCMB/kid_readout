"""
TODO: move this to analysis eventually.
"""
from __future__ import division
import numpy as np
import lmfit


# TODO: rename
def model(params, f):
    return amp(params, f) + TLS(params, f) + qp(params, f)


def amp(params, f):
    S_amp = params['S_amp'].value
    return S_amp * np.ones(f.size)


def TLS(params, f):
    S_TLS = params['S_TLS'].value
    f_r = params['f_r'].value
    return (1 + (f / f_r)**2)**-1 * (f / 1)**(-1/2) * S_TLS


def qp(params, f):
    S_qp = params['S_qp'].value
    f_r = params['f_r'].value
    f_qp = params['f_qp'].value
    return (1 + (f / f_r)**2)**-1 * (1 + (f / f_qp)**2)**-1 * S_qp


def guess(f, S):
    f_r = 500
    f_qp = 500
    S_amp = S[-10:].mean()
    S_TLS = S[np.argmin(f - 1)]
    S_qp = S[np.argmin(f - f_r) - 10:np.argmin(f - f_r) + 10].mean()
    return parameters(S_amp, S_TLS, S_qp, f_r, f_qp)


def bandwidth_limited_guess(f, S, f_r):
    # Make f_qp negligible
    f_qp = 10 * f.max()
    S_amp = S[-10:].mean()
    S_qp = S[np.argmin(f - f_r)] - S_amp
    S_TLS = S[np.argmin(f - 1)] - S_amp - S_qp
    return parameters(S_amp, S_TLS, S_qp, f_r, f_qp)


def parameters(S_amp, S_TLS, S_qp, f_r, f_qp):
    params = lmfit.Parameters()
    params.add('S_amp', value=S_amp, min=0)
    params.add('S_TLS', value=S_TLS, min=0)
    params.add('S_qp', value=S_qp, min=0)
    params.add('f_r', value=f_r, min=0)
    params.add('f_qp', value=f_qp, min=0)
    return params