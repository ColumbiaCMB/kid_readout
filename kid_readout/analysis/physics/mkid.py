from scipy.constants import c, h, k as k_B, pi
from scipy.special import psi
import lmfit
from physics.mkid import noroozian
from kid_readout.analysis import fitter


def inverse_Q(params, T):
    return (inverse_Q_qp(params, T) +
            inverse_Q_TLS(params, T) +
            params['inverse_Q_other'].value)

def inverse_Q_qp(params, T):
    f = params['f'].value
    alpha_k = params['alpha_k'].value
    gamma = params['gamma'].value
    Delta = params['Delta'].value
    N_0 = params['N_0'].value
    S_1 = noroozian.S_1(f, T, Delta)
    n_qp = noroozian.equation_2_3(T, Delta, N_0)
    return (2 * alpha_k * gamma * S_1 * n_qp /
            (4 * N_0 * Delta))

def inverse_Q_TLS(params, T):
    F_TLS_delta_0 = params['F_TLS_delta_0'].value
    f = params['f'].value
    return F_TLS_delta_0 * np.tanh(h * f / (2 * k_B * T))

def x(params, T):
    return (x_qp(params, T) + # - x_qp(params, T_0) +
            x_TLS(params, T) + # - x_TLS(params, T_0) +
            params['x_other'].value)

def x_qp(params, T):
    f = params['f'].value
    alpha_k = params['alpha_k'].value
    gamma = params['gamma'].value
    Delta = params['Delta'].value
    N_0 = params['N_0'].value
    S_2 = noroozian.S_2(f, T, Delta)
    n_qp = noroozian.equation_2_3(T, Delta, N_0)
    return (-alpha_k * gamma * S_2 * n_qp /
            (4 * N_0 * Delta))

def x_TLS(params, T):
    F_TLS_delta_0 = params['F_TLS_delta_0'].value
    f = params['f'].value
    return F_TLS_delta_0 / pi * (psi(0.5 + h * f / (2j * pi * k_B * T)).real - np.log(h * f / (k_B * T)))

def dual_residual(params, T, x, x_error, inverse_Q, inverse_Q_error):
    pass