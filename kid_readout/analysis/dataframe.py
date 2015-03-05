"""
This module is for functions that operate on DataFrames.

Only the functions that use groupby should return the DataFrame.
"""
from __future__ import division
from copy import deepcopy
import numpy as np
import pandas as pd
from kid_readout.analysis.khalil import qi_error
from kid_readout.analysis import archive
import lmfit


def analyze(df, channel_to_location=None, maximum_fractional_f_r_error=1e-5, maximum_iQi_error=1e-5):
    rename_f(df)
    add_Q_i_err(df)
    add_channel(df)
    if channel_to_location is not None:
        add_location(df, channel_to_location)
    add_zbd_power(df)
    add_resonator_fit_good(df, maximum_fractional_f_r_error, maximum_iQi_error)
    df = archive.add_noise_fits(df)
    df = archive.add_total_mmw_attenuator_turns(df)
    return df


def analyze_mmw_power_steps(df, channel_to_location=None, maximum_X_error=1e-6, maximum_I_error=1e-5,
                            initial_responsivity=None, masker = lambda g: np.array((g.timestream_modulation_duty_cycle==0) &
                            g.resonator_fit_good)):
    pass
    """
    if initial_responsivity is None:
        # implement guess
        raise NotImplementedError()
    if masker is None:
        masker = lambda g: np.array((g.timestream_modulation_duty_cycle==0) &
                                    g.resonator_fit_good)
    df = add_responsivities(df, initial_responsivity, masker=masker)
    df = add_NEP2(df)
    return df
    """

def rename_f(df):
    df.rename(columns={'f_0': 'f_r', 'f_0_err': 'f_r_err'}, inplace=True)


# TODO: this should be deprecated when the error on Q_i or Q_i**-1 is computed upstream.
def add_Q_i_err(df):
    df['Q_i_err'] = [qi_error(row.Q, row.Q_err, row.Q_e_real, row.Q_e_real_err, row.Q_e_imag, row.Q_e_imag_err)
                     for index, row in df.iterrows()]


# This should work except in pathological cases; groupby sorts by key automatically.
def add_channel(df):
    resonator_index_to_channel = np.array(
        [group.f_r.mean() for index, group in df.groupby('resonator_index')]).argsort()
    df['channel'] = resonator_index_to_channel[np.array(df.resonator_index)]


def add_location(df, channel_to_location):
    df['location'] = channel_to_location[np.array(df.channel)]


# TODO: upgrade to use ZBD responsivity and account for lock-in
def add_zbd_power(df, zbd_volts_per_watt=2200):
    df['zbd_power'] = df.zbd_voltage / zbd_volts_per_watt


def add_resonator_fit_good(df, maximum_fractional_f_r_error, maximum_iQi_error):
    df['resonator_fit_good'] = ((df.f_r_err / df.f_r < maximum_fractional_f_r_error) &
                                (df.Q_i_err / df.Q_i**2 < maximum_iQi_error) &
                                (df.Q_i > 0))  # This shouldn't be necessary, but it currently is.


# Response fitting
# TODO: move the fitting functions to another module and rename them.
def n_qp_over_n_star(P, P_0, P_star):
    return (1 + (P + P_0) / P_star) ** (1 / 2) - 1


def X(P, P_0, P_star, X_0):
    return X_0 * n_qp_over_n_star(P, P_0, P_star)


def X_model(params, P):
    return X(P, params['P_0'].value, params['P_star'].value, params['X_0'].value)


def X_inverse(X, P_0, P_star, X_0):
    return P_star * ((1 + X / X_0) ** 2 - 1) - P_0


def X_residual(params, P, X_data, X_errors):
    return (X_data - X_model(params, P)) / X_errors


def I(P, P_0, P_star, I_0, I_C):
    return I_0 * n_qp_over_n_star(P, P_0, P_star) + I_C


def I_model(params, P):
    return I(P, params['P_0'].value, params['P_star'].value, params['I_0'].value, params['I_C'].value)


def I_inverse(I, P_0, P_star, I_0, I_C):
    return P_star * ((1 + (I - I_C) / I_0) ** 2 - 1) - P_0


def I_residual(params, P, I_data, I_errors):
    return (I_data - I_model(params, P)) / I_errors


def residual(params, P, X_data, I_data, X_errors, I_errors):
    return np.concatenate((X_residual(params, P, X_data, X_errors),
                           I_residual(params, P, I_data, I_errors)))


def fit(P, X_data, I_data, initial, X_errors=1, I_errors=1, **kwargs):
    return lmfit.minimize(residual, initial, args=(P, X_data, I_data, X_errors, I_errors), **kwargs)


def guess(P, X_data, I_data):
    return parameters(4e-8, 1e-6, 20e-6, 5e-6, 5e-6)


def parameters(P_0, P_star, X_0, I_0, I_C):
    params = lmfit.Parameters()
    params.add('P_0', value=P_0, min=0)
    params.add('P_star', value=P_star, min=0)
    params.add('X_0', value=X_0, min=0)
    params.add('I_0', value=I_0, min=0)
    params.add('I_C', value=I_C, min=0)
    return params


def dX_dP(P, P_0, P_star, X_0):
    return X_0 / (2 * n_qp_over_n_star(P, P_0, P_star))


def dI_dP(P, P_0, P_star, I_0):
    return I_0 / (2 * n_qp_over_n_star(P, P_0, P_star))


def add_XI_responsivity(df, power='zbd_power', masker=None):
    def XI_responsivity(group):
        if masker is None:
            mask = np.ones(group.shape[0], dtype=np.bool)
        else:
            mask = masker(group)
        #group.sort(power, inplace=True)
        f_r_max_row = group[mask][group[mask].f_r == group[mask].f_r.max()]
        f_r_max = float(f_r_max_row.f_r)
        f_r_max_err = float(f_r_max_row.f_r_err)
        group.loc[mask, '{}_f_r_max'.format(power)] = f_r_max
        group.loc[mask, '{}_f_r_max_err'.format(power)] = f_r_max_err
        group.loc[mask, '{}_X'.format(power)] = f_r_max / group.f_r - 1
        group.loc[mask, '{}_X_err'.format(power)] = ((f_r_max_err / f_r_max)**2 + (group.f_r_err / group.f_r)**2)**(1/2)
        group.loc[mask, '{}_I'.format(power)] = group.Q_i**-1
        group.loc[mask, '{}_I_err'.format(power)] = group.Q_i**-2 * group.Q_i_err
        return group
    return df.groupby(('channel', 'atten')).apply(XI_responsivity).reset_index(drop=True)


# Remove bad data and power-off points before fitting.
def fit_XI_responsivity(df, initial, power='zbd_power', masker=lambda group: np.ones(group.shape[0], dtype=np.bool)):
    def XI_responsivity(group):
        mask = masker(group)
        # Add try...except for cases where the fit fails because too many points are masked.
        try:
            result = fit(group[mask][power], group[mask]['{}_X'.format(power)], group[mask]['{}_I'.format(power)],
                         deepcopy(initial),  # This is crucial because minimize modifies the input Parameters.
                         X_errors=group[mask]['{}_X_err'.format(power)], I_errors=group[mask]['{}_I_err'.format(power)])
            for p in result.params.values():
                group.loc[mask, '{}_XI_fit_{}'.format(power, p.name)] = p.value
                group.loc[mask, '{}_XI_fit_{}_err'.format(power, p.name)] = p.stderr
            group.loc[mask, '{}_XI_fit_redchi'.format(power)] = result.redchi
            group.loc[mask, '{}_dX_dP'.format(power)] = dX_dP(group[power], result.params['P_0'].value,
                                                              result.params['P_star'].value, result.params['X_0'].value)
            group.loc[mask, '{}_dI_dP'.format(power)] = dI_dP(group[power], result.params['P_0'].value,
                                                              result.params['P_star'].value, result.params['I_0'].value)
        except TypeError:
            pass
        return group
    return df.groupby(('channel', 'atten')).apply(XI_responsivity).reset_index(drop=True)


def add_NEP2(df, power='zbd_power'):
    df['{}_NEP2_device'.format(power)] = df.noise_fit_device_noise / df['{}_dX_dP'.format(power)] ** 2
    df['{}_NEP2_amplifier'.format(power)] = df.noise_fit_amplifier_noise / df['{}_dX_dP'.format(power)] ** 2
    return df

# TODO: these are deprecated; remove
def z(params, P):
    z_0 = params['z_0'].value
    P_0 = params['P_0'].value
    P_star = params['P_star'].value
    return z_0 * ((1 + (P + P_0) / P_star) ** (1 / 2) - 1)


def dz_dP(params, P):
    z_0 = params['z_0'].value
    P_0 = params['P_0'].value
    P_star = params['P_star'].value
    return z_0 / (2 * P_star) * (1 + (P + P_0) / P_star) ** (-1 / 2)


def z_residual(params, P, data, errors=1):
    return (data - z(params, P)) / errors


def fit_z(P, data, initial, errors=1, **kwargs):
    return lmfit.minimize(z_residual, initial, args=(P, data, errors), **kwargs)


def z_initial(z_0=1, P_0=1, P_star=1):
    params = lmfit.Parameters()
    params.add('z_0', value=z_0)
    params.add('P_0', value=P_0, min=0)
    params.add('P_star', value=P_star, min=0)
    return params


def z_inverse(params, z):
    z_0 = params['z_0'].value
    P_0 = params['P_0'].value
    P_star = params['P_star'].value
    return P_star * ((1 + z / z_0) ** 2 - 1) - P_0


# Mask bad data before fitting.
def add_zbd_power_responsivities(df):
    def calculate_zbd_power_responsivities(group):
        group.sort('zbd_power', inplace=True)
        mask = group.timestream_modulation_duty_cycle == 0  # 1 is source off and 0 is source on
        group.loc[mask, 'f_r_max'] = group[mask].f_r.max()
        group.loc[mask, 'x'] = group.f_r / group.f_r_max - 1

        P_zbd_x_fit = fit_z(group[mask].zbd_power, group[mask].x, z_initial(-10e-6, 10e-6, 10e-6),
                            xtol=1e-10, ftol=1e-10)  # Experimental.
        group.loc[mask, 'P_zbd_x_fit_z_0'] = P_zbd_x_fit.params['z_0'].value
        group.loc[mask, 'P_zbd_x_fit_P_0'] = P_zbd_x_fit.params['P_0'].value
        group.loc[mask, 'P_zbd_x_fit_P_star'] = P_zbd_x_fit.params['P_star'].value
        group.loc[mask, 'dx_dP'] = dz_dP(P_zbd_x_fit.params, group[mask].zbd_power)

        P_zbd_iQi_fit = fit_z(group[mask].zbd_power, 1 / group[mask].Q_i, z_initial(10e-6, 1e-6, 10e-6),
                              xtol=1e-10, ftol=1e-10)  # Experimental.
        group.loc[mask, 'P_zbd_iQi_fit_z_0'] = P_zbd_iQi_fit.params['z_0'].value
        group.loc[mask, 'P_zbd_iQi_fit_P_0'] = P_zbd_iQi_fit.params['P_0'].value
        group.loc[mask, 'P_zbd_iQi_fit_P_star'] = P_zbd_iQi_fit.params['P_star'].value
        group.loc[mask, 'diQi_dP'] = dz_dP(P_zbd_iQi_fit.params, group[mask].zbd_power)

        return group

    return df.groupby(('channel', 'atten')).apply(calculate_zbd_power_responsivities).reset_index(drop=True)


# Be warned that the noise functions currently calculate the noise differently: device_noise includes the amplifier
# noise, while noise_fit_device_noise does not.

def add_NET2(df):
    df['NET2_device'] = df.device_noise / (2 * df.dx_dT ** 2)
    df['NET2_amplifier'] = df.amplifier_noise / (2 * df.dx_dT ** 2)
    return df


def add_NET2_fit(df):
    df['NET2_fit_device'] = df.noise_fit_device_noise / (2 * df.dx_dT ** 2)
    df['NET2_fit_amplifier'] = df.noise_fit_amplifier_noise / (2 * df.dx_dT ** 2)
    return df


def add_NEP2_zbd(df):
    df['NEP2_zbd_device'] = df.device_noise / df.dx_dP ** 2
    df['NEP2_zbd_amplifier'] = df.amplifier_noise / df.dx_dP ** 2
    return df


def add_NEP2_zbd_fit(df):
    df['NEP2_zbd_fit_device'] = df.noise_fit_device_noise / df.dx_dP ** 2
    df['NEP2_zbd_fit_amplifier'] = df.noise_fit_amplifier_noise / df.dx_dP ** 2
    return df