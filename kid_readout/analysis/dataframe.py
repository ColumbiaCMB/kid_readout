"""
This module is for functions that operate on DataFrames.

Only the functions that use groupby should return the DataFrame.
"""
from __future__ import division
from copy import deepcopy
import numpy as np
import lmfit
from kid_readout.analysis.khalil import qi_error
from kid_readout.analysis import archive


def analyze(df, channel_to_location=None, maximum_fractional_f_r_error=1e-5, maximum_iQi_error=1e-5):
    # TODO: the next three functions should be deprecated when the underlying code is updated
    rename_f(df)
    add_Q_i_err(df)
    add_channel(df)

    if channel_to_location is not None:
        add_location(df, channel_to_location)
    add_resonator_fit_good(df, maximum_fractional_f_r_error, maximum_iQi_error)
    df = archive.add_noise_fits(df)
    return df


def analyze_mmw_source_data(df, zbd_fraction, optical_frequency=None):
    # TODO: this is a hack that should be deprecated eventually the voltages in other code are updated
    try:
        df.rename(columns={'zbd_voltage': 'lockin_rms_voltage'}, inplace=True)
    except AttributeError:
        pass
    add_zbd_voltage(df)
    add_zbd_power(df, optical_frequency)
    add_source_power(df, zbd_fraction)
    df = archive.add_total_mmw_attenuator_turns(df)
    return df


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


def add_zbd_voltage(df):
    """
    The modulated square wave has minimum 0 and maximum V_z, equivalent to an offset square wave with peak V_z / 2.
    The lock-in measures the Fourier component at the reference frequency and reports RMS voltage, so in this case it
    will report
    V_l = (V_z / 2) (4 / \pi) 2^{-1/2} = (2^{1/2} / \pi) V_z.
    The peak ZBD voltage is thus
    V_z = 2^{-1/2} \pi V_l
    """
    df['zbd_voltage'] = 2**(-1/2) * np.pi * df.lockin_rms_voltage


def add_zbd_power(df, optical_frequency=None):
    """
    :param df: The dataframe.
    :param optical_frequency: The frequency in hertz detected by the ZBD; the default responsivity is close to the
    measured 150 GHz responsivity, and was previously used in most code.
    :return: None; the dataframe is modified.
    """
    if optical_frequency is None:
        zbd_volts_per_watt = 2200
    else:
        from equipment.vdi import zbd
        zbd_volts_per_watt = zbd.ZBD().responsivity(optical_frequency)
    df['zbd_power'] = df['zbd_voltage'] / zbd_volts_per_watt


def add_source_power(df, zbd_fraction):
    df['source_power'] = df.zbd_power / zbd_fraction


def add_resonator_fit_good(df, maximum_fractional_f_r_error, maximum_iQi_error):
    df['resonator_fit_good'] = ((df.f_r_err / df.f_r < maximum_fractional_f_r_error) &
                                (df.Q_i_err / df.Q_i**2 < maximum_iQi_error) &
                                (df.Q_i > 0))  # This shouldn't be necessary, but it currently is.

def add_temperature_groups(df,temperature_deviation_K=5e-3,temperature_field='sweep_primary_package_temperature',
                           debug=False):
    levels = []
    mask = None
    current_data = df[temperature_field]
    group_field = temperature_field+'_group'
    df.loc[:,group_field] = np.nan
    while True:
        level = current_data.iloc[0]
        new_mask = np.abs(df[temperature_field] - level) < temperature_deviation_K
        level_mean = df[temperature_field][new_mask].mean()
        level = np.round(level_mean/temperature_deviation_K)*temperature_deviation_K
        if debug:
            print "found level", level, "matching", (new_mask.sum()), "data points. Mean of this set:",level_mean
        df.loc[new_mask,group_field] = level
        if mask is None:
            mask = new_mask
        else:
            mask = mask | new_mask
        current_data = df[temperature_field][~mask]
        if debug:
            print len(current_data),"points remaining"
        levels.append(level)
        if len(current_data) == 0:
            return levels

# TODO: implement this in a separate module
def fit_XI_temperature_response():
    pass


# TODO: move the response fitting functions to another module and rename them.
def n_qp_over_n_star(P, P_0, P_star):
    return (1 + (P + P_0) / P_star) ** (1 / 2) - 1


def X(P, P_0, P_star, X_0):
    return X_0 * n_qp_over_n_star(P, P_0, P_star)


def X_model(params, P):
    return X(P, params['P_0'].value, params['P_star'].value, params['X_0'].value)


def dX_dP(P, P_0, P_star, X_0):
    return X_0 * (2 * P_star) ** -1 * (1 + (P + P_0) / P_star) ** (-1 / 2)


def X_inverse(X, P_0, P_star, X_0):
    return P_star * ((1 + X / X_0) ** 2 - 1) - P_0


def X_residual(params, P, X_data, X_errors):
    return (X_data - X_model(params, P)) / X_errors


def I(P, P_0, P_star, I_0, I_C):
    return I_0 * n_qp_over_n_star(P, P_0, P_star) + I_C


def I_model(params, P):
    return I(P, params['P_0'].value, params['P_star'].value, params['I_0'].value, params['I_C'].value)


def dI_dP(P, P_0, P_star, I_0):
    return I_0 * (2 * P_star) ** -1 * (1 + (P + P_0) / P_star) ** (-1 / 2)


def I_inverse(I, P_0, P_star, I_0, I_C):
    return P_star * ((1 + (I - I_C) / I_0) ** 2 - 1) - P_0


def I_residual(params, P, I_data, I_errors):
    return (I_data - I_model(params, P)) / I_errors


def residual(params, P, X_data, I_data, X_errors, I_errors):
    return np.concatenate((X_residual(params, P, X_data, X_errors),
                           I_residual(params, P, I_data, I_errors)))


def fit(P, X_data, I_data, initial, X_errors=1, I_errors=1, **kwargs):
    return lmfit.minimize(residual, initial, args=(P, X_data, I_data, X_errors, I_errors), **kwargs)


# TODO: write a function that actually guesses
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


def add_XI_response(df, key, masker=None):
    def XI_response(group):
        if masker is None:
            mask = np.ones(group.shape[0], dtype=np.bool)
        else:
            mask = masker(group)
        f_r_max_row = group[mask][group[mask].f_r == group[mask].f_r.max()]
        if f_r_max_row.shape[0] > 0:  # If no data is left after masking, return
            f_r_max = float(f_r_max_row.f_r)
            f_r_max_err = float(f_r_max_row.f_r_err)
            group.loc[mask, '{}_f_r_max'.format(key)] = f_r_max
            group.loc[mask, '{}_f_r_max_err'.format(key)] = f_r_max_err
            group.loc[mask, '{}_X'.format(key)] = f_r_max / group.f_r - 1
            group.loc[mask, '{}_X_err'.format(key)] = ((f_r_max_err / f_r_max)**2 + (group.f_r_err / group.f_r)**2)**(1/2)
            group.loc[mask, '{}_I'.format(key)] = group.Q_i**-1
            group.loc[mask, '{}_I_err'.format(key)] = group.Q_i**-2 * group.Q_i_err
        return group
    return df.groupby(('channel', 'atten')).apply(XI_response).reset_index(drop=True)


# Remove bad data and power-off points before fitting.
def fit_XI_power_response(df, initial, power, masker=lambda group: np.ones(group.shape[0], dtype=np.bool)):
    def XI_power_response(group):
        mask = masker(group)
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
    return df.groupby(('channel', 'atten')).apply(XI_power_response).reset_index(drop=True)


def add_NEP2(df, power):
    df['{}_NEP2_device'.format(power)] = df.noise_fit_device_noise / df['{}_dX_dP'.format(power)] ** 2
    df['{}_NEP2_amplifier'.format(power)] = df.noise_fit_amplifier_noise / df['{}_dX_dP'.format(power)] ** 2
