"""
This module is for functions that operate on DataFrames.

For consistency, all the functions should return the DataFrame.
"""
from __future__ import division
import numpy as np
import pandas as pd
import lmfit

# This should work except in pathological cases; groupby sorts by key automatically.
def add_channel(df):
    resonator_index_to_channel = np.array([group.f_0.mean() for index, group in df.groupby('resonator_index')]).argsort()
    df['channel'] = resonator_index_to_channel[np.array(df.resonator_index)]
    return df


def add_location(df, channel_to_location):
    df['location'] = channel_to_location[np.array(df.channel)]
    return df


def add_zbd_power(df, zbd_volts_per_watt=2200):
    df['zbd_power'] = df.zbd_voltage / zbd_volts_per_watt
    return df


# Mask bad data before calling.
def add_load_temperature_responsivities(df):
    def calculate_load_temperature_responsivities(group):
        group.sort('sweep_primary_load_temperature', inplace=True)
        mask = group.timestream_modulation_duty_cycle == 1  # 1 is source off and 0 is source on
        group.loc[mask, 'f_0_max'] = group[mask].f_0.max()
        group.loc[mask, 'x'] = group[mask].f_0 / group.f_0_max - 1
        T_load_x_fit = np.polyfit(group[mask].sweep_primary_load_temperature, group[mask].x, 1)
        group.loc[mask, 'dx_dT'] = T_load_x_fit[0]
        group.loc[mask, 'T_load_x_fit_1'] = T_load_x_fit[1]
        T_load_iQi_fit = np.polyfit(group[mask].sweep_primary_load_temperature, group[mask].Q_i**-1, 1)
        group.loc[mask, 'diQi_dT'] = T_load_iQi_fit[0]
        group.loc[mask, 'T_load_iQi_fit_1'] = T_load_iQi_fit[1]
        return group
    return df.groupby(('channel', 'atten')).apply(calculate_load_temperature_responsivities).reset_index(drop=True)


# TODO: move the fitting functions to another module and rename them.
def z(params, P):
    z_0 = params['z_0'].value
    P_0 = params['P_0'].value
    P_star = params['P_star'].value
    return z_0 * ((1 + (P + P_0) / P_star)**(1/2) - 1)

def dz_dP(params, P):
    z_0 = params['z_0'].value
    P_0 = params['P_0'].value
    P_star = params['P_star'].value
    return z_0 / (2 * P_star) * (1 + (P + P_0) / P_star)**(-1/2)

def z_residual(params, P, data, errors=1):
    return (data - z(params, P)) / errors

def fit_z(P, data, initial, errors=1):
    return lmfit.minimize(z_residual, initial, args=(P, data, errors))

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
    return P_star * ((1 + z / z_0)**2 - 1) - P_0


# Mask bad data before fitting.
def add_zbd_power_responsivities(df):
    def calculate_zbd_power_responsivities(group):
        group.sort('zbd_power', inplace=True)
        mask = group.timestream_modulation_duty_cycle == 0  # 1 is source off and 0 is source on
        group.loc[mask, 'f_0_max'] = group[mask].f_0.max()
        group.loc[mask, 'x'] = group.f_0 / group.f_0_max - 1

        P_zbd_x_fit = fit_z(group[mask].zbd_power, group[mask].x, z_initial(-10e-6, 10e-6, 10e-6))
        group.loc[mask, 'P_zbd_x_fit_z_0'] = P_zbd_x_fit.params['z_0'].value
        group.loc[mask, 'P_zbd_x_fit_P_0'] = P_zbd_x_fit.params['P_0'].value
        group.loc[mask, 'P_zbd_x_fit_P_star'] = P_zbd_x_fit.params['P_star'].value
        group.loc[mask, 'dx_dP'] = dz_dP(P_zbd_x_fit.params, group[mask].zbd_power)

        P_zbd_iQi_fit = fit_z(group[mask].zbd_power, 1 / group[mask].Q_i, z_initial(10e-6, 1e-6, 10e-6))
        group.loc[mask, 'P_zbd_iQi_fit_z_0'] = P_zbd_iQi_fit.params['z_0'].value
        group.loc[mask, 'P_zbd_iQi_fit_P_0'] = P_zbd_iQi_fit.params['P_0'].value
        group.loc[mask, 'P_zbd_iQi_fit_P_star'] = P_zbd_iQi_fit.params['P_star'].value
        group.loc[mask, 'diQi_dP'] = dz_dP(P_zbd_iQi_fit.params, group[mask].zbd_power)

        return group
    return df.groupby(('channel', 'atten')).apply(calculate_zbd_power_responsivities).reset_index(drop=True)


# Be warned that the noise functions currently calculate the noise differently: device_noise includes the amplifier
# noise, while noise_fit_device_noise does not.

def add_NET2(df):
    df['NET2_device'] = df.device_noise / (2 * df.dx_dT**2)
    df['NET2_amplifier'] = df.amplifier_noise / (2 * df.dx_dT**2)
    return df


def add_NET2_fit(df):
    df['NET2_fit_device'] = df.noise_fit_device_noise / (2 * df.dx_dT**2)
    df['NET2_fit_amplifier'] = df.noise_fit_amplifier_noise / (2 * df.dx_dT**2)
    return df


def add_NEP2_zbd(df):
    df['NEP2_zbd_device'] = df.device_noise / df.dx_dP**2
    df['NEP2_zbd_amplifier'] = df.amplifier_noise / df.dx_dP**2
    return df


def add_NEP2_zbd_fit(df):
    df['NEP2_zbd_fit_device'] = df.noise_fit_device_noise / df.dx_dP**2
    df['NEP2_zbd_fit_amplifier'] = df.noise_fit_amplifier_noise / df.dx_dP**2
    return df