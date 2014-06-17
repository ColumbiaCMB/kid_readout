"""
This module contains functions that load data taken by Peter and
Byeong-Ho at JPL.

The functions that load specific data sets contain plenty of magic
numbers, some of which fix problems specific to that data set.

The frequency data has units of Hz and the S21 data is complex and has
units of V / V, i.e., not dB.
"""

from __future__ import division
from os import path
from glob import glob
from collections import OrderedDict
import numpy as np

from kid_readout.analysis.resonator import Resonator
from kid_readout.analysis.khalil import generic_guess, generic_s21
from kid_readout.analysis.khalil import delayed_generic_s21, delayed_generic_guess


def read_sweep(filename):
    """
    Return a tuple (freq_data, s21_data) containing data from the
    given filename.
    """
    f, I, Q = np.loadtxt(filename, unpack=True)
    return f, I + 1j * Q


def read_all_sweeps(directory, pattern='tr*.txt'):
    """
    Return a list of tuples (freq_data, s21_data) from all text files
    in the given directory that match the given pattern.
    """
    sweeps = {}
    for filename in glob(path.join(directory, pattern)):
        sweeps[int(path.splitext(path.split(filename)[1])[0][2:])] = read_sweep(filename)
    return sweeps


def JPL_2014_May_light_blackbody_temperature(directory):
    """
    Parameter directory is a string that is the path to the unzipped
    contents of the file Log0522JAlSKIP20_bbT_sweep.zip, which is
    posted on the SKIP wiki.

    Return a dictionary with keys that are integers from 0 through 13
    and values that are lists of 14 Resonators fitted to frequency
    sweeps taken at blackbody temperatures from 4.2 K to 40 K in
    ascending order. Each list contains data for a single detector.

    The fits must include cable delay as a free parameter because it
    has not been removed.
    
    The frequency data in the files is in Hz and is converted to MHz
    here to match our data.

    The original data directory contains 150 sweeps, with indices
    ranging from 203 through 352, taken at ten different temperatures
    between 40 K and 4.2 K. There are 14 working resonators. Each
    group of sweeps contains one sweep of the entire frequency range
    then 14 sweeps of individual resonators in order of increasing
    resonance frequency. Sweep 203 thus contains a sweep of the entire
    band at the highest temperature of 40 K, and sweep 352 covers the
    resonator with highest resonance frequency at the lowest
    temperature of 4.2 K.

    Note that each list in the returned dictionary contains sweeps in
    order of increasing black body temperature.
    """
    log = path.join(directory, 'Log0522JAlSKIP20_bbT_sweep.txt')
    sweeps_per_group = 15
    # Each element of sweep_number is the XX in trXX.txt
    sweep_number, readout_power_dBm, bb_temp_K = np.loadtxt(log,
                                                            usecols=(0, 5, 9),
                                                            delimiter=' ',
                                                            skiprows=15,
                                                            unpack=True,
                                                            converters={0: lambda s: path.splitext(s)[0][2:]})
    sweeps = read_all_sweeps(directory)
    first_sweep = min(sweeps.keys())
    resonators = dict([(n, []) for n in range(sweeps_per_group - 1)])
    for n in range(len(sweeps)):
        if (n % sweeps_per_group) == 0: # skip the first sweep in each group
            continue
        else:
            r = Resonator(sweeps[n + first_sweep][0] / 1e6, sweeps[n + first_sweep][1], # Hz to MHz
                          guess = delayed_generic_guess, model = delayed_generic_s21)
            r.T_bath = 0.2 # The device temperature, as listed in the header.
            r.T_bb = bb_temp_K[n]
            r.P_readout = readout_power_dBm[n]
            resonators[(n % sweeps_per_group) - 1].insert(0, r) # Insert in ascending order
    return resonators


def JPL_2014_May_light_bath_temperature(directory):
    """
    Parameter directory is a string that is the path to the unzipped
    contents of the file Log0522JAlSKIP20_Tsweep.zip, which is
    posted on the SKIP wiki.

    Return a dictionary with keys that are integers from 0 through 13
    and values that are lists of 14 Resonators fitted to frequency
    sweeps taken at bath temperatures from 0.02 K to 0.4 K, in
    ascending order. Each list contains data for a single detector.


    The fits must include cable delay as a free parameter because it
    has not been removed.

    The frequency data in the files is in Hz and is converted to MHz
    here to match our data.
    
    The data file contains 300 sweeps, with indices ranging from 1218
    through 1517, taken at ten different bath temperatures between
    0.02 K and 0.4 K in steps of 0.02 K. There are 14 working
    resonators. Each group of sweeps contains one sweep of the entire
    frequency range then 14 sweeps of individual resonators in order
    of increasing resonance frequency. The groups of sweeps are taken
    in order of increasing bath temperature.  Sweep 1218 thus contains
    a sweep of the entire band at the lowest temperature of 0.02 K,
    and sweep 1517 covers the resonator with highest resonance
    frequency at the highest temperature of 0.4 K.
    """
    log = path.join(directory, 'Log0522JAlSKIP20_Tsweep.txt')
    sweeps_per_group = 15
    # Each element of sweep_number is the XX in trXX.txt
    sweep_number, readout_power_dBm, kid_temp_K = np.loadtxt(log,
                                                             usecols=(0, 5, 9),
                                                             delimiter=' ',
                                                             skiprows=15,
                                                             unpack=True,
                                                             converters={0: lambda s: path.splitext(s)[0][2:]})
    sweeps = read_all_sweeps(directory)
    first_sweep = min(sweeps.keys())
    resonators = dict([(n, []) for n in range(sweeps_per_group - 1)])
    for n in range(len(sweeps)):
        if (n % sweeps_per_group) == 0: # skip the first sweep in each group
            continue
        else:
            r = Resonator(sweeps[n + first_sweep][0] / 1e6, sweeps[n + first_sweep][1], # Hz to MHz
                          guess = delayed_generic_guess, model = delayed_generic_s21)
            r.T_bath = kid_temp_K[n]
            r.T_bb = 4.2 # The black body temperature, as reported by Peter.
            r.P_readout = readout_power_dBm[n]
            resonators[(n % sweeps_per_group) - 1].append(r)
    return resonators


def JPL_2013_August_dark(directory):
    """
    Return a list of 17 Resonators from Peter's sweeps of the first
    JPL chip.
    """
    sweeps =  read_all_sweeps(directory)
    # The 18th sweep is the full readout band.
    del(sweeps[18])
    split_9 = 150.025e6
    split_10 = 150.041e6
    resonators = []
    for n in range(1, 18):
        f, s21 = sweeps[n]
        if n == 9:
            mask = (f < split_9)
        elif n ==10:
            mask = (f > split_10)
        else:
            mask = np.ones_like(f).astype('bool')
        # Cable delay seems to be removed, so don't give the model an
        # extra free parameter.
        resonators.append(Resonator(f, s21, mask=mask,
                                    model=generic_s21,
                                    guess=generic_guess))
    return resonators


def JPL_2013_August_dark_bath_temperature(directory):
    """
    Return a list of Resonators from Peter's temperature sweeps of one
    resonance.
    """
    log = path.join(directory, 'Log0829LSKIP_T_sweep.txt')
    numbers, temp_list = np.loadtxt(log,
                                    usecols=(0, 9),
                                    delimiter=' ',
                                    skiprows=12,
                                    unpack=True,
                                    converters={0: lambda s: path.splitext(s)[0][2:]}) # extract XX from trXX.txt
    sweeps = read_all_sweeps(directory)
    resonators = []
    for n, T_bath in zip(numbers, temp_list):
        r = Resonator(sweeps[n][0], sweeps[n][1], guess = generic_guess, model = generic_s21)
        r.T_bath = T_bath
        resonators.append(r)
    return resonators

