"""
This module contains functions to handle data from Peter Day.
"""
from __future__ import division
from os.path import join, split, splitext
from glob import glob
import numpy as np

def read_sweep(filename):
    """
    Return f, S_21 for the given filename.
    """
    f, I, Q = np.loadtxt(filename, unpack=True)
    s21 = I + 1j * Q
    return f, s21

def read_all_sweeps(directory, pattern='tr*.txt'):
    """
    Return a dictionary of the form
    {1: (f, s21), etc.}
    """
    sweeps = {}
    for filename in glob(join(directory, pattern)):
        sweeps[int(splitext(split(filename)[1])[0][2:])] = read_sweep(filename)
    return sweeps
