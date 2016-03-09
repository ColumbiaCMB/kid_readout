"""
This module contains helper functions and data structures for running tests.
"""
import numpy as np
from kid_readout.measurement import core, legacy


corners = {'None': None, 'True': True, 'False': False,
           'empty_list': [],
           'int_list': [-1, 0, 1, 2],
           'float_list': [-0.1, 1, np.pi],
           'str_list': ['zero', 'one', 'two', ''],
           'bool_list': [False, True, False],
           'none_dict': {'None': None},
           'dict_dict': {'1': 1, 'dict': {'None': None, 'False': False, 'another_dict': {}}},
           'list_dict': {'empty_list': [],
                         'int_list': [-1, 0, 1, 2],
                         'float_list': [-0.1, 1, np.pi],
                         'str_list': ['zero', 'one', 'two', ''],
                         'bool_list': [False, True, False]}}


def compare_measurements(a, b, verbose=False):
    """
    Recursively compare two measurements. At each level, the function tests that both instances have the same public
    attributes (meaning those that do not start with an underscore), that all these attributes are equal,
    and that the classes of the measurements are equal. The function does not test private variables at all,
    and does not even check whether the instances have the same private attributes.

    :param a: a Measurement instance.
    :param b: a Measurement instance.
    :return: None
    """
    keys_a = [k for k in a.__dict__ if not k.startswith('_')] + ['__class__']
    keys_b = [k for k in b.__dict__ if not k.startswith('_')] + ['__class__']
    assert set(keys_a) == set(keys_b)
    for key in keys_a:
        va = getattr(a, key)
        vb = getattr(b, key)
        if verbose:
            print("{}: {}, {}".format(key, type(va), repr(va)))
        if issubclass(va.__class__, core.Measurement):
            compare_measurements(va, vb, verbose=verbose)
        elif issubclass(va.__class__, core.MeasurementSequence):
            for ma, mb in zip(va, vb):
                compare_measurements(ma, mb, verbose=verbose)
        elif isinstance(va, np.ndarray):  # This allows declared arrays to contain NaN and still compare correctly.
            assert np.all(np.isnan(va) == np.isnan(vb))
            assert np.all(va[~np.isnan(va)] == vb[~np.isnan(vb)])
        else:  # This will fail for NaN or sequences that contain any NaN values.
            assert va == vb


# TODO: replace this with a function that generates complex measurements without reading data from disk.
def get_measurement():
    #"""
    from kid_readout.measurement.io import readoutnc
    nc_filename = '/data/readout/2015-05-12_113832_mmw_noise_broadband.nc'
    rnc = readoutnc.ReadoutNetCDF(nc_filename)
    index = 1
    m = legacy.sweepstreamarray_from_rnc(rnc, index, index)
    rnc.close()
    m.state['corners'] = corners
    return m
    """
    m = core.Measurement(corners)
    m.int_list = range(-1, 3)
    m.float_list = list(np.linspace(-1, 2, 10))
    m.str_list = ['one', 'two', 'three']
    m.bool_list = [True, False]
    return m
    """
