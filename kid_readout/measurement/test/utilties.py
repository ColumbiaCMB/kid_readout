"""
This module contains helper functions for running tests.
"""
import numpy as np
from kid_readout.measurement import core, legacy


def compare_measurements(a, b):
    """
    Recursively compare two measurements. At each level, the function tests that both instances have the same public
    attributes (meaning those that do not start with an underscore), that all these attributes are equal,
    and that the classes of the measurements are equal. The function does not test private variables at all,
    and does not even check whether the instances have the same private attributes.

    :param a: a Measurement instance.
    :param b: a Measurement instance.
    :return: None
    """
    keys_a = [k for k in a.__dict__.keys() if not k.startswith('_')]
    keys_b = [k for k in b.__dict__.keys() if not k.startswith('_')]
    for key_a in keys_a:
        assert key_a in keys_b
    for key_b in keys_b:
        assert key_b in keys_a
    keys_a.append("__class__")
    for key in keys_a:
        va = getattr(a, key)
        vb = getattr(b, key)
        if issubclass(va.__class__, core.Measurement):
            compare_measurements(va, vb)
        elif issubclass(va.__class__, core.MeasurementSequence):
            for n, ma in enumerate(va):
                mb = vb[n]
                compare_measurements(ma, mb)
        elif isinstance(va, np.ndarray):
            assert np.all(va == vb)
        else:
            assert va == vb


# TODO: replace this with a function that generates complex measurements.
def get_measurement():
    """
    from kid_readout.measurement.io import readoutnc
    nc_filename = '/data/readout/2015-05-12_113832_mmw_noise_broadband.nc'
    rnc = readoutnc.ReadoutNetCDF(nc_filename)
    index = 1
    return legacy.sweepstreamarray_from_rnc(rnc, index, index)
    """
    return core.Measurement({})