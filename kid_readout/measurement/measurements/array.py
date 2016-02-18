"""
This module contains classes that represent single-channel measurements.
"""
from __future__ import division
from kid_readout.measurement.core import Measurement


class StreamArray(Measurement):
    """
    This class is an array of streams.
    """
    pass


class SweepArray(Measurement):
    """
    This class is an array of sweeps.
    """
    pass
