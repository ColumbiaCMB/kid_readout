"""
This module contains classes that represent single-channel measurements.
"""
from __future__ import division
import numpy as np
from matplotlib.pyplot import mlab  # TODO: replace with a scipy PSD estimator
import pandas as pd
from kid_readout.measurement.core import Measurement, MeasurementTuple
from kid_readout.analysis import resonator
from kid_readout.utils.despike import deglitch_window


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
