"""
This module contains classes to load and save data used for testing resonators.
"""
from __future__ import division
import os
import time
from collections import OrderedDict

from kid_readout.measurement import core
from kid_readout.measurement.io import npy


DATA_DIRECTORY = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')


class DistilledSweep(core.Measurement):
    """
    This class defines a data format for storing reduced sweep data. Each of the streams is reduced to its mean value
    and error.
    """

    _version = 0

    dimensions = OrderedDict([('frequency', ('frequency',)),
                              ('s21_point', ('frequency',)),
                              ('s21_point_error', ('frequency',))])

    def __init__(self, frequency, s21_point, s21_point_error, roach_state, state=None, description=''):
        self.frequency = frequency
        self.s21_point = s21_point
        self.s21_point_error = s21_point_error
        self.roach_state = roach_state
        super(DistilledSweep, self).__init__(state=state, description=description)

    def fit_resonator(self, model, params=None):
        """
        Fit the s21 data with the given resonator model and, if given, the initial Parameters.

        Parameters
        ----------
        model : BaseResonator
            The resonator model to use for the fit.
        params : lmfit.Parameters
            A parameters object to use for initial values and limits in the fit.
        """
        resonator = model(frequency=self.frequency, s21=self.s21_point, errors=self.s21_point_error)
        resonator.fit(params)
        return resonator


def distill_sweep(single_sweep, roach_state_index=0, description=None):
    if description is None:
        description = single_sweep.description
    return DistilledSweep(frequency=single_sweep.frequency, s21_point=single_sweep.s21_point,
                          s21_point_error=single_sweep.s21_point_error,
                          roach_state=single_sweep.streams[roach_state_index].roach_state, state=single_sweep.state,
                          description=description)


def distill_and_save(single_sweep, roach_state_index=0, description=None, suffix='', io_class=npy.NumpyDirectory):
    """

    Parameters
    ----------
    single_sweep : measurement.basic.SingleSweep
    roach_state_index : int
    description : str
    suffix : str
    io_class : measurement.core.IO
        The IO subclass to use to save the data.

    Returns
    -------
    str
        The root path used to save the data.
    """
    basename = os.path.join(DATA_DIRECTORY, time.strftime('%Y-%m-%d_%H%M%S'))
    if suffix:
        basename += '_' + suffix
    root_path = basename + io_class.EXTENSION
    io = io_class(root_path)
    io.write(distill_sweep(single_sweep=single_sweep, roach_state_index=roach_state_index, description=description))
    io.close()
    return root_path



