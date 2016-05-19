"""
This module contains classes that encapsulate hardware and return its state in the proper format.

The Hardware class is a container for the classes that actually communicate with equipment.

The Thing class can be used to add state for hardware that doesn't have a class that communicates with it.

Examples
--------
> led = Thing('cold_led', {'current': 0.2})
> led.name
cold_led
> led.state
{'current': 0.2}
> hw = Hardware(led, <other equipment>)
> hw.state().cold_led.current  # Note that the `name` attribute enters the StateDict.
0.2
"""
import warnings
from collections import namedtuple

from kid_readout.measurement import core


# This is a list of names hardware that should always be included for ROACH measurements.
REQUIRED_HARDWARE = ['signal_conditioner']


# This is a template that can be used to add information to the state in the proper format; see above.
Thing = namedtuple('Thing', ['name', 'state'])


class Hardware(object):

    def __init__(self, *args, **kwargs):
        if not kwargs.get('quiet'):
            names = {arg.name for arg in args}
            if len(args) != len(names):
                raise ValueError("Duplicate name in {}".format(args))
            for required_name in REQUIRED_HARDWARE:
                if required_name not in names:
                    warnings.warn("You have not specified a '%s'; this will complicate later analysis." % required_name)
        for arg in args:
            setattr(self, arg.name, arg)

    def state(self, fast=False):
        """
        Get the state of all hardware.

        Parameters
        ----------
        fast: bool
            If True, get only minimal data that changes frequently. For now this applies only to the lock-in.

        Returns
        -------
        StateDict
            Contains an entry with state information for each piece of hardware.
        """
        state_ = {}
        for name, obj in self.__dict__.items():
            if fast:
                try:
                    state_[name] = obj.fast_state
                except AttributeError:
                    state_[name] = obj.state
            else:
                state_[name] = obj.state
        return core.StateDict(state_)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, ', '.join(self.__dict__.keys()))
