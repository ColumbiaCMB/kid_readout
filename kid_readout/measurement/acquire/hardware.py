import warnings
from kid_readout.measurement import core


class Hardware(object):

    def __init__(self, *args, **kwargs):
        if not kwargs.pop('quiet'):
            names = {arg.name for arg in args}
            for required_name in ['signal_conditioner']:
                warnings.warn("You have not specified a '%s'; this will complicate later analysis." % required_name)
        for arg in args:
            setattr(self, arg.name, arg)

    def state(self, fast=False):
        """
        Get the state of all hardware

        Parameters
        ----------
        fast: bool, if True, only get minimal data that changes frequently. For now this only applies to the lockin

        Returns
        -------

        """
        state_ = {}
        #  this implementation might seem a bit ugly, but it works fine and is clear what is happening
        for obj in self.__dict__.values():
            if fast and obj.name == 'lockin':
                state_[obj.name] = obj.state(measurement_only=True)
            else:
                state_[obj.name] = obj.state()
        return core.StateDict(state_)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, ', '.join(self.__dict__.keys()))

