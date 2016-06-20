import numpy as np

from collections import OrderedDict
from kid_readout.measurement.core import Measurement


class ADCSnap(Measurement):

    dimensions = OrderedDict([('x', ('sample',)),
                              ('y', ('sample',))])

    def __init__(self, epoch, x, y, state=None, description='', validate=True):
        self.epoch = epoch
        self.x = x
        self.y = y
        super(ADCSnap, self).__init__(state=state, description=description, validate=validate)

    @property
    def sample(self):
        return np.arange(self.x.size)



