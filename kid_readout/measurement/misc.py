from collections import OrderedDict
from kid_readout.measurement.core import Measurement


class ADCSnap(Measurement):

    dimensions = dict(x=('sample',),
                      y=('sample',))

    def __init__(self, epoch, x, y, state=None,description=''):
        self.epoch = epoch
        self.x = x
        self.y = y
        super(ADCSnap, self).__init__(state=state, description=description)

