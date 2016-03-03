from collections import OrderedDict
from kid_readout.measurement.core import Measurement


class ADCSnap(Measurement):

    dimensions = OrderedDict([('epoch', ('epoch',)),
                              ('x', ('epoch',)),
                              ('y', ('epoch',))])

    # TODO: should x and y become Stream instances?
    def __init__(self, epoch, x, y, state=None, analyze=False, description='ADCSnap'):
        self.epoch = epoch
        self.x = x
        self.y = y
        super(ADCSnap, self).__init__(state, analyze, description)
