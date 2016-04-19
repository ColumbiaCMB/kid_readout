from kid_readout.measurement import core


class Hardware(object):

    def __init__(self, *args):
        for arg in args:
            setattr(self, arg.name, arg)

    def state(self):
        return core.StateDict([(obj.name, obj.state()) for obj in self.__dict__.values()])

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, ', '.join(self.__dict__.keys()))

