from kid_readout.measurement import core


class Hardware(dict):

    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__
    __getstate__ = lambda: None
    __slots__ = ()

    def __init__(self, *args):
        super(Hardware, self).__init__([(arg.name, arg) for arg in args])

    def state(self):
        return core.StateDict([(equipment.name, equipment.state()) for equipment in self.values()])

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, ', '.join(self.keys()))
