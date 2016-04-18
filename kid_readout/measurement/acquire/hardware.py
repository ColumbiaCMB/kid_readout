from collections import namedtuple
from kid_readout.measurement import core


def Hardware(*args):
    return namedtuple('Hardware', [arg.name for arg in args])(*args)


def state(hardware):
    return core.StateDict([(hw.name, hw.state()) for hw in hardware])
