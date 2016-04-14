from kid_readout.measurement import core


class MMWSweepStreams(core.Measurement):

    def __init__(self, sweep, streams, state, analyze=False, description='MMWSweepStreams'):
        self.sweep = sweep
        self.streams = streams
        super(MMWSweepStreams, self).__init__(state, analyze, description)

