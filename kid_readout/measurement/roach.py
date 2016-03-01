from kid_readout.measurement.core import Measurement


class ADCSnap(Measurement):

    def __init__(self, real, imag, state=None, analyze=False):
        self.real = real
        self.imag = imag
        super(ADCSnap, self).__init__(state, analyze)
