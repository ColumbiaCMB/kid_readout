from kid_readout.measurement.core import StateDict
__author__ = 'gjones'

class SignalConditioner(object):
    name = 'signal_conditioner'

    def state(self):
        return StateDict(description=self.description,
                         adc_chain_gain=self.adc_chain_gain,
                         dac_chain_gain=self.dac_chain_gain)


class Baseband(SignalConditioner):
    description = 'Baseband'
    def __init__(self,dac_chain_gain=-39, adc_chain_gain=0.0):
        self.dac_chain_gain = dac_chain_gain
        self.adc_chain_gain = adc_chain_gain

class HeterodyneMarkI(SignalConditioner):
    description = 'Mark I 1-2 GHz'
    def __init__(self,dac_chain_gain=-39, adc_chain_gain=0.0):
        self.dac_chain_gain = dac_chain_gain
        self.adc_chain_gain = adc_chain_gain


class HeterodyneMarkII(SignalConditioner):
    description = 'Mark II 0.5-4 GHz'
    def __init__(self,dac_chain_gain=-39, adc_chain_gain=0.0):
        self.dac_chain_gain = dac_chain_gain
        self.adc_chain_gain = adc_chain_gain
