from kid_readout.measurement import core,basic
import numpy as np


class MMWSweepList(basic.SweepStreamList):

    def __init__(self, sweep, stream_list, state, description=''):
        super(MMWSweepList, self).__init__(sweep=sweep,stream_list=stream_list,state=state,description=description)
    def single_sweep_stream_list(self,index):
        return MMWResponse(self.sweep.sweep(index),
                                     core.MeasurementList(sa.stream(index) for sa in self.stream_list),
                                     state=self.state, description=self.description)


class MMWResponse(basic.SingleSweepStreamList):
    def __init__(self, single_sweep, stream_list, state, description=''):
        super(MMWResponse,self).__init__(single_sweep=single_sweep,stream_list=stream_list,state=state,description=description)


    @property
    def lockin_voltage(self):
        return np.array(self.state_vector('lockin_voltage'),dtype='float')
    @property
    def hittite_frequency(self):
        return np.array(self.state_vector('hittite_frequency'),dtype='float')

    @property
    def mmw_frequency(self):
        return 12.*self.hittite_frequency

    def sweep_stream_list(self,deglitch=False):
        result = []
        for stream in self.stream_list:
            sss = basic.SingleSweepStream(sweep=self.sweep,stream=stream,state=stream.state,
                                        description=stream.description)
            sss._set_q_and_x(deglitch=deglitch)
            result.append(sss)
        return result

    def folded_x(self):
        sweep_stream_list = self.sweep_stream_list()
        result = []
        for sss in sweep_stream_list:
            fx = sss.fold(sss.x)
            result.append(fx)
        return np.array(result)

    def fractional_frequency_response(self):
        folded = self.folded_x()
        period = folded.shape[-1]
        return np.abs(folded[...,period//8:3*period//8].mean(-1) - folded[...,5*period//8:7*period//8].mean(-1))


class MMWSweepOnMod(core.Measurement):

    def __init__(self, sweep, on_stream, mod_stream, state=None, description=''):
        self.sweep = self.add_measurement(sweep)
        self.on_stream = self.add_measurement(on_stream)
        self.mod_stream = self.add_measurement(mod_stream)
        super(MMWSweepOnMod, self).__init__(state=state, description=description)
