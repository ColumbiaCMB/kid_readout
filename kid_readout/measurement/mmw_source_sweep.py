from __future__ import division
import time

import numpy as np
import pandas as pd
from memoized_property import memoized_property
# The ZBD object loads a few data files from disk. If this import fails then the functions that use it below will still
# work, but only with default arguments.
try:
    from equipment.vdi.zbd import ZBD
    zbd = ZBD()
except ImportError:
    zbd = None

from kid_readout.measurement import core, basic


class MMWSweepList(basic.SweepStreamList):

    def __init__(self, sweep, stream_list, state, description=''):
        super(MMWSweepList, self).__init__(sweep=sweep, stream_list=stream_list, state=state, description=description)

    def single_sweep_stream_list(self, index):
        return MMWResponse(self.sweep.sweep(index),
                           core.MeasurementList(sa.stream(index) for sa in self.stream_list),
                           number=index,
                           state=self.state, description=self.description)

    def to_dataframe(self,add_origin=True):
        rows = []
        for number in range(self.sweep.num_channels):
            sssl = self.single_sweep_stream_list(number)
            this_df = sssl.to_dataframe()
            rows.append(this_df)
        return pd.concat(rows,ignore_index=True)


class MMWResponse(basic.SingleSweepStreamList):

    def __init__(self, single_sweep, stream_list, state, number=0, description=''):
        super(MMWResponse,self).__init__(single_sweep=single_sweep, stream_list=stream_list, state=state, number=number,
                                         description=description)
    @property
    def lockin_rms_voltage(self):
        try:
            return np.array(self.state_vector('lockin','rms_voltage'),dtype='float')
        except KeyError:
            return np.nan

    def zbd_power(self, linearize=False):
        return zbd_voltage_to_power(self.zbd_voltage(linearize=linearize), mmw_frequency=self.mmw_frequency)

    def zbd_voltage(self, linearize=False):
        return lockin_rms_to_zbd_voltage(self.lockin_rms_voltage, linearize=linearize)

    @property
    def hittite_frequency(self):
        return np.array(self.state_vector('hittite','frequency'), dtype='float')

    @property
    def mmw_frequency(self):
        return 12.*self.hittite_frequency

    @memoized_property
    def sweep_stream_list(self):
        return self.get_sweep_stream_list()

    def get_sweep_stream_list(self, deglitch=False):
        result = []
        for stream in self.stream_list:
            sss = basic.SingleSweepStream(sweep=self.sweep, stream=stream, state=stream.state,
                                          description=stream.description)
            sss.set_q_and_x(deglitch=deglitch)
            result.append(sss)
        return result

    @memoized_property
    def folded_x(self):
        sweep_stream_list = self.sweep_stream_list
        result = []
        for sss in sweep_stream_list:
            fx = sss.fold(sss.x)
            # TODO: this is a hack
            phase = np.angle(np.fft.rfft(sss.x)[128])
            roll_by = int(np.round(phase*256/(2*np.pi)))
            result.append(np.roll(fx,-roll_by))
        return np.array(result)

    @memoized_property
    def folded_q(self):
        sweep_stream_list = self.sweep_stream_list
        result = []
        for sss in sweep_stream_list:
            fq = sss.fold(sss.q)
            result.append(fq)
        return np.array(result)

    @memoized_property
    def folded_normalized_s21(self):
        sweep_stream_list = self.sweep_stream_list
        result = []
        for sss in sweep_stream_list:
            fs21 = sss.fold(sss.normalized_s21)
            result.append(fs21)
        return np.array(result)

    @memoized_property
    def folded_s21_raw(self):
        sweep_stream_list = self.sweep_stream_list
        result = []
        for sss in sweep_stream_list:
            fs21 = sss.fold(sss.stream.s21_raw)
            result.append(fs21)
        return np.array(result)

    @memoized_property
    def fractional_frequency_response(self):
        return self.get_fractional_frequency_response()

    def get_fractional_frequency_response(self):
        folded = self.folded_x
        period = folded.shape[-1]
        template = np.ones((period,),dtype='float')
        template[:period//2] = -1
        response = np.abs(np.fft.irfft(np.fft.rfft(template)*np.fft.rfft(folded,axis=-1),axis=-1)*2./period).max(-1)
        return response

    def to_dataframe(self, add_origin=True):
        data = {'number': self.number, 'analysis_epoch':time.time(), 'start_epoch':self.start_epoch()}
        try:
            for thermometer, temperature in self.state['temperature'].items():
                data['temperature_{}'.format(thermometer)] = temperature
        except KeyError:
            pass
        try:
            for key, value in self.stream_list[0].roach_state.items():
                data['roach_{}'.format(key)] = value
        except KeyError:
            pass

        flat_state = self.state.flatten(wrap_lists=True)
        data.update(flat_state)

        for param in self.sweep.resonator.current_result.params.values():
            data['res_{}'.format(param.name)] = param.value
            data['res_{}_error'.format(param.name)] = param.stderr
        data['res_redchi'] = self.sweep.resonator.current_result.redchi
        data['res_Q_i'] = self.sweep.resonator.Q_i
        data['res_Q_e'] = self.sweep.resonator.Q_e

        data['res_s21_data'] = [self.sweep.resonator.data]
        data['res_frequency_data'] = [self.sweep.resonator.frequency]
        data['res_s21_errors'] = [self.sweep.resonator.errors]
        modelf = np.linspace(self.sweep.resonator.frequency.min(),self.sweep.resonator.frequency.max(),1000)
        models21 = self.sweep.resonator.model.eval(params=self.sweep.resonator.current_params,f=modelf)
        data['res_model_frequency'] = [modelf]
        data['res_model_s21'] = [models21]

        data['fractional_frequency_response'] = [self.fractional_frequency_response]
        data['folded_s21_raw'] = [self.folded_s21_raw]
        data['folded_x'] = [self.folded_x]

        data['mmw_frequency'] = [self.mmw_frequency]
        data['lockin_rms_voltage'] = [self.lockin_rms_voltage]
        data['zbd_power_linearized'] = [self.zbd_power(linearize=True)]

        dataframe = pd.DataFrame(data, index=[0])
        if add_origin:
            self.add_origin(dataframe)
        return dataframe



class MMWSweepOnMod(core.Measurement):

    _version = 0

    def __init__(self, sweep, off_stream, on_stream, mod_stream, state=None, description=''):
        self.sweep = sweep
        self.on_stream = on_stream
        self.mod_stream = mod_stream
        if off_stream:
            self.off_stream = off_stream
        else:
            self.off_stream = None
        super(MMWSweepOnMod, self).__init__(state=state, description=description)

    @property
    def on_sweep_stream_array(self):
        return basic.SweepStreamArray(sweep_array=self.sweep, stream_array=self.on_stream,state=self.state,
                                      description=self.description)
    @property
    def off_sweep_stream_array(self):
        if self.off_stream:
            return basic.SweepStreamArray(sweep_array=self.sweep, stream_array=self.off_stream,state=self.state,
                                          description=self.description)
        else:
            raise AttributeError("No off stream measurement defined")
    @property
    def mod_sweep_stream_array(self):
        return basic.SweepStreamArray(sweep_array=self.sweep, stream_array=self.mod_stream,state=self.state,
                                      description=self.description)

    def sweep_stream_set(self,number):
        sweep = self.sweep.sweep(number)
        on_sweep_stream = self.on_stream.stream(number)
        mod_sweep_stream = self.mod_stream.stream(number)
        try:
            off_sweep_stream = self.off_stream.stream(number)
        except AttributeError:
            off_sweep_stream = None
        if off_sweep_stream:
            return (basic.SingleSweepStream(sweep,off_sweep_stream,number=number,state=self.state,
                                            description=self.description),
                    basic.SingleSweepStream(sweep,on_sweep_stream,number=number,state=self.state,
                                            description=self.description),
                    basic.SingleSweepStream(sweep,mod_sweep_stream,number=number,state=self.state,
                                            description=self.description),
                    )
        else:
            return (None,
                    basic.SingleSweepStream(sweep,on_sweep_stream,number=number,state=self.state,
                                            description=self.description),
                    basic.SingleSweepStream(sweep,mod_sweep_stream,number=number,state=self.state,
                                            description=self.description),
                    )

    def to_dataframe(self, add_origin=True):
        on_rows = []
        mod_rows = []
        off_rows = []
        for n in range(self.sweep.num_channels):
            off_ss, on_ss, mod_ss = self.sweep_stream_set(n)
            on_rows.append(on_ss.to_dataframe(add_origin=False))
            mod_rows.append(mod_ss.to_dataframe(deglitch=False,add_origin=False))
            if off_ss:
                off_rows.append(off_ss.to_dataframe(add_origin=False))
        df_on = pd.concat(on_rows,ignore_index=True)
        df_mod = pd.concat(mod_rows,ignore_index=True)
        dfs = [df_on,df_mod]
        if off_rows:
            df_off = pd.concat(off_rows,ignore_index=True)
            dfs.append(df_off)
        else:
            df_off = None
        if add_origin:
            if self._io is None:
                self.sweep.add_origin(df_on,prefix='sweep_')
                self.on_stream.add_origin(df_on,prefix='stream_')
                self.sweep.add_origin(df_mod,prefix='sweep_')
                self.mod_stream.add_origin(df_mod,prefix='stream_')
                if off_rows:
                    self.sweep.add_origin(df_off,prefix='sweep_')
                    self.off_stream.add_origin(df_off,prefix='stream_')
            else:
                self.add_origin(df_on)
                self.add_origin(df_mod)
                if off_rows:
                    self.add_origin(df_off)

        df_on['lockin_rms_voltage'] = df_mod['lockin_rms_voltage']
        if df_off is not None:
            df_off['lockin_rms_voltage'] = df_mod['lockin_rms_voltage']

        return pd.concat(dfs,ignore_index=True)
        
        
def lockin_rms_to_zbd_voltage(lockin_rms_voltage, linearize=False):
    zbd_voltage = (np.pi / np.sqrt(2)) * lockin_rms_voltage
    if linearize:
        zbd_voltage /= zbd.linearity(zbd_voltage)
    return zbd_voltage


def zbd_voltage_to_power(zbd_voltage, mmw_frequency=None):
    if mmw_frequency is None:
        volts_per_watt = 2200  # 2200 V/W is the approximate responsivity
    else:
        volts_per_watt = zbd.responsivity(mmw_frequency)
    return zbd_voltage / volts_per_watt
