from __future__ import division
import numpy as np
import pandas as pd
from matplotlib.pyplot import mlab
from kid_readout.measurement.core import Measurement, get_class
#from kid_readout.measurement.measurements.single import Sweep, Stream, SweepStream
#from kid_readout.measurement.measurements.array import SweepArray, StreamArray, SweepStreamArray


class _SweepNoiseMeasurement(Measurement):
    """
    This class is intended to be a port of SweepNoiseMeasurement code to the Measurement paradigm.

    It's currently broken because it's nearly a copy-paste of the SweepNoiseMeasurement code.
    """
    def __init__(self):
        """
        sweep_filename : str
            NetCDF4 file with at least the sweep data. By default, this is also used for the timestream data.

        sweep_group_index : int (default 0)
            Index of the sweep group to process

        timestream_filename : str or None (optional)
            If None, use sweep_filename for the timestream data
            otherwise, this is the NetCDF4 file with the timestream data

        timestream_group_index : int (default 0)
            Index of the timestream group to process

        resonator_index : int (default 0)
            index of the resonator to process data for

        low_pass_cutoff_Hz : float
            Cutoff frequency used for low pass filtering the timeseries for display

        dac_chain_gain : float
            Estimate of the gain from output of DAC to device.
            Default value of -49 represents -2 dB loss intrinsic to analog signal conditioning
            board, 7 dB misc cable loss and 40 dB total cold
            attenuation.

        delay_estimate : float
            Estimate of basic cable delay to help fitting proceed more smoothly. Default value
            is appropriate for the wideband noise firmware

        deglitch_threshold : float
            Threshold for glitch detection in units of median absolute deviation.

        cryostat : str
            (Optional) Override the cryostat used to take this data. By default, the cryostat
            is guessed based on the machine you are processing data on. The guess is made by
            the get_experiment_info_at function


        """

    def analyze(self):
        if type(sweep_filename) is str:
            self.sweep_filename = sweep_filename
            if timestream_filename:
                self.timestream_filename = timestream_filename
            else:
                self.timestream_filename = self.sweep_filename
            self._sweep_file = None
            self._timestream_file = None
            self._close_file = True
        else:
            # TODO: Fix this to allow different timestream file
            self._sweep_file = sweep_filename
            self._timestream_file = sweep_filename
            self._close_file = False
            self.sweep_filename = self._sweep_file.filename
            self.timestream_filename = self._timestream_file.filename
        self.sweep_group_index = sweep_group_index
        self.timestream_group_index = timestream_group_index
        self._use_sweep_group_timestream = use_sweep_group_timestream

        self.sweep_epoch = self.sweep.start_epoch
        pkg1,pkg2,load1,load2 = get_temperatures_at(self.sweep.start_epoch)
        self.sweep_primary_package_temperature = pkg1
        self.sweep_secondary_package_temperature = pkg2
        self.sweep_primary_load_temperature = load1
        self.sweep_secondary_load_temperature = load2
        self.start_temp = self.sweep_primary_package_temperature
        self.resonator_index = resonator_index

        info = experiments.get_experiment_info_at(self.sweep_epoch, cryostat=cryostat)
        self.experiment_description = info['description']
        self.experiment_info = info
        self.chip_name = info['chip_id']
        self.is_dark = info['is_dark']
        self.optical_state = info['optical_state']
        self.dac_chain_gain = dac_chain_gain
        self.mmw_atten_turns = self._sweep_file.mmw_atten_turns

        try:
            self.atten, self.total_dac_atten = self._sweep_file.get_effective_dac_atten_at(self.sweep_epoch)
            self.power_dbm = dac_chain_gain - self.total_dac_atten
        except:
            print "failed to find attenuator settings"
            self.atten = np.nan
            self.total_dac_atten = np.nan
            self.power_dbm = np.nan

        # This uses the error calculation in readoutnc.SweepGroup
        self.sweep_freqs_MHz, self.sweep_s21, self.sweep_errors = self.sweep.select_by_index(resonator_index)
        self.conjugate_data = conjugate_data
        if conjugate_data:
            self.sweep_s21 = np.conj(self.sweep_s21)

        # find the time series that was measured closest to the sweep frequencies
        # this is a bit sloppy...
        if self._use_sweep_group_timestream:
            if delay_estimate is None:
                self.delay_estimate_microseconds = self._sweep_file.get_delay_estimate()*1e6
            else:
                self.delay_estimate_microseconds = delay_estimate


            if mask_sweep_indicies is None:
                rr = fit_best_resonator(self.sweep_freqs_MHz,self.sweep_s21,errors=self.sweep_errors,
                                        delay_estimate=self.delay_estimate_microseconds)
            else:
                mask = np.ones(self.sweep_s21.shape,dtype=np.bool)
                mask[mask_sweep_indicies] = False
                rr = fit_best_resonator(self.sweep_freqs_MHz[mask],self.sweep_s21[mask],errors=self.sweep_errors[mask],
                                        delay_estimate=self.delay_estimate_microseconds)
            timestream_index = np.argmin(abs(self.timestream.measurement_freq-rr.f_0))
        else:
            timestream_index = np.argmin(abs(self.timestream.measurement_freq-self.sweep_freqs_MHz.mean()))
        self.timestream_index = timestream_index

        original_timeseries = self.timestream.get_data_index(timestream_index)
        if conjugate_data:
            original_timeseries = np.conj(original_timeseries)
        self.adc_sampling_freq_MHz = self.timestream.adc_sampling_freq[timestream_index]
        self.noise_measurement_freq_MHz = self.timestream.measurement_freq[timestream_index]
        self.nfft = self.timestream.nfft[timestream_index]
        self.timeseries_sample_rate = self.timestream.sample_rate[timestream_index]
        self.timestream_modulation_duty_cycle = self.timestream.modulation_duty_cycle[timestream_index]
        self.timestream_modulation_freq = self.timestream.modulation_freq[timestream_index]
        self.timestream_modulation_phase = self.timestream.modulation_phase[timestream_index]
        self.timestream_modulation_period_samples = self.timestream.modulation_period_samples[timestream_index]
        if not self._use_sweep_group_timestream:
            self.timestream_mmw_source_freq = self.timestream.mmw_source_freq[timestream_index]
            old_style_source_modulation_freq = self.timestream.mmw_source_modulation_freq[timestream_index]
        else:
            self.timestream_mmw_source_freq = np.nan
            old_style_source_modulation_freq = np.nan

        if (np.isfinite(old_style_source_modulation_freq) and
            (old_style_source_modulation_freq != self.timestream_modulation_freq) and
            (old_style_source_modulation_freq != 0)):
            print ("found old style modulation frequency", old_style_source_modulation_freq,
                   "which doesn't match the new style",
                   self.timestream_modulation_freq,"using the old style value")
            self.timestream_modulation_freq = old_style_source_modulation_freq
            self.timestream_modulation_period_samples = int(self.timeseries_sample_rate/old_style_source_modulation_freq)
            self.timestream_modulation_duty_cycle = 0.5

        self.timestream_epoch = self.timestream.epoch[timestream_index]
        self.timestream_duration = original_timeseries.shape[0]/self.timeseries_sample_rate
        # The following hack helps fix a long standing timing bug which was recently fixed/improved
        if self.timestream_epoch < 1399089567:
            self.timestream_epoch -= self.timestream_duration
        # end hack
        self.timestream_temperatures_sample_times = np.arange(self.timestream_duration)
        pkg1,pkg2,load1,load2 = get_temperatures_at(self.timestream_epoch + self.timestream_temperatures_sample_times)
        self.timestream_primary_package_temperature = pkg1
        self.timestream_secondary_package_temperature = pkg2
        self.timestream_primary_load_temperature = load1
        self.timestream_secondary_load_temperature = load2
        self.end_temp = self.timestream_primary_package_temperature[-1]


        # We can use the timestream measurement as an additional sweep point.
        # We average only the first 2048 points of the timeseries to avoid any drift.
        if False:
            self.sweep_freqs_MHz = np.hstack((self.sweep_freqs_MHz,[self.noise_measurement_freq_MHz]))
            self.sweep_s21 = np.hstack((self.sweep_s21,[original_timeseries[:2048].mean()]))
            self.sweep_errors = np.hstack((self.sweep_errors,
                                               [original_timeseries[:2048].real.std()/np.sqrt(2048)
                                                +1j*original_timeseries[:2048].imag.std()/np.sqrt(2048)]))

        # Now put all the sweep data in increasing frequency order so it plots nicely
        order = self.sweep_freqs_MHz.argsort()
        self.sweep_freqs_MHz = self.sweep_freqs_MHz[order]
        self.sweep_s21 = self.sweep_s21[order]
        self.sweep_errors = self.sweep_errors[order]

        if delay_estimate is None:
            self.delay_estimate_microseconds = self._sweep_file.get_delay_estimate()*1e6
        else:
            self.delay_estimate_microseconds = delay_estimate


        if mask_sweep_indicies is None:
            rr = fit_best_resonator(self.sweep_freqs_MHz,self.sweep_s21,errors=self.sweep_errors,
                                    delay_estimate=self.delay_estimate_microseconds)
        else:
            mask = np.ones(self.sweep_s21.shape,dtype=np.bool)
            mask[mask_sweep_indicies] = False
            rr = fit_best_resonator(self.sweep_freqs_MHz[mask],self.sweep_s21[mask],errors=self.sweep_errors[mask],
                                    delay_estimate=self.delay_estimate_microseconds)
        self._resonator_model = rr
        self.Q_i = rr.Q_i
        self.Q_i_err = qi_error(rr.result.params['Q'].value, rr.result.params['Q'].stderr,
                                rr.result.params['Q_e_real'].value, rr.result.params['Q_e_real'].stderr,
                                rr.result.params['Q_e_imag'].value, rr.result.params['Q_e_imag'].stderr)
        self.fit_params = rr.result.params

        decimation_factor = self.timeseries_sample_rate/low_pass_cutoff_Hz
        normalized_timeseries = rr.normalize(self.noise_measurement_freq_MHz,original_timeseries)
        self.low_pass_normalized_timeseries = low_pass_fir(normalized_timeseries, num_taps=1024, cutoff=low_pass_cutoff_Hz,
                                                          nyquist_freq=self.timeseries_sample_rate, decimate_by=decimation_factor)
        self.normalized_timeseries_mean = normalized_timeseries.mean()

        projected_timeseries = rr.project_s21_to_delta_freq(self.noise_measurement_freq_MHz,normalized_timeseries,
                                                            s21_already_normalized=True)

        # calculate the number of samples for the deglitching window.
        # the following will be the next power of two above 1 second worth of samples
        window = int(2**np.ceil(np.log2(self.timeseries_sample_rate)))
        # reduce the deglitching window if we don't have enough samples
        if window > projected_timeseries.shape[0]:
            window = projected_timeseries.shape[0]//2
        self.deglitch_window = window
        self.deglitch_threshold = deglitch_threshold
        if deglitch_threshold:
            deglitched_timeseries = deglitch_window(projected_timeseries,window,thresh=deglitch_threshold)
        else:
            deglitched_timeseries = projected_timeseries

        # TODO: should nyquist_freq be half the sample rate?
        self.low_pass_projected_timeseries = low_pass_fir(deglitched_timeseries, num_taps=1024, cutoff=low_pass_cutoff_Hz,
                                                nyquist_freq=self.timeseries_sample_rate, decimate_by=decimation_factor)
        self.low_pass_timestep = decimation_factor/self.timeseries_sample_rate

        self.normalized_model_s21_at_meas_freq = rr.normalized_model(self.noise_measurement_freq_MHz)
        self.normalized_model_s21_at_resonance = rr.normalized_model(rr.f_0)
        self.normalized_ds21_df_at_meas_freq = rr.approx_normalized_gradient(self.noise_measurement_freq_MHz)

        self.sweep_normalized_s21 = rr.normalize(self.sweep_freqs_MHz,self.sweep_s21)

        self.sweep_model_freqs_MHz = np.linspace(self.sweep_freqs_MHz.min(),self.sweep_freqs_MHz.max(),1000)
        self.sweep_model_normalized_s21 = rr.normalized_model(self.sweep_model_freqs_MHz)
        self.sweep_model_normalized_s21_centered = self.sweep_model_normalized_s21 - self.normalized_timeseries_mean

        fractional_fluctuation_timeseries = deglitched_timeseries / (self.noise_measurement_freq_MHz*1e6)
        self._fractional_fluctuation_timeseries = fractional_fluctuation_timeseries
        fr,S,evals,evects,angles,piq = iqnoise.pca_noise(fractional_fluctuation_timeseries,
                                                         NFFT=None, Fs=self.timeseries_sample_rate)

        self.pca_freq = fr
        self.pca_S = S
        self.pca_eigvals = evals
        self.pca_eigvects = evects
        self.pca_angles = angles
        self.pca_piq = piq

        self.freqs_coarse,self.prr_coarse,self.pii_coarse = self.get_projected_fractional_fluctuation_spectra(NFFT=2**12)

        self._normalized_timeseries = normalized_timeseries[:2048].copy()
        self._close_files()

    @property
    def original_timeseries(self):
        """
        get the original demodulated timeseries
        """
        try:
            if self.conjugate_data:
                return np.conj(self.timestream.get_data_index(self.timestream_index))
        except AttributeError:
            pass
        return self.timestream.get_data_index(self.timestream_index)

    @property
    def normalized_timeseries(self):
        """
        get the timeseries after normalizing it to remove the arbitrary amplitude, phase, and delay corrections
        determined by the resonator fit
        """
        return self.resonator_model.normalize(self.noise_measurement_freq_MHz,self.original_timeseries)

    @property
    def projected_timeseries(self):
        """
        get the timeseries after projecting it into the dissipation and frequency directions (in units of Hz)
        determined by the resonator fit
        """
        return self.resonator_model.project_s21_to_delta_freq(self.noise_measurement_freq_MHz,self.normalized_timeseries,
                                                            s21_already_normalized=True)

    @property
    def fractional_fluctuation_timeseries(self):
        """
        get the fractional fluctuation timeseries (projected divided by resonant frequency)
        """
        if self._fractional_fluctuation_timeseries is None:
            self._fractional_fluctuation_timeseries = self.get_deglitched_timeseries()/(self.noise_measurement_freq_MHz*1e6)
        return self._fractional_fluctuation_timeseries

    def get_deglitched_timeseries(self,window_in_seconds=1.0, thresh=None):
        """
        Get the deglitched, projected timeseries

        window_in_seconds : float
            deglitching window duration

        thresh : float
            threshold in median absoute deviations.
        """
        # calculate the number of samples for the deglitching window.
        # the following will be the next power of two above 1 second worth of samples
        window = int(2**np.ceil(np.log2(window_in_seconds*self.timeseries_sample_rate)))
        # reduce the deglitching window if we don't have enough samples
        projected_timeseries = self.projected_timeseries
        if window > projected_timeseries.shape[0]:
            window = projected_timeseries.shape[0]//2

        if thresh is None:
            thresh = self.deglitch_threshold
        if thresh is None:
            deglitched_timeseries = projected_timeseries
        else:
            deglitched_timeseries = deglitch_window(projected_timeseries,window,thresh=thresh)

        return deglitched_timeseries

    def get_projected_fractional_fluctuation_spectra(self,NFFT=2**12,window=mlab.window_none):
        """
        Calculate the PSD of the fractional fluctuation timeseries using mlab.psd.

        NFFT : int
            length of the FFT to compute. Sets freuqency resolution.

        window : function
            windowing function. i.e. mlab.window_hamming etc.

        Returns: freqs,prr,pii

        freqs : float array
            Frequencies in Hz

        prr : float array
            dissipation spectrum. Units are 1/Hz

        pii : float array
            frequency fluctuation spectrum. Units are 1/Hz
        """
        prr,freqs = mlab.psd(self.fractional_fluctuation_timeseries.real,NFFT=NFFT,
                                                           window=window,Fs=self.timeseries_sample_rate)
        pii,freqs = mlab.psd(self.fractional_fluctuation_timeseries.imag,NFFT=NFFT,
                                                           window=window,Fs=self.timeseries_sample_rate)
        return freqs,prr,pii

    @property
    def sweep(self):
        """
        Get the sweep group from the netcdf file
        """
        self._open_sweep_file()
        return self._sweep_file.sweeps[self.sweep_group_index]

    @property
    def timestream(self):
        """
        Get the timestream group from the netcdf file
        """
        if hasattr(self,'_use_sweep_group_timestream') and self._use_sweep_group_timestream:
            self._open_sweep_file()
            return self._sweep_file.sweeps[self.sweep_group_index].timestream_group
        else:
            self._open_timestream_file()
            return self._timestream_file.timestreams[self.timestream_group_index]

    @property
    def resonator_model(self):
        if self._resonator_model is None:
            self._restore_resonator_model()
        return self._resonator_model

    def __getstate__(self):
        """
        For pickling, make sure we get rid of everything unpicklable and large
        """
        self._close_files()
        d = self.__dict__.copy()
#        del d['_sweep_file']
#        del d['_timestream_file']
#        del d['sweep']
#        del d['timestream']
        d['_resonator_model'] = None
        d['_fractional_fluctuation_timeseries'] = None
        return d

    def __setstate__(self,state):
        """
        Restore from pickling
        """
        self.__dict__ = state
#        try:
#            self._open_netcdf_files()
#        except IOError:
#            print "Warning: could not open associated NetCDF datafiles when unpickling."
#            print "Some features of the class will not be available"
#        try:
#            self._restore_resonator_model()
#        except Exception, e:
#            print "error while restoring resonator model:",e

    def _open_sweep_file(self):
        if self._sweep_file is None:
            self._sweep_file = readoutnc.ReadoutNetCDF(find_nc_file(self.sweep_filename))

    def _open_timestream_file(self):
        if self._timestream_file is None:
            self._timestream_file = readoutnc.ReadoutNetCDF(find_nc_file(self.timestream_filename))

    def _close_files(self):
        if self._sweep_file:
            if self._close_file:
                self._sweep_file.close()
            self._sweep_file = None
        if self._timestream_file:
            if self._close_file:
                self._timestream_file.close()
            self._timestream_file = None

    def _restore_resonator_model(self):
        self._resonator_model = fit_best_resonator(self.sweep_freqs_MHz,self.sweep_s21,errors=self.sweep_errors,
                                                  delay_estimate=self.fit_params['delay'].value)


    def to_dataframe(self):
        data = {}
        for param in self.fit_params.values():
            data[param.name] = [param.value]
            data[param.name+'_err'] = [param.stderr]

        attrs = self.__dict__.keys()
        attrs.remove('fit_params')

        private = [x for x in attrs if x.startswith('_')]
        for private_var in private:
            attrs.remove(private_var)
        for pn in attrs:
            data[pn]= [getattr(self,pn)]

        return pd.DataFrame(data,index=[0])

# These functions are intended to use the new code to read old data.

def stream_from_rnc(rnc, stream_index, channel):
    tg = rnc.timestreams[stream_index]
    tg_channel_index = tg.measurement_freq.argsort()[channel]
    stream = get_class('Stream')(tg.measurement_freq[tg_channel_index],
                                 tg.get_data_index(tg_channel_index),
                                 tg.epoch[tg_channel_index],
                                 tg.epoch[tg_channel_index] + tg.data_len_seconds[tg_channel_index])
    return stream


def streamarray_from_rnc(rnc, stream_index):
    tg = rnc.timestreams[stream_index]
    tg_channel_index = tg.measurement_freq.argsort()
    frequency = tg.measurement_freq[tg_channel_index]
    epoch = np.linspace(tg.epoch[tg_channel_index],
                        tg.epoch[tg_channel_index] + tg.data_len_seconds[tg_channel_index],
                        n_samples)
    stream = get_class('Stream')(tg.measurement_freq[tg_channel_index],
                                 tg.get_data_index(tg_channel_index),
                                 )
    return StreamArray(frequency, epoch, s21)


def sweep_from_rnc(rnc, sweep_index, channel):
    sg = rnc.sweeps[sweep_index]
    n_channels = np.unique(sg.index).size
    if not sg.frequency.size % n_channels == 0:
        raise ValueError("Bad number of frequency points.")
    frequencies_per_index = int(sg.frequency.size / n_channels)
    streams = []
    for i in range(frequencies_per_index * channel,
                   frequencies_per_index * (channel + 1)):
        streams.append(get_class('Stream')(sg.timestream_group.measurement_freq[i],
                                           sg.timestream_group.data[i,:],
                                           sg.timestream_group.epoch[channel],
                                           sg.timestream_group.epoch[channel] +
                                           sg.timestream_group.data_len_seconds[channel]))
    sweep = get_class('ResonatorSweep')(streams)
    return sweep


def sweepstream_from_rnc(rnc, sweep_index, stream_index, channel, analyze=False):
    return get_class('SweepStream')(sweep=sweep_from_rnc(rnc, sweep_index, channel),
                                    stream=stream_from_rnc(rnc, stream_index, channel),
                                    analyze=analyze)


def snm_from_rnc(sweep_filename, sweep_group_index=0, timestream_filename=None, timestream_group_index=0,
                 resonator_index=0, low_pass_cutoff_Hz=4.0, dac_chain_gain = -49, delay_estimate=None,
                 deglitch_threshold=5, cryostat=None, mask_sweep_indicies=None, use_sweep_group_timestream=False,
                 conjugate_data=False):
    pass