from __future__ import division
import numpy as np
from kid_readout.measurement.single import Stream, Sweep, ResonatorSweep, SweepStream
from kid_readout.measurement.multiple import StreamArray, SweepArray, ResonatorSweepArray, SweepStreamArray


# These functions extract state information from legacy data classes.


def state_from_rnc(rnc):
    state = {'gitinfo': rnc.gitinfo,
             'mmw_source': mmw_source_state_from_rnc(rnc),
             'roach': roach_state_from_rnc(rnc),
             }


    return state


def mmw_source_state_from_rnc(rnc):
    state = {'attenuator_turns': rnc.mmw_atten_turns}
    return state


def roach_state_from_rnc(rnc):
    state = {'boffile': rnc.boffile,
             'delay_estimate': rnc.get_delay_estimate(),
             'heterodyne': rnc.heterodyne,
            }


def roach_state_from_timestream_group(rnc, timestream_group_index):
    tg = rnc.timestreams[timestream_group_index]
    if np.any(np.diff(tg.epoch)):
        raise ValueError("TimestreamGroup epoch values differ.")
    start_epoch = tg.epoch[0]
    hardware_state_index = rnc._get_hwstate_index_at(start_epoch)
    hardware_state_epoch =  rnc.hardware_state_epoch[hardware_state_index]
    if start_epoch < hardware_state_epoch:
        raise ValueError("start_epoch < hardware_state_epoch")
    adc_attenuation = rnc.adc_atten[hardware_state_index]
    dac_attenuation, output_attenuation = rnc.get_effective_dac_atten_at(start_epoch)
    number_of_tones = rnc.num_tones[hardware_state_index]
    if np.any(np.diff(tg.tone_nsamp)):
        raise ValueError("TimestreamGroup tone_nsamp values differ.")
    number_of_tone_samples = tg.tone_nsamp[0]
    modulation_rate, modulation_output = rnc.get_modulation_state_at(start_epoch)
    state = {'hardware_state_index': hardware_state_index,
             'hardware_state_epoch': hardware_state_epoch,
             'adc_attenuation': adc_attenuation,
             'dac_attenuation': dac_attenuation,
             'output_attenuation': output_attenuation,
             'number_of_tones': number_of_tones,
             'number_of_tone_samples': number_of_tone_samples,
             'modulation_rate': modulation_rate,
             'modulation_output': modulation_output}
    return state


def roach_state_from_sweep_group(rnc, sweep_group_index):
    sg = rnc.sweeps[sweep_group_index]
    start_epoch = sg.start_epoch
    hardware_state_index = rnc._get_hwstate_index_at(start_epoch)
    hardware_state_epoch =  rnc.hardware_state_epoch[hardware_state_index]
    if start_epoch < hardware_state_epoch:
        raise ValueError("start_epoch < hardware_state_epoch")
    adc_attenuation = rnc.adc_atten[hardware_state_index]
    dac_attenuation, output_attenuation = rnc.get_effective_dac_atten_at(start_epoch)
    number_of_tones = rnc.num_tones[hardware_state_index]
    if np.any(np.diff(sg.timestream_group.tone_nsamp)):
        raise ValueError("TimestreamGroup tone_nsamp values differ.")
    number_of_tone_samples = sg.timestream_group.tone_nsamp[0]
    modulation_rate, modulation_output = rnc.get_modulation_state_at(start_epoch)
    state = {'hardware_state_index': hardware_state_index,
             'hardware_state_epoch': hardware_state_epoch,
             'adc_attenuation': adc_attenuation,
             'dac_attenuation': dac_attenuation,
             'output_attenuation': output_attenuation,
             'number_of_tones': number_of_tones,
             'number_of_tone_samples': number_of_tone_samples,
             'modulation_rate': modulation_rate,
             'modulation_output': modulation_output}
    return state


def roach_arrays_from_timestream_group(tg):
    """
    Return a dictionary containing relevant roach arrays:
    tone_bin is the roach output tone bin; all arrays are sorted so that this array is in ascending order.
    amplitude should be the tone amplitude; currently this is not saved so this function returns an array of NaN values.
    phase should be the tone phase; currently this is not saved so this function returns an array of NaN values.
    fft_bin should be the roach interface fft_bin; the quantity we have been saving is actually
    fpga_fft_readout_indexes + 1, and fftbin is commented out in ReadoutNetCDF, so this function returns an array of NaN values.

    :param tg: a TimestreamGroup instance.
    :return: a dictionary containing four relevant roach arrays.
    """
    order = tg.tonebin.argsort()
    arrays = {'tone_bin': tg.tonebin[order],
              'amplitude': np.nan * np.empty(tg.tonebin.size),
              'phase': np.nan * np.empty(tg.tonebin.size),
              'fft_bin': np.nan * np.empty(tg.tonebin.size)}
    return arrays



# These functions are intended to use the new code to read legacy data.

def stream_from_rnc(rnc, timestream_group_index, channel):
    tg = rnc.timestreams[timestream_group_index]
    tg_channel_index = tg.measurement_freq.argsort()[channel]
    frequency = tg.measurement_freq[tg_channel_index]
    # All the epoch and data_len_seconds values are the same. Assume regular sampling.
    epoch = np.linspace(tg.epoch[tg_channel_index],
                        tg.epoch[tg_channel_index] + tg.data_len_seconds[tg_channel_index],
                        tg.num_data_samples)
    s21 = tg.data[tg_channel_index, :]
    state = {}
    return Stream(frequency, epoch, s21, state)


def streamarray_from_rnc(rnc, timestream_group_index):
    tg = rnc.timestreams[timestream_group_index]
    tg_channel_order = tg.measurement_freq.argsort()
    frequency = tg.measurement_freq[tg_channel_order]
    # All the epoch and data_len_seconds values are the same. Assume regular sampling.
    epoch = np.linspace(tg.epoch[0],
                        tg.epoch[0] + tg.data_len_seconds[0],
                        tg.num_data_samples)
    s21 = tg.data[tg_channel_order, :]
    state = {}
    return StreamArray(frequency, epoch, s21, state)


def sweep_from_rnc(rnc, sweep_group_index, channel, resonator=True):
    sg = rnc.sweeps[sweep_group_index]
    tg = sg.timestream_group
    n_channels = np.unique(tg.sweep_index).size
    if tg.measurement_freq.size % n_channels:
        raise ValueError("Bad number of frequency points.")
    frequencies_per_index = int(tg.measurement_freq.size / n_channels)
    streams= []
    # Extract simultaneously-sampled data
    for n in range(frequencies_per_index):
        frequency = tg.measurement_freq[n::frequencies_per_index][channel]
        # All of the epochs are the same
        epoch = np.linspace(tg.epoch[n], tg.epoch[n] + tg.data_len_seconds[n], tg.num_data_samples)
        s21 = tg.data[n::frequencies_per_index][channel, :]
        streams.append(Stream(frequency, epoch, s21))
    state = {}
    if resonator:
        return ResonatorSweep(streams, state=state)
    else:
        return Sweep(streams, state=state)


def sweeparray_from_rnc(rnc, sweep_group_index, resonator=True):
    sg = rnc.sweeps[sweep_group_index]
    tg = sg.timestream_group
    n_channels = np.unique(tg.sweep_index).size
    if tg.measurement_freq.size % n_channels:
        raise ValueError("Bad number of frequency points.")
    frequencies_per_index = int(tg.measurement_freq.size / n_channels)
    streamarrays = []
    # Extract simultaneously-sampled data
    for n in range(frequencies_per_index):
        frequency = tg.measurement_freq[n::frequencies_per_index]
        # All of the epochs are the same
        epoch = np.linspace(tg.epoch[n], tg.epoch[n] + tg.data_len_seconds[n], tg.num_data_samples)
        s21 = tg.data[n::frequencies_per_index]
        streamarrays.append(StreamArray(frequency, epoch, s21))
    state = {}
    if resonator:
        return ResonatorSweepArray(streamarrays, state=state)
    else:
        return SweepArray(streamarrays, state=state)


def sweepstream_from_rnc(rnc, sweep_group_index, timestream_group_index, channel, analyze=False):
    return SweepStream(sweep=sweep_from_rnc(rnc, sweep_group_index, channel),
                       stream=stream_from_rnc(rnc, timestream_group_index, channel),
                       analyze=analyze)


def sweepstreamarray_from_rnc(rnc, sweep_group_index, timestream_group_index):
    sweep_array = sweeparray_from_rnc(rnc, sweep_group_index)
    stream_array = streamarray_from_rnc(rnc, timestream_group_index)
    state = {}
    return SweepStreamArray(sweep_array, stream_array, state=state)


def streamarray_from_timestream_group(tg):
    tg_channel_order = tg.measurement_freq.argsort()
    frequency = tg.measurement_freq[tg_channel_order]
    # All the epoch and data_len_seconds values are the same. Assume regular sampling.
    epoch = np.linspace(tg.epoch[0],
                        tg.epoch[0] + tg.data_len_seconds[0],
                        tg.num_data_samples)
    s21 = tg.data[tg_channel_order, :]
    state = {}
    return StreamArray(frequency, epoch, s21, state)
