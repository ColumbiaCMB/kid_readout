from __future__ import division
import numpy as np
from kid_readout.measurement.single import Stream, Sweep, ResonatorSweep, SweepStream
from kid_readout.measurement.multiple import StreamArray, SweepArray, ResonatorSweepArray, SweepStreamArray


# These functions extract state information from legacy data classes.


def sweep_state_from_rnc(rnc, sweep_group_index):
    """
    Return a dictionary containing the state information from the given ReadoutNetCDF object that is common to all
    channels for the given SweepGroup index in the file. This function does not return any per-channel arrays.

    :param rnc: a ReadoutNetCDF object.
    :param sweep_group_index: the index of the SweepGroup in the rnc.sweeps list.
    :return: a dictionary containing common state information.
    """
    state = {'gitinfo': rnc.gitinfo,
             'mmw_source': mmw_source_state_from_rnc(rnc),
             'roach': sweep_roach_state_from_rnc(rnc, sweep_group_index)}
    return state


def timestream_state_from_rnc(rnc, timestream_group_index):
    """
    Return a dictionary containing the state information from the given ReadoutNetCDF object that is common to all
    channels for the given TimestreamGroup index in the file. This function does not return any per-channel arrays.

    :param rnc: a ReadoutNetCDF object.
    :param timestream_group_index: the index of the TimestreamGroup in the rnc.timestreams list.
    :return: a dictionary containing common state information.
    """
    state = {'gitinfo': rnc.gitinfo,
             'mmw_source': mmw_source_state_from_rnc(rnc),
             'roach': timestream_roach_state_from_rnc(rnc, timestream_group_index)}
    return state


# TODO: change the format to single variables mickey and minnie, and keep NaN.
def mmw_source_state_from_rnc(rnc):
    """
    Return a dictionary containing millimeter-wave source information from the given ReadoutNetCDF object.

    :param rnc: a ReadoutNetCDF object.
    :return: a dictionary containing state information.
    """
    mickey_turns, minnie_turns = rnc.mmw_atten_turns
    state = {'mickey_turns': float(mickey_turns),
             'minnie_turns': float(minnie_turns)}
    return state


def common_roach_state_from_rnc(rnc):
    """
    Return a dictionary containing roach state information from the given ReadoutNetCDF that is common to all
    measurements in the file.

    :param rnc:
    :return:
    """
    state = {'boffile': rnc.boffile,
             'delay_estimate': rnc.get_delay_estimate(),
             'heterodyne': rnc.heterodyne}
    return state


def timestream_roach_state_from_rnc(rnc, timestream_group_index):
    tg = rnc.timestreams[timestream_group_index]
    if np.any(np.diff(tg.epoch)):
        raise ValueError("TimestreamGroup epoch values differ.")
    state = extract_timestream_group_roach_state(rnc, tg)
    return state


def sweep_roach_state_from_rnc(rnc, sweep_group_index):
    sg = rnc.sweeps[sweep_group_index]
    state = extract_timestream_group_roach_state(rnc, sg.timestream_group)
    return state


def extract_timestream_group_roach_state(rnc, timestream_group):
    start_epoch = float(timestream_group.epoch[0])
    hardware_state_index = int(rnc._get_hwstate_index_at(start_epoch))
    hardware_state_epoch = float(rnc.hardware_state_epoch[hardware_state_index])
    if start_epoch < hardware_state_epoch:
        raise ValueError("start_epoch < hardware_state_epoch")
    adc_attenuation = float(rnc.adc_atten[hardware_state_index])
    dac_attenuation, output_attenuation = [float(v) for v in rnc.get_effective_dac_atten_at(start_epoch)]
    num_tones = int(rnc.num_tones[hardware_state_index])
    if np.any(np.diff(timestream_group.tone_nsamp)):
        raise ValueError("TimestreamGroup tone_nsamp values differ.")
    num_tone_samples = int(timestream_group.tone_nsamp[0])
    modulation_rate, modulation_output = [int(v) for v in rnc.get_modulation_state_at(start_epoch)]
    state = {'hardware_state_index': hardware_state_index,
             'hardware_state_epoch': hardware_state_epoch,
             'adc_attenuation': adc_attenuation,
             'dac_attenuation': dac_attenuation,
             'output_attenuation': output_attenuation,
             'num_tones': num_tones,
             'num_tone_samples': num_tone_samples,
             'modulation_rate': modulation_rate,
             'modulation_output': modulation_output}
    return state


def timestream_arrays_from_rnc(rnc, timestream_group_index):
    """
    Return a dictionary containing relevant roach arrays:
    tone_bin is the roach output tone bin; all arrays are sorted so that this array is in ascending order.
    amplitude should be the tone amplitude; currently this is not saved so this function returns an array of NaN values.
    phase should be the tone phase; currently this is not saved so this function returns an array of NaN values.
    fft_bin should be the roach interface fft_bin; the quantity we have been saving is actually
    fpga_fft_readout_indexes + 1, and fftbin is commented out in ReadoutNetCDF, so this function returns an array of NaN
    values.

    :param rnc: a ReadoutNetCDF instance.
    :param timestream_group_index: the index of the desired timestream in rnc.timestreams.
    :return: a dictionary containing four relevant roach arrays.
    """
    tg = rnc.timestreams[timestream_group_index]
    order = tg.tonebin.argsort()
    arrays = {'tone_bin': tg.tonebin[order],
              'amplitude': np.nan * np.empty(tg.tonebin.size),
              'phase': np.nan * np.empty(tg.tonebin.size),
              'fft_bin': np.nan * np.empty(tg.tonebin.size)}
    return arrays


def nan_to_none(iterable):
    values = []
    for value in iterable:
        if np.isnan(value):
            values.append(None)
        else:
            values.append(float(value))
    return values


# These functions are intended to use the new code to read legacy data.
# TODO: change description functionality to add_legacy_origin


def stream_from_rnc(rnc, timestream_group_index, channel, description=None):
    state = timestream_state_from_rnc(rnc, timestream_group_index)
    if description is None:
        description = 'ReadoutNetCDF(\"{}\").timestreams[{}]'.format(rnc.filename, timestream_group_index)
    tg = rnc.timestreams[timestream_group_index]
    tg_channel_index = tg.measurement_freq.argsort()[channel]
    frequency = tg.measurement_freq[tg_channel_index]
    # All the epoch and data_len_seconds values are the same. Assume regular sampling.
    epoch = np.linspace(tg.epoch[tg_channel_index],
                        tg.epoch[tg_channel_index] + tg.data_len_seconds[tg_channel_index],
                        tg.num_data_samples)
    s21 = tg.data[tg_channel_index, :]
    return Stream(frequency, epoch, s21, state, description=description)


def streamarray_from_rnc(rnc, timestream_group_index, description=None):
    state = timestream_state_from_rnc(rnc, timestream_group_index)
    if description is None:
        description = 'ReadoutNetCDF(\"{}\").timestreams[{}]'.format(rnc.filename, timestream_group_index)
    tg = rnc.timestreams[timestream_group_index]
    # A TimestreamGroup not part of a SweepGroup has arrays in roach FPGA order.
    tg_channel_order = tg.measurement_freq.argsort()
    frequency = tg.measurement_freq[tg_channel_order]
    # All the epoch and data_len_seconds values are the same. Assume regular sampling.
    epoch = np.linspace(tg.epoch[0],
                        tg.epoch[0] + tg.data_len_seconds[0],
                        tg.num_data_samples)
    s21 = tg.data[tg_channel_order, :]
    return StreamArray(frequency, epoch, s21, state=state, description=description)


def sweep_from_rnc(rnc, sweep_group_index, channel, resonator=True, description=None):
    state = sweep_state_from_rnc(rnc, sweep_group_index)
    if description is None:
        description = 'ReadoutNetCDF(\"{}\").sweeps[{}]'.format(rnc.filename, sweep_group_index)
    sg = rnc.sweeps[sweep_group_index]
    tg = sg.timestream_group
    n_channels = np.unique(tg.sweep_index).size
    if tg.measurement_freq.size % n_channels:
        raise ValueError("Bad number of frequency points.")
    frequencies_per_index = int(tg.measurement_freq.size / n_channels)
    streams = []
    # Extract simultaneously-sampled data
    for n in range(frequencies_per_index):
        frequency = tg.measurement_freq[n::frequencies_per_index][channel]
        # All of the epochs are the same
        epoch = np.linspace(tg.epoch[n], tg.epoch[n] + tg.data_len_seconds[n], tg.num_data_samples)
        s21 = tg.data[n::frequencies_per_index][channel, :]
        streams.append(Stream(frequency, epoch, s21, {}))
    if resonator:
        return ResonatorSweep(streams, state=state, description=description)
    else:
        return Sweep(streams, state=state, description=description)


def sweeparray_from_rnc(rnc, sweep_group_index, resonator=True, description=None):
    state = sweep_state_from_rnc(rnc, sweep_group_index)
    if description is None:
        description = 'ReadoutNetCDF(\"{}\").sweeps[{}]'.format(rnc.filename, sweep_group_index)
    sg = rnc.sweeps[sweep_group_index]
    tg = sg.timestream_group
    n_channels = np.unique(tg.sweep_index).size
    if tg.measurement_freq.size % n_channels:
        raise ValueError("Bad number of frequency points.")
    frequencies_per_index = int(tg.measurement_freq.size / n_channels)
    stream_arrays = []
    # Extract simultaneously-sampled data
    for n in range(frequencies_per_index):
        frequency = tg.measurement_freq[n::frequencies_per_index]
        # All of the epochs are the same
        epoch = np.linspace(tg.epoch[n], tg.epoch[n] + tg.data_len_seconds[n], tg.num_data_samples)
        s21 = tg.data[n::frequencies_per_index]
        stream_arrays.append(StreamArray(frequency, epoch, s21, {}))
    if resonator:
        return ResonatorSweepArray(stream_arrays, state, description=description)
    else:
        return SweepArray(stream_arrays, state, description=description)


def sweepstream_from_rnc(rnc, sweep_group_index, timestream_group_index, channel):
    state = common_roach_state_from_rnc(rnc)
    sweep = sweep_from_rnc(rnc, sweep_group_index, channel)
    stream = stream_from_rnc(rnc, timestream_group_index, channel)
    return SweepStream(sweep, stream, state)


def sweepstreamarray_from_rnc(rnc, sweep_group_index, timestream_group_index):
    state = common_roach_state_from_rnc(rnc)
    sweep_array = sweeparray_from_rnc(rnc, sweep_group_index)
    stream_array = streamarray_from_rnc(rnc, timestream_group_index)
    return SweepStreamArray(sweep_array, stream_array, state)
