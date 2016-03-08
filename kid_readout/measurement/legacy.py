from __future__ import division
import numpy as np
from kid_readout.measurement.single import Stream, Sweep, ResonatorSweep, SweepStream
from kid_readout.measurement.multiple import StreamArray, SweepArray, ResonatorSweepArray, SweepStreamArray

# High-level functions that extract state information for all the hardware.

def sweep_state_from_rnc(rnc, sweep_group_index):
    """
    Return a dictionary containing the state information from the given ReadoutNetCDF instance that is common to all
    channels for the given SweepGroup index in the file. This function does not return any per-channel arrays.

    :param rnc: a ReadoutNetCDF instance.
    :param sweep_group_index: the index of the SweepGroup in the rnc.sweeps list.
    :return: a dict containing common state information.
    """
    state = {'gitinfo': rnc.gitinfo}
    # The TimestreamGroup of a SweepGroup doesn't have any millimeter-wave source info or ZBD info.
    mmw_source_state = mmw_source_state_from_rnc(rnc)
    if mmw_source_state:
        state['mmw_source'] = mmw_source_state
    state['roach'] = sweep_roach_state_from_rnc(rnc, sweep_group_index)
    return state


def timestream_state_from_rnc(rnc, timestream_group_index):
    """
    Return a dictionary containing the state information from the given ReadoutNetCDF instance that is common to all
    channels for the given TimestreamGroup index in the file. This function does not return any per-channel arrays.

    :param rnc: a ReadoutNetCDF instance.
    :param timestream_group_index: the index of the TimestreamGroup in the rnc.timestreams list.
    :return: a dict containing common state information.
    """
    tg = rnc.timestreams[timestream_group_index]
    state = {'gitinfo': rnc.gitinfo}
    mmw_source_state = mmw_source_state_from_rnc(rnc)
    if mmw_source_state:
        state['mmw_source'] = mmw_source_state
        state['mmw_source'].update(mmw_source_state_from_tg(tg))
    lockin_state = lockin_state_from_tg(tg)
    if lockin_state:
        state['lockin'] = lockin_state
    state['roach'] = timestream_roach_state_from_rnc(rnc, timestream_group_index)
    return state

# Roach.

def global_roach_state_from_rnc(rnc):
    """
    Return a dictionary containing roach state information from the given ReadoutNetCDF instance that is common to all
    measurements in the file.

    :param rnc: a ReadoutNetCDF instance.
    :return: a dict containing state information.
    """
    state = {'boffile': rnc.boffile,
             'delay_estimate': rnc.get_delay_estimate(),
             'heterodyne': rnc.heterodyne}
    return state


def timestream_roach_state_from_rnc(rnc, timestream_group_index):
    tg = rnc.timestreams[timestream_group_index]
    if np.any(np.diff(tg.epoch)):
        raise ValueError("TimestreamGroup epoch values differ.")
    roach_state = roach_state_from_rnc_at_epoch(rnc, tg.epoch.min())
    tg_roach_state = roach_state_from_tg(tg)
    roach_state['modulation'].update(tg_roach_state.pop('modulation'))
    roach_state.update(tg_roach_state)
    return roach_state


def sweep_roach_state_from_rnc(rnc, sweep_group_index):
    sg = rnc.sweeps[sweep_group_index]
    roach_state = roach_state_from_rnc_at_epoch(rnc, sg.start_epoch)
    tg_roach_state = roach_state_from_tg(sg.timestream_group)
    roach_state['modulation'].update(tg_roach_state.pop('modulation'))
    roach_state.update(tg_roach_state)
    return roach_state


def roach_state_from_rnc_at_epoch(rnc, epoch):
    hardware_state_index = int(rnc._get_hwstate_index_at(epoch))
    hardware_state_epoch = float(rnc.hardware_state_epoch[hardware_state_index])
    if epoch < hardware_state_epoch:
        raise ValueError("epoch < hardware_state_epoch")
    adc_attenuation = float(rnc.adc_atten[hardware_state_index])
    dac_attenuation, output_attenuation = [float(v) for v in rnc.get_effective_dac_atten_at(epoch)]
    num_tones = int(rnc.num_tones[hardware_state_index])
    modulation_rate, modulation_output = [int(v) for v in rnc.get_modulation_state_at(epoch)]
    state = {'hardware_state_index': hardware_state_index,
             'hardware_state_epoch': hardware_state_epoch,
             'adc_attenuation': adc_attenuation,
             'dac_attenuation': dac_attenuation,
             'output_attenuation': output_attenuation,
             'num_tones': num_tones,
             'modulation': {'rate': modulation_rate,
                            'output': modulation_output}}
    return state


def roach_state_from_tg(tg):
    """
    Return a dictionary containing state information from the given TimestreamGroup tg.

    This function returns values that do not vary per-channel and are not recorded in the rnc by epoch.

    :param tg: the TimestreamGroup from which to extract roach state.
    :return: a dict containing state information.
    """
    state = {'adc_sample_rate_MHz': float(common(tg.adc_sampling_freq)),
#             'dac_sample_rate': float(common(tg.adc_sampling_freq)),  # TODO: verify this for legacy data.
             'nfft': int(common(tg.nfft)),
             'num_data_samples': int(tg.num_data_samples),
             'num_tone_samples': int(common(tg.tone_nsamp)),
             'audio_sample_rate': float(common(tg.sample_rate)),
             'wavenorm': float(common(tg.wavenorm)),
             'modulation': {'duty_cycle': float(common(tg.modulation_duty_cycle)),
                            'frequency': float(common(tg.modulation_freq)),  # tg.mmw_source_modulation_freq deprecated
                            'num_samples': int(common(tg.modulation_period_samples)),
                            'phase': float(common(tg.modulation_phase))}}
    return state


def timestream_arrays_from_rnc(rnc, timestream_group_index):
    """
    Return a dictionary containing per-channel roach arrays.

    tone_bin is the roach output tone bin; returned arrays are sorted so that this array is in ascending order.
    amplitude should be the tone amplitude; currently this is not saved so this function returns None.
    phase should be the tone phase; currently this is not saved so this function returns None.
    fft_bin should be the roach interface fft_bin but the quantity we saved in the legacy data is actually
    fpga_fft_readout_indexes + 1, so this function returns None.

    :param rnc: a ReadoutNetCDF instance.
    :param timestream_group_index: the index of the desired timestream in rnc.timestreams.
    :return: a dictionary containing four relevant roach arrays.
    """
    tg = rnc.timestreams[timestream_group_index]
    order = tg.tonebin.argsort()
    arrays = {'tone_bin': tg.tonebin[order],
              'amplitude': None,
              'phase': None,
              'fft_bin': None}  # tg.fftbin[order]}
    return arrays

# Millimeter-wave source.

# TODO: implement ticks instead of turns, but keep turns available?
# TODO: add modulation_source entry.
def mmw_source_state_from_rnc(rnc):
    """
    Return a dictionary containing millimeter-wave source information from the given ReadoutNetCDF instance.

    :param rnc: a ReadoutNetCDF instance.
    :return: a dict containing state information.
    """
    state = {}
    turns = rnc.mmw_atten_turns
    if not np.any(np.isnan(turns)):
        state['mickey_turns'] = float(turns[0])
        state['minnie_turns'] = float(turns[1])
    return state


def mmw_source_state_from_tg(tg):
    """
    Return a dictionary containing millimeter-wave source information from the given TimestreamGroup instance.

    The only value currently returned is the millimeter-wave source output frequency. This is given in Hz in continuous-
    wave mode, and is None in broadband mode.

    :param tg: a TimestreamGroup instance.
    :return: a dict containing state information.
    """
    state = {'output_frequency': None}
    f_mmw = float(common(tg.mmw_source_freq))
    if f_mmw:
        state['output_frequency'] = f_mmw
    return state

# Lock-in amplifier.

def lockin_state_from_tg(tg):
    state = {'rms_voltage': float(common(tg.zbd_voltage)),
             'zbd_power_db_arb': float(common(tg.zbd_power_dbm))}
    return state

# Helper functions

def common(sequence):
    """
    Return the common value if all elements of the sequence are equal, or raise a ValueError if not.

    :param sequence: a sequence containing nominally equal numbers.
    :return: the common value.
    """
    if len(sequence) == 1 or np.all(np.diff(sequence) == 0):
        return sequence[0]
    else:
        raise ValueError("Not all sequence elements are equal: {}".format(sequence))


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
    state = global_roach_state_from_rnc(rnc)
    sweep = sweep_from_rnc(rnc, sweep_group_index, channel)
    stream = stream_from_rnc(rnc, timestream_group_index, channel)
    return SweepStream(sweep, stream, state)


def sweepstreamarray_from_rnc(rnc, sweep_group_index, timestream_group_index):
    state = global_roach_state_from_rnc(rnc)
    sweep_array = sweeparray_from_rnc(rnc, sweep_group_index)
    stream_array = streamarray_from_rnc(rnc, timestream_group_index)
    return SweepStreamArray(sweep_array, stream_array, state)
