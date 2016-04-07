from __future__ import division
import numpy as np
from kid_readout.measurement.basic import (SingleStream, SingleSweep, SingleResonatorSweep, SingleSweepStream,
                                           StreamArray, SweepArray, ResonatorSweepArray, SweepStreamArray)

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
             'heterodyne': rnc.heterodyne}
    return state


def timestream_roach_state_from_rnc(rnc, timestream_group_index):
    tg = rnc.timestreams[timestream_group_index]
    if np.any(np.diff(tg.epoch)):
        raise ValueError("TimestreamGroup epoch values differ.")
    roach_state = global_roach_state_from_rnc(rnc)
    roach_state.update(roach_state_from_rnc_at_epoch(rnc, tg.epoch.min()))
    roach_state.update(roach_state_from_tg(tg))
    return roach_state


def sweep_roach_state_from_rnc(rnc, sweep_group_index):
    sg = rnc.sweeps[sweep_group_index]
    roach_state = global_roach_state_from_rnc(rnc)
    roach_state.update(roach_state_from_rnc_at_epoch(rnc, sg.start_epoch))
    roach_state.update(roach_state_from_tg(sg.timestream_group))
    return roach_state


def roach_state_from_rnc_at_epoch(rnc, epoch):
    hardware_state_index = int(rnc._get_hwstate_index_at(epoch))
    hardware_state_epoch = float(rnc.hardware_state_epoch[hardware_state_index])
    if epoch < hardware_state_epoch:
        raise ValueError("epoch < hardware_state_epoch")
    adc_attenuation = float(rnc.adc_atten[hardware_state_index])
    dac_attenuation = float(rnc.dac_atten[hardware_state_index])
    num_tones = int(rnc.num_tones[hardware_state_index])
    modulation_rate, modulation_output = [int(v) for v in rnc.get_modulation_state_at(epoch)]
    state = {'hardware_state_index': hardware_state_index,
             'hardware_state_epoch': hardware_state_epoch,
             'adc_attenuation': adc_attenuation,
             'dac_attenuation': dac_attenuation,
             'num_tones': num_tones,
             'modulation_rate': modulation_rate,
             'modulation_output': modulation_output}
    return state


def roach_state_from_tg(tg):
    """
    Return a dictionary containing state information from the given TimestreamGroup tg.

    This function returns values that do not vary per-channel and are not recorded in the rnc by epoch. All frequency
    and rate values are converted to Hz if they are in other units.

    :param tg: the TimestreamGroup from which to extract roach state.
    :return: a dict containing state information.
    """
    state = {'adc_sample_rate': 1e6 * float(common(tg.adc_sampling_freq)),
             'dac_sample_rate': 1e6 * float(common(tg.adc_sampling_freq)),
             'num_filterbank_channels': int(common(tg.nfft)),
             'num_tone_samples': int(common(tg.tone_nsamp)),
             'stream_sample_rate': float(common(tg.sample_rate)),
             'waveform_normalization': float(common(tg.wavenorm)),
             'bank': None,  # I don't think this is available in the data.
             # These were removed because they are derived:
             #'modulation': {'duty_cycle': float(common(tg.modulation_duty_cycle)),
             #               'frequency': float(common(tg.modulation_freq)),  # tg.mmw_source_modulation_freq deprecated
             #               'num_samples': int(common(tg.modulation_period_samples)),
             #               'tone_phase': float(common(tg.modulation_phase))}}
             }
    try:
        state['lo_frequency'] = 1e6 * float(common(tg.lo))
    except AttributeError:
        pass
    return state

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
        ticks_per_turn = 25
        state['mickey_ticks'] = ticks_per_turn * float(turns[0])
        state['minnie_ticks'] = ticks_per_turn * float(turns[1])
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


def stream_from_rnc(rnc, timestream_group_index, tone_index, description=None):
    roach_state = timestream_roach_state_from_rnc(rnc, timestream_group_index)
    state = timestream_state_from_rnc(rnc, timestream_group_index)
    if description is None:
        description = 'ReadoutNetCDF(\"{}\").timestreams[{}]'.format(rnc.filename, timestream_group_index)
    tg = rnc.timestreams[timestream_group_index]
    # A TimestreamGroup has arrays in roach FPGA order.
    increasing_order = tg.tonebin.argsort()
    tone_bin = tg.tonebin[increasing_order]
    amplitude = np.ones(tone_bin.size, dtype=np.float)
    phase = np.zeros(tone_bin.size, dtype=np.float)
    # TODO: decide what to do about these values: invert to RoachInterface values or ignore them?
    fpga_fft_bin_plus_one = int(tg.fftbin[increasing_order][tone_index])
    # All the epoch and data_len_seconds values are the same. Assume regular sampling.
    epoch = int(common(tg.epoch))
    s21_raw = tg.data[increasing_order, :][tone_index]
    data_demodulated = True  # Modify this if possible to determine from the rnc.
    return SingleStream(tone_bin=tone_bin, tone_amplitude=amplitude, tone_phase=phase, tone_index=tone_index,
                        filterbank_bin=fpga_fft_bin_plus_one, epoch=epoch, s21_raw=s21_raw, data_demodulated=data_demodulated,
                        roach_state=roach_state, state=state, description=description)


def streamarray_from_rnc(rnc, timestream_group_index, description=None):
    roach_state = timestream_roach_state_from_rnc(rnc, timestream_group_index)
    state = timestream_state_from_rnc(rnc, timestream_group_index)
    if description is None:
        description = 'ReadoutNetCDF(\"{}\").timestreams[{}]'.format(rnc.filename, timestream_group_index)
    tg = rnc.timestreams[timestream_group_index]
    # A TimestreamGroup has arrays in roach FPGA order.
    increasing_order = tg.tonebin.argsort()
    tone_bin = tg.tonebin[increasing_order]
    amplitude = np.ones(tone_bin.size, dtype=np.float)
    phase = np.zeros(tone_bin.size, dtype=np.float)
    tone_index = np.arange(tone_bin.size)
    # TODO: decide what to do about these values: invert to RoachInterface values or ignore them?
    fpga_fft_bin_plus_one = tg.fftbin[increasing_order].astype(np.int)
    # All the epoch and data_len_seconds values are the same. Assume regular sampling.
    epoch = int(common(tg.epoch))
    s21_raw = tg.data[increasing_order, :]
    data_demodulated = True  # Modify this if possible to determine from the rnc.
    return StreamArray(tone_bin=tone_bin, tone_amplitude=amplitude, tone_phase=phase, tone_index=tone_index,
                       filterbank_bin=fpga_fft_bin_plus_one, epoch=epoch, s21_raw=s21_raw,
                       data_demodulated=data_demodulated, roach_state=roach_state, state=state, description=description)


def sweep_from_rnc(rnc, sweep_group_index, tone_index, resonator=True, description=None):
    roach_state = sweep_roach_state_from_rnc(rnc, sweep_group_index)
    state = sweep_state_from_rnc(rnc, sweep_group_index)
    if description is None:
        description = 'ReadoutNetCDF(\"{}\").sweeps[{}]'.format(rnc.filename, sweep_group_index)
    sg = rnc.sweeps[sweep_group_index]
    tg = sg.timestream_group
    sweep_indices = np.unique(tg.sweep_index)
    start_epochs = np.unique(tg.epoch)
    if not tg.data.shape == (sweep_indices.size * start_epochs.size, tg.num_data_samples):
        raise ValueError("Data shape problem.")
    streams = []
    # Extract simultaneously-sampled data
    for start_epoch in start_epochs:
        simultaneous = tg.epoch == start_epoch
        unordered_tone_bin = tg.tonebin[simultaneous]
        increasing_order = unordered_tone_bin.argsort()
        tone_bin = unordered_tone_bin[increasing_order]  # Assume monotonic frequencies at each epoch:
        amplitude = np.ones(tone_bin.size, dtype=np.float)
        phase = np.zeros(tone_bin.size, dtype=np.float)
        fpga_fft_bin_plus_one = int(tg.fftbin[simultaneous][increasing_order][tone_index])
        # All of the epochs are the same
        epoch = int(common(tg.epoch[simultaneous]))
        s21_raw = tg.data[simultaneous, :][increasing_order][tone_index]
        data_demodulated = True  # Modify this if possible to determine from the rnc.
        streams.append(SingleStream(tone_bin=tone_bin, tone_amplitude=amplitude, tone_phase=phase, tone_index=tone_index,
                                    filterbank_bin=fpga_fft_bin_plus_one, epoch=epoch, s21_raw=s21_raw,
                                    data_demodulated=data_demodulated, roach_state=roach_state))
    if resonator:
        return SingleResonatorSweep(streams=streams, state=state, description=description)
    else:
        return SingleSweep(streams=streams, state=state, description=description)


def sweeparray_from_rnc(rnc, sweep_group_index, resonator=True, description=None):
    roach_state = sweep_roach_state_from_rnc(rnc, sweep_group_index)
    state = sweep_state_from_rnc(rnc, sweep_group_index)
    if description is None:
        description = 'ReadoutNetCDF(\"{}\").sweeps[{}]'.format(rnc.filename, sweep_group_index)
    sg = rnc.sweeps[sweep_group_index]
    tg = sg.timestream_group
    sweep_indices = np.unique(tg.sweep_index)
    start_epochs = np.unique(tg.epoch)
    if not tg.data.shape == (sweep_indices.size * start_epochs.size, tg.num_data_samples):
        raise ValueError("Data shape problem.")
    stream_arrays = []
    # Extract simultaneously-sampled data
    for start_epoch in start_epochs:
        simultaneous = tg.epoch == start_epoch
        unordered_tone_bin = tg.tonebin[simultaneous]
        increasing_order = unordered_tone_bin.argsort()
        tone_bin = unordered_tone_bin[increasing_order]  # Assume the the frequencies remain monotonic at each epoch.
        amplitude = np.ones(tone_bin.size, dtype=np.float)
        phase = np.zeros(tone_bin.size, dtype=np.float)
        tone_index = np.arange(tone_bin.size)  # For these data, all the tones are read out.
        fpga_fft_bin_plus_one = tg.fftbin[simultaneous][increasing_order].astype(np.int)
        # All of the epochs are the same
        epoch = int(common(tg.epoch[simultaneous]))
        s21_raw = tg.data[simultaneous, :][increasing_order]
        data_demodulated = True  # Modify this if possible to determine from the rnc.
        stream_arrays.append(StreamArray(tone_bin=tone_bin, tone_amplitude=amplitude, tone_phase=phase,
                                         tone_index=tone_index, filterbank_bin=fpga_fft_bin_plus_one, epoch=epoch,
                                         s21_raw=s21_raw, data_demodulated=data_demodulated, roach_state=roach_state))
    if resonator:
        return ResonatorSweepArray(stream_arrays=stream_arrays, state=state, description=description)
    else:
        return SweepArray(stream_arrays=stream_arrays, state=state, description=description)


def sweepstream_from_rnc(rnc, sweep_group_index, timestream_group_index, tone_index):
    state = global_roach_state_from_rnc(rnc)
    sweep = sweep_from_rnc(rnc, sweep_group_index, tone_index)
    stream = stream_from_rnc(rnc, timestream_group_index, tone_index)
    return SingleSweepStream(sweep, stream, state)


def sweepstreamarray_from_rnc(rnc, sweep_group_index, timestream_group_index):
    state = global_roach_state_from_rnc(rnc)
    sweep_array = sweeparray_from_rnc(rnc, sweep_group_index)
    stream_array = streamarray_from_rnc(rnc, timestream_group_index)
    return SweepStreamArray(sweep_array, stream_array, state)
