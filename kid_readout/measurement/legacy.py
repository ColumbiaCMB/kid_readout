from __future__ import division
import numpy as np
from kid_readout.measurement.single import Stream, Sweep, ResonatorSweep, SweepStream
from kid_readout.measurement.array import StreamArray, SweepArray, ResonatorSweepArray, SweepStreamArray


# These functions are intended to use the new code to read old data.

def stream_from_rnc(rnc, stream_index, channel):
    tg = rnc.timestreams[stream_index]
    tg_channel_index = tg.measurement_freq.argsort()[channel]
    frequency = tg.measurement_freq[tg_channel_index]
    # All the epoch and data_len_seconds values are the same. Assume regular sampling.
    epoch = np.linspace(tg.epoch[tg_channel_index],
                        tg.epoch[tg_channel_index] + tg.data_len_seconds[tg_channel_index],
                        tg.num_data_samples)
    s21 = tg.data[tg_channel_index, :]
    state = {}
    return Stream(frequency, epoch, s21, state)


def streamarray_from_rnc(rnc, stream_index):
    tg = rnc.timestreams[stream_index]
    tg_channel_order = tg.measurement_freq.argsort()
    frequency = tg.measurement_freq[tg_channel_order]
    # All the epoch and data_len_seconds values are the same. Assume regular sampling.
    epoch = np.linspace(tg.epoch[0],
                        tg.epoch[0] + tg.data_len_seconds[0],
                        tg.num_data_samples)
    s21 = tg.data[tg_channel_order, :]
    state = {}
    return StreamArray(frequency, epoch, s21, state)


def sweep_from_rnc(rnc, index_of_sweep, channel, resonator=True):
    sg = rnc.sweeps[index_of_sweep]
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
        streams.append(StreamArray(frequency, epoch, s21))
    state = {}
    if resonator:
        return ResonatorSweep(streams, state=state)
    else:
        return Sweep(streams, state=state)


def sweeparray_from_rnc(rnc, index_of_sweep, resonator=True):
    sg = rnc.sweeps[index_of_sweep]
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


def sweepstream_from_rnc(rnc, index_of_sweep, index_of_stream, channel, analyze=False):
    return SweepStream(sweep=sweep_from_rnc(rnc, index_of_sweep, channel),
                       stream=stream_from_rnc(rnc, index_of_stream, channel),
                       analyze=analyze)


def sweepstreamarray_from_rnc(rnc, index_of_sweep, index_of_stream):
    sweep_array = sweeparray_from_rnc(rnc, index_of_sweep)
    stream_array = streamarray_from_rnc(rnc, index_of_stream)
    state = {}
    return SweepStreamArray(sweep_array, stream_array, state=state)