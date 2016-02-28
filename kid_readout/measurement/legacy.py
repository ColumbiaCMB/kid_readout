from __future__ import division
import numpy as np
from kid_readout.measurement.single import Stream, ResonatorSweep, SweepStream
from kid_readout.measurement.array import SweepArray, StreamArray, SweepStreamArray


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


def sweep_from_rnc(rnc, sweep_index, channel):
    sg = rnc.sweeps[sweep_index]
    n_channels = np.unique(sg.index).size
    if not sg.frequency.size % n_channels == 0:
        raise ValueError("Bad number of frequency points.")
    frequencies_per_index = int(sg.frequency.size / n_channels)
    streams = []
    for i in range(frequencies_per_index * channel,
                   frequencies_per_index * (channel + 1)):
        frequency = sg.timestream_group.measurement_freq[i]
        epoch = np.linspace(sg.timestream_group.epoch[i],
                            sg.timestream_group.epoch[i] + sg.timestream_group.data_len_seconds[i],
                            sg.timestream_group.num_data_samples)
        s21 = sg.timestream_group.data[i, :]
        state = {}
        streams.append(Stream(frequency=frequency, epoch=epoch, s21=s21, state=state))
    return ResonatorSweep(streams)


def sweeparray_from_rnc(rnc, sweep_index):
    sg = rnc.sweeps[sweep_index]
    n_channels = np.unique(sg.index).size
    if not sg.frequency.size % n_channels == 0:
        raise ValueError("Bad number of frequency points.")
    frequencies_per_index = int(sg.frequency.size / n_channels)
    tg = sg.timestream_group
    streamarrays = []
    for n in range(frequencies_per_index):
        frequency = tg.measurement_freq[::]



def sweepstream_from_rnc(rnc, sweep_index, stream_index, channel, analyze=False):
    return SweepStream(sweep=sweep_from_rnc(rnc, sweep_index, channel),
                       stream=stream_from_rnc(rnc, stream_index, channel),
                       analyze=analyze)


