from __future__ import division
import numpy as np
from kid_readout.measure import measurement


def stream_from_rnc(rnc, stream_index, channel):
    tg = rnc.timestreams[stream_index]
    tg_channel_index = tg.measurement_freq.argsort()[channel]
    stream = measurement.Stream(tg.measurement_freq[tg_channel_index],
                                tg.get_data_index(tg_channel_index),
                                tg.epoch[tg_channel_index],
                                tg.epoch[tg_channel_index] + tg.data_len_seconds[tg_channel_index])
    return stream


def sweep_from_rnc(rnc, sweep_index, channel):
    sg = rnc.sweeps[sweep_index]
    n_channels = np.unique(sg.index).size
    if not sg.frequency.size % n_channels == 0:
        raise ValueError("Bad number of frequency points.")
    frequencies_per_index = int(sg.frequency.size / n_channels)
    streams = []
    for i in range(frequencies_per_index * channel,
                   frequencies_per_index * (channel + 1)):
        streams.append(measurement.Stream(sg.timestream_group.measurement_freq[i],
                                          sg.timestream_group.data[i,:],
                                          sg.timestream_group.epoch[channel],
                                          sg.timestream_group.epoch[channel] +
                                          sg.timestream_group.data_len_seconds[channel]))
    sweep = measurement.ResonatorSweep(streams)
    return sweep


def sweepstream_from_rnc(rnc, sweep_index, stream_index, channel, analyze=False):
    return measurement.SweepStream(sweep=sweep_from_rnc(rnc, sweep_index, channel),
                                   stream=stream_from_rnc(rnc, stream_index, channel),
                                   analyze=analyze)


