import time
import numpy as np
from kid_readout.measurement import multiple


def make_stream_array(ri, frequency, num_tone_samples, length, amplitude=None, phase=None, state=None, description=''):
    variables = get_stream_data(ri=ri, frequency=frequency, num_tone_samples=num_tone_samples, length=length,
                                amplitude=amplitude, phase=phase)
    variables['state'] = state
    variables['description'] = description
    return multiple.StreamArray(**variables)


def get_stream_data(ri, frequency, num_tone_samples, length, amplitude=None, phase=None):
    ri.set_tone_freqs(freqs=frequency, nsamp=num_tone_samples, amps=amplitude, phases=phase)
    variables = {'roach_state': ri.state}
    variables.update(ri.active_state_arrays)
    start_epoch = time.time()
    s21, sequence_numbers = ri.get_data_seconds(length)
    end_epoch = time.time()
    variables['s21'] = s21.T
    num_samples = variables['s21'].shape[1]
    # TODO: this should calculate based on the number of samples
    variables['epoch'] = np.linspace(start_epoch, end_epoch, num_samples)
    return variables

