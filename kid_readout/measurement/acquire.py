"""
Basic framework for acquiring a roach measurement that includes both sweep(s) and stream(s).

Acquire
-Initialize equipment.
-Initialize roach: preload frequencies, if necessary.
-Create state dictionary containing state from all equipment, including temperatures, if possible.
-Run a coarse sweep, if necessary: create a SweepArray and extract resonance frequencies.
-Run fine sweeps to map out resonance frequencies carefully.
If desired, we can combine the data from coarse and fine sweeps into a single SweepArray.
All streams in these sweeps are created with the same roach state, which should not change during the sweeps.
The sweep(s) are created with the experiment state, which should also not change.

Acquire streams:
-Initialize equipment for stream(s).
-Initialize roach for stream(s).
-Create experiment state dictionary.
-Acquire a StreamArray.
-Repeat the stream acquisition as needed
-Instantiate the final measurement with all data, and save it to disk.
-Clean up equipment.

If instead we want to save data as it is collected, we can do that by writing a blank final measurement to disk, then
writing the sub-measurements as they are acquired.
"""
from __future__ import division
import os
import sys
import time
import inspect
import subprocess
import logging

import numpy as np

from kid_readout import settings
from kid_readout.measurement import core, basic
from kid_readout.measurement.io import nc, npy
from kid_readout.analysis.resources import experiments

logger = logging.getLogger(__name__)

# Frequency sweep

def load_baseband_sweep_tones(ri, tone_banks, num_tone_samples):
    return ri.set_tone_freqs(freqs=np.vstack(tone_banks), nsamp=num_tone_samples)


def load_heterodyne_sweep_tones(ri, tone_banks, num_tone_samples):
    return ri.set_tone_freqs(freqs=np.vstack(tone_banks), nsamp=num_tone_samples)


def run_sweep(ri, tone_banks, num_tone_samples, length_seconds=0, state=None, description='', verbose=False,
              wait_for_sync=True, **kwargs):
    """
    Return a SweepArray acquired using the given tone banks.

    :param ri: a RoachInterface subclass instance.
    :param tone_banks: an iterable of arrays of frequencies to use for the sweep.
    :param num_tone_samples: the number of samples in the playback buffer; must be a power of two.
    :param length_seconds: the duration of each data stream. 0 means the minimum unit of data that can be read out in the current configuration
    :param state: the non-roach state to pass to the SweepArray.
    :param description: a string containing a description of the measurement.
    :param kwargs: keyword arguments passed to ri.get_measurement()
    :return: a SweepArray instance.
    """
    stream_arrays = core.MeasurementList()
    if verbose:
        print("Measuring bank")
    for n, tone_bank in enumerate(tone_banks):
        if verbose:
            print n,
            sys.stdout.flush()
        ri.set_tone_freqs(tone_bank, nsamp=num_tone_samples)
        ri.select_fft_bins(np.arange(tone_bank.size))
        # we wait a bit here to let the roach2 sync catch up.  figuring this out still.
        if wait_for_sync:
            time.sleep(0.1)
        stream_arrays.append(ri.get_measurement(num_seconds=length_seconds, **kwargs))
    return basic.SweepArray(stream_arrays, state=state, description=description)


def run_loaded_sweep(ri, length_seconds=0, state=None, description='', tone_bank_indices=None, bin_indices=None,
                     verbose=False,
                     **kwargs):
    """
    Return a SweepArray acquired using previously-loaded tones.

    :param ri: a RoachInterface subclass instance.
    :param tone_bank_indices: the indices of the tone banks to use in the sweep; the default is to use all existing.
    :param length_seconds: the duration of each data stream. 0 means the minimum unit of data that can be read out in the current configuration
    :param state: the non-roach state to pass to the SweepArray.
    :param description: a string containing a description of the measurement.
    :param kwargs: keyword arguments passed to ri.get_measurement()
    :return: a SweepArray instance.
    """
    if tone_bank_indices is None:
        tone_bank_indices = np.arange(ri.tone_bins.shape[0])
    if bin_indices is None:
        bin_indices = range(ri.tone_bins.shape[1])
    stream_arrays = core.MeasurementList()
    if verbose:
        print "Measuring bank:",
    for tone_bank_index in tone_bank_indices:
        if verbose:
            print tone_bank_index,
            sys.stdout.flush()
        ri.select_bank(tone_bank_index)
        ri.select_fft_bins(bin_indices)
        ri._sync()
        stream_arrays.append(ri.get_measurement(num_seconds=length_seconds, **kwargs))
    return basic.SweepArray(stream_arrays, state=state, description=description)


def run_multipart_sweep(ri, length_seconds=0, state=None, description='', num_tones_read_at_once=32, verbose=False,
                        **kwargs):
    num_tones = ri.tone_bins.shape[1]
    num_steps = num_tones // num_tones_read_at_once
    if num_steps == 0:
        num_steps = 1
    indices_to_read = range(num_tones)
    parts = []
    for step in range(num_steps):
        if verbose:
            print("running sweep step {} of {}.".format(step,num_steps))
        parts.append(run_loaded_sweep(ri, length_seconds=length_seconds, state=state, description=description,
                                      bin_indices=indices_to_read[step::num_steps], **kwargs))
    stream_arrays = core.MeasurementList()
    for part in parts:
        stream_arrays.extend(list(part.stream_arrays))
    return basic.SweepArray(stream_arrays, state=state, description=description)


# Metadata

def script_code():
    """
    Return the source code of a module running as '__main__'. Acquisition scripts can use this to save their code.

    If attempting to load the source code raises an exception, return a string representation of the exception.

    Returns
    -------
    str
        The code, with lines separated by newline characters.
    """
    try:
        return inspect.getsource(sys.modules['__main__'])
    except Exception as e:
        return str(e)


def git_log():
    import kid_readout
    kid_readout_directory = os.path.dirname(os.path.abspath(kid_readout.__file__))
    try:
        return subprocess.check_output(("cd {}; git log -1".format(kid_readout_directory)), shell=True)
    except Exception as e:
        return str(e)


def all_metadata():
    meta = {'script_code': script_code(),
            'git_log': git_log(),
            'cryostat': settings.CRYOSTAT,
            'cooldown': settings.COOLDOWN}
    return meta


# IO object creation

def new_nc_file(suffix='', directory=settings.BASE_DATA_DIR, metadata=None):
    if suffix and not suffix.startswith('_'):
        suffix = '_' + suffix
    if metadata is None:
        metadata = all_metadata()
    root_path = os.path.join(directory, time.strftime('%Y-%m-%d_%H%M%S') + suffix + '.nc')
    logger.debug("Creating new NCFile with path %s" % root_path)
    return nc.NCFile(root_path, metadata=metadata)


def new_npy_directory(suffix='', directory=settings.BASE_DATA_DIR, metadata=None):
    if suffix and not suffix.startswith('_'):
        suffix = '_' + suffix
    if metadata is None:
        metadata = all_metadata()
    root_path = os.path.join(directory, time.strftime('%Y-%m-%d_%H%M%S') + suffix)
    return npy.NumpyDirectory(root_path, metadata=metadata)
