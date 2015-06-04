"""
Routines for getting temperature data for starcryo cryostat
Currently assumes SRS logs are in /home/data/SRS
"""

import glob
import os
import time
import bisect
import numpy as np
import netCDF4
import kid_readout.utils.parse_srs

import kid_readout.analysis.resources.experiments


temperature_log_file_dir = '/data/readout/SRS/'

_filecache = {}
epochs = []
temperature_log_filenames = []
temperature_log_filenames_last_updated = 0


def get_temperature_log_file_list():
    # First the original logs
    log_filenames = glob.glob(os.path.join(temperature_log_file_dir, '2*.txt'))
    log_filenames.sort()
    if len(log_filenames) == 0:
        print("Could not find any temperature log files in %s" % temperature_log_file_dir)
    epochs = []
    for filename in log_filenames:
        base, fname = os.path.split(filename)
        epochs.append(time.mktime(time.strptime(fname, '%Y%m%d-%H%M%S.txt')))

    epochs = np.array(epochs)
    return epochs, log_filenames


def refresh_temperature_log_file_list():
    global epochs
    global temperature_log_filenames
    global temperature_log_filenames_last_updated
    if not temperature_log_filenames or time.time() - temperature_log_filenames_last_updated > 20 * 3600:
        epochs, temperature_log_filenames = get_temperature_log_file_list()
        temperature_log_filenames_last_updated = time.time()
        print "updated file list"


refresh_temperature_log_file_list()


def get_temperatures_at(t):
    refresh_temperature_log_file_list()
    global epochs
    global temperature_log_filenames
    global _filecache

    if len(temperature_log_filenames) == 0:
        if np.isscalar(t):
            results = np.nan
        else:
            results = np.nan * np.ones(t.shape)
        return results, results, results, results

    if np.isscalar(t):
        start_time = t
        end_time = t
    else:
        start_time = t.min()
        end_time = t.max()
    idx = bisect.bisect_right(epochs, start_time)
    idx = idx - 1
    if idx < 0:
        idx = 0
    filename = temperature_log_filenames[idx]

    if _filecache.has_key(filename) and (
        time.time() - end_time) > 20 * 60:  # if we're looking for data from more than 20 minutes ago, look in the cache
        datetime_timestamp, data = _filecache[filename]
    else:
        datetime_timestamp, data = kid_readout.utils.parse_srs.get_load_log(filename)
        _filecache[filename] = (datetime_timestamp, data)

    info = kid_readout.analysis.resources.experiments.get_experiment_info_at(start_time, cryostat='StarCryo')
    thermometry = info['thermometry_config']

    times = data[:, 0]
    if end_time > times[-1]:
        print "Warning: requested times may span more than one log file, so results may not be as intended"
        print "log file is: %s, last requested time is %s" % (filename, time.ctime(end_time))
    primary_package_temperature = np.interp(t, times, data[:, thermometry['package']])
    if np.isscalar(t):
        default = np.nan
    else:
        default = np.nan * np.ones_like(t)
    secondary_package_temperature = default
    primary_load_temperature = default
    secondary_load_temperature = default

    for identifier in ['stage', 'secondary_package']:
        if identifier in thermometry:
            secondary_package_temperature = np.interp(t, times, data[:, thermometry[identifier]])
            break
    if 'load' in thermometry:
        primary_load_temperature = np.interp(t, times, data[:, thermometry['load']])

    for identifier in ['copper', 'secondary_load']:
        if identifier in thermometry:
            secondary_load_temperature = np.interp(t, times, data[:, thermometry[identifier]])
            break

    return primary_package_temperature, secondary_package_temperature, primary_load_temperature, secondary_load_temperature
