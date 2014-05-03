"""
Routines for getting temperature data for starcryo cryostat
Currently assumes SRS logs are in ~heather/SRS
"""

import glob
import os
import time
import bisect
import numpy as np
import netCDF4
import kid_readout.utils.parse_srs



temperature_log_file_dir = '/home/heather/SRS/'

_filecache = {}
epochs = []
temperature_log_filenames = []
temperature_log_filenames_last_updated = 0

def get_temperature_log_file_list():
    # First the original logs
    log_filenames = glob.glob(os.path.join(temperature_log_file_dir,'2*.txt'))
    log_filenames.sort()
    if len(log_filenames) == 0:
        raise Exception("Could not find any temperature log files in %s" % temperature_log_file_dir)
    epochs = []
    for filename in log_filenames:
        base,fname = os.path.split(filename)
        epochs.append(time.mktime(time.strptime(fname,'%Y%m%d-%H%M%S.txt')))    
        
    epochs = np.array(epochs) 
    return epochs,log_filenames

def refresh_temperature_log_file_list():
    global epochs
    global temperature_log_filenames
    global temperature_log_filenames_last_updated
    if not temperature_log_filenames or time.time() - temperature_log_filenames_last_updated > 20*3600:
        epochs,temperature_log_filenames = get_temperature_log_file_list()
        temperature_log_filenames_last_updated = time.time()
        print "updated file list"
    
refresh_temperature_log_file_list()

def get_temperatures_at(t):
    refresh_temperature_log_file_list()
    global epochs
    global temperature_log_filenames
    global _filecache
    
    if np.isscalar(t):
        start_time = t
        end_time = t
    else:
        start_time = t.min()
        end_time = t.max()
    idx = bisect.bisect_right(epochs,start_time)
    idx = idx - 1
    if idx < 0:
        idx = 0
    filename = temperature_log_filenames[idx]
    
    if _filecache.has_key(filename) and (time.time()-end_time) > 20*60: #if we're looking for data from more than 20 minutes ago, look in the cache
        datetime_timestamp,data = _filecache[filename]
    else:
        datetime_timestamp,data = kid_readout.utils.parse_srs.get_load_log(filename)
        _filecache[filename] = (datetime_timestamp,data)
        
    times = data[:,0]
    if end_time > times[-1]:
        print "Warning: requested times may span more than one log file, so results may not be as intended"
        print "log file is: %s, last requested time is %s" % (filename, time.ctime(end_time))
    primary_package_temperature = np.interp(t,times,data[:,-4])
    secondary_package_temperature = np.interp(t,times,data[:,-2])
    primary_load_temperature = np.interp(t,times,data[:,1])
    secondary_load_temperature = np.interp(t,times,data[:,2])
    return primary_package_temperature, secondary_package_temperature, primary_load_temperature, secondary_load_temperature
    
