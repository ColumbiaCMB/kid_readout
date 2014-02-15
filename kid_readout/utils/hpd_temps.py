"""
Routines for getting temperature data for HPD cryostat from the 'adc' machine

Currently assumes the /home/adclocal/data/cooldown_logs directory on 'adc' is mounted to /home/data/adc_mount
"""

import glob
import os
import time
import bisect
import numpy as np
import netCDF4

nc_dir = "/home/data/adc_mount"

_filecache = {}

def get_nc_list():
    ncs = glob.glob(os.path.join(nc_dir,'2014*.nc'))
    ncs.sort()
    if len(ncs) == 0:
        raise Exception("Could not find any nc files; is the data directory mounted properly?")
    epochs = []
    for nc in ncs:
        base,fname = os.path.split(nc)
        epochs.append(time.mktime(time.strptime(fname,'%Y%m%d_%H%M%S.nc')))
        
    epochs = np.array(epochs) - 5*3600 # this will certainly break with DST...
    return epochs,ncs

epochs,ncs = get_nc_list()

def get_temperature_at(t):
    global epochs
    global ncs
    global _filecache
    
    idx = bisect.bisect_right(epochs,t)
    idx = idx - 1
    if idx < 0:
        idx = 0
    ncname = ncs[idx]
    if _filecache.has_key(ncname):# and (time.time()-t) > 1*3600: #if we're looking for data from moret han 30 hours ago, look in the cache
        times,temps = _filecache[ncname]
    else:
        times,temps = get_temperature_from_nc(ncname)
        _filecache[ncname] = (times,temps)
    return np.interp(t,times,temps)

def mjd_to_unix(mjd):
    return (mjd - 56701.933890492095)*86400 + 1392330288.13867  # super stupid, but should work

def get_temperature_from_nc(ncname):
    tic = time.time()
    done = False
    while (not done) and (time.time() - tic < 10):
        try:
            nc = netCDF4.Dataset(ncname)
            mjd = nc.variables['mjd_slow'][:]
            fridge = nc.groups['fridge']
            temps = fridge.variables['bridge_temp_value'][:]
            done = True
        except:
            print "retrying..."
            nc.close()
            time.sleep(0.1)
    unix = mjd_to_unix(mjd)
    return unix,temps