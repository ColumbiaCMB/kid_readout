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

try:
    rx102a_dat = np.loadtxt('/data/detectors/RX-102A.tbl')
except IOError:
    raise ImportError("Could not find /data/detectors/RX-102A.tbl so can't work with HPD temperatures")
order = rx102a_dat[:,1].argsort()
rx102a_dat = rx102a_dat[order,:]

def rx102a_curve(R):
    return np.interp(R,rx102a_dat[:,1],rx102a_dat[:,0])

nc_dir = "/data/adc"

_filecache = {}

def get_nc_list():
    # First the original logs
    ncs = glob.glob(os.path.join(nc_dir,'cooldown_logs/20140*.nc'))
    ncs.sort()
    if len(ncs) == 0:
        raise Exception("Could not find any nc files; is the data directory mounted properly?")
    epochs = []
    for nc in ncs:
        base,fname = os.path.split(nc)
        epochs.append(time.mktime(time.strptime(fname,'%Y%m%d_%H%M%S.nc')) - 5*3600) # this will certainly break with DST...
        
    # Then the interim logs
    ncs2 = glob.glob(os.path.join(nc_dir,'cooldown_logs/201*-*.nc'))
    ncs2.sort()
    if len(ncs2) == 0:
        raise Exception("Could not find any nc files; is the data directory mounted properly?")
    
    for nc in ncs2:
        base,fname = os.path.split(nc)
        epochs.append(time.mktime(time.strptime(fname,'%Y-%m-%d_%H-%M-%S.nc')))
    ncs = ncs + ncs2
    
        
    epochs = np.array(epochs) 
    return epochs,ncs

epochs,ncs = get_nc_list()

def get_temperatures_at(t,get_load_temp=False):
    global epochs
    global ncs
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
    ncname = ncs[idx]
    if _filecache.has_key(ncname) and (time.time()-end_time) > 1*3600: #if we're looking for data from more than 1 hours ago, look in the cache
        times,temps,load = _filecache[ncname]
    else:
        times,temps,load = get_temperature_from_nc(ncname)
        _filecache[ncname] = (times,temps,load)
        
    if end_time > times[-1]:
        print "Warning: requested times may span more than one log file, so results may not be as intended"
        print "log file is: %s, last requested time is %s" % (ncname, time.ctime(end_time))

    load_temperature = np.interp(t,times,load)
    package_temperature = np.interp(t,times,temps)
    return package_temperature,None,load_temperature,None

def get_temperature_at(t,get_load_temp=False):
    global epochs
    global ncs
    global _filecache
    
    idx = bisect.bisect_right(epochs,t)
    idx = idx - 1
    if idx < 0:
        idx = 0
    ncname = ncs[idx]
    if _filecache.has_key(ncname) and (time.time()-t) > 1*3600: #if we're looking for data from more than 1 hours ago, look in the cache
        times,temps,load = _filecache[ncname]
    else:
        times,temps,load = get_temperature_from_nc(ncname)
        _filecache[ncname] = (times,temps,load)
    if get_load_temp:
        return np.interp(t,times,load)
    else:
        return np.interp(t,times,temps)

def mjd_to_unix(mjd):
    return (mjd - 56701.933890492095)*86400 + 1392330288.13867  # super stupid, but should work

def get_temperature_from_nc(ncname):
    tic = time.time()
    done = False
    while (not done) and (time.time() - tic < 300):
        try:
            nc = netCDF4.Dataset(ncname)
            if ncname.find('20140') < 0:
                unix = nc.groups['sim900'].variables['time'][:]
                res = nc.groups['sim900'].variables['bridge_res_value'][:]
                temps = rx102a_curve(res)
                load = nc.groups['sim900'].variables['therm_temperature2'][:]
            else:
                mjd = nc.variables['mjd_slow'][:]
                fridge = nc.groups['fridge']
                #temps = fridge.variables['bridge_temp_value'][:]
                res = fridge.variables['bridge_res_value'][:]
                load = fridge.variables['therm_temperature2'][:]
                temps = rx102a_curve(res)
                unix = mjd_to_unix(mjd)

            done = True
            nc.close()
        except:
            print "retrying..."
            nc.close()
            time.sleep(0.1)
    if not done:
        raise IOError("Timed out while trying to read temperature from %s" % ncname)
    return unix,temps,load