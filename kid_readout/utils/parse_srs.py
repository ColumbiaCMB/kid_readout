import time
import numpy as np
import os
import glob

_timecache = []
_tempcache = []
_lastupdate = -1

def get_all_temperature_data(logdir = '/home/heather/SRS'):
    logs = glob.glob(os.path.join(logdir,'201*.txt'))
    logs.sort()
    times = []
    temps = []
    for log in logs:
        t0,temp0 = parse_srs_log(log)
        if len(t0) == 0:
            t0,temp0 = parse_srs_log(log,sensor=1)
        times = times + t0.tolist()
        temps = temps + temp0.tolist()
    return times,temps
        
def get_temperature_at(epoch):
    global _timecache, _tempcache, _lastupdate
    if time.time() - _lastupdate > 20*60:
        _timecache, _tempcache = get_all_temperature_data()
        _lastupdate = time.time()
        print "get_temperature_at: updated cache"
    if len(_tempcache) == 0:
        return 0
    temp = np.interp(epoch,_timecache,_tempcache)
    return temp

def parse_srs_log(fname,sensor=2):
    """
    Parse log file created by Heather's SRS logger
    *fname* : file name of log
    *sensor* : which sensor to extract (right now 2 corresponds to the package thermometer)
    returns numpy arrays of unix times and the temperature values
    """
    fh = open(fname,'r')
    lines = fh.readlines()
    fh.close()
    temps = []
    times = []
    for line in lines:
        parts = line.split(' ')
        try:
            if len(parts) >9:
                times.append(time.mktime(time.strptime(parts[0].strip(),'%Y%m%d-%H%M%S')))
                temps.append(float(parts[9]))
                continue
            if int(parts[0]) != sensor:
                continue
            if len(parts) < 4:
                continue
            temps.append(float(parts[2]))
            times.append(time.mktime(time.strptime(parts[3].strip(),'%Y%m%d-%H%M%S')))
        except ValueError:
            print "failed to parse",repr(line),"skipping"
    return np.array(times),np.array(temps)


class SRSLogFile(object):
    """
    Usage example: log_file[1].R returns an array of resistances for
    channel 1.
    """

    class Channel(object):

        def __init__(self, time, R, T):
            self.time = time
            self.R = R
            self.T = T

    def __init__(self, filename):
        self._channels = {}
        channels, resistances, temperatures, times = np.loadtxt(filename,
                                                                unpack=True,
                                                                converters={3: self.convert_timestamp})
        for channel in np.unique(channels.astype('int')):
            mask = (channels == channel)
            self._channels[channel] = self.Channel(times[mask],
                                                   resistances[mask],
                                                   temperatures[mask])

    def __getitem__(self, item):
        return self._channels[item]

    def convert_timestamp(self, timestamp):
        return time.mktime(time.strptime(timestamp.strip(),'%Y%m%d-%H%M%S'))
