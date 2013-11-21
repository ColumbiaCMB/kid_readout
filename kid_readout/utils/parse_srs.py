import time
import numpy as np
import os
import glob

def get_all_temperature_data(logdir = '/home/heather/SRS'):
    logs = glob.glob(os.path.join(logdir,'2013*.txt'))
    logs.sort()
    times = []
    temps = []
    for log in logs:
        t0,temp0 = parse_srs_log(log)
        times = times + t0.tolist()
        temps = temps + temp0.tolist()
    return times,temps
        

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
        try:
            parts = line.split(' ')
            if int(parts[0]) != sensor:
                continue
            if len(parts) < 4:
                continue
            temps.append(float(parts[2]))
            times.append(time.mktime(time.strptime(parts[3].strip(),'%Y%m%d-%H%M%S')))
        except ValueError:
            print "failed to parse",repr(line),"skipping"
    return np.array(times),np.array(temps)
