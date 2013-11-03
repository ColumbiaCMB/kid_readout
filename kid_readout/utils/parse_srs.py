import time
import numpy as np

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
        if int(parts[0]) != sensor:
            continue
        if len(parts) < 4:
            continue
        temps.append(float(parts[2]))
        times.append(time.mktime(time.strptime(parts[3].strip(),'%Y%m%d-%H%M%S')))
    return np.array(times),np.array(temps)
