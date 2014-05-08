import bisect
import socket
if socket.gethostname() == 'detectors':
    default_cryostat = 'HPD'
else:
    default_cryostat = 'StarCryo'

import starcryo_experiments
import hpd_experiments


def get_experiment_info_at(unix_time,cryostat=None):
    if cryostat is None:
        cryostat = default_cryostat
    if cryostat.lower() == 'hpd':
        _unix_time_index = hpd_experiments._unix_time_index
        by_unix_time_table = hpd_experiments.by_unix_time_table
    else:
        _unix_time_index = starcryo_experiments._unix_time_index
        by_unix_time_table = starcryo_experiments.by_unix_time_table
    index = bisect.bisect(_unix_time_index,unix_time)
    if index == len(_unix_time_index):
        index -= 1
    date_string,description,optical_load = by_unix_time_table[index]
    if optical_load == 'dark':
        is_dark = True
    else:
        is_dark = False
    return description,is_dark,optical_load