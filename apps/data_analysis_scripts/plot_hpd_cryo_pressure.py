from kid_readout.utils.easync import EasyNetCDF4

import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import pandas as pd

data_dir = '/home/data/adc_mount/garbage_cooldown_logs'
def get_data():
    files = glob.glob(os.path.join(data_dir,'2*.nc'))
    files.sort()
    data = {}
    var_names = ['pressure_high',
                 'comp_on',
                 'pressure_low',
                 'time',
                 'temp_helium',
                 'avg_pressure_delta']
    for name in var_names:
        data[name] = []
    for fn in files:
        print "processing",fn
        try:
            nc = EasyNetCDF4(fn)
        except RuntimeError:
            continue
        this_data = {}
        try:
            for name in var_names:
                print "Getting:", name
                this_data[name] = nc.cryomech.variables[name][:]
        except:
            nc.close()
            continue
        nc.close()
        for name in var_names:
            data[name].append(this_data[name])

    for name in var_names:
        data[name] = np.concatenate(data[name])
    df = pd.DataFrame.from_dict(data)
    return df,data

def fix_data(df):
    var_names = ['pressure_high',
                 'comp_on',
                 'pressure_low',
                 'temp_helium',
                 'avg_pressure_delta']
    msk = df.time <= 1397080000
    if df.avg_pressure_delta[0] < 1000:
        print "looks like it's already been fixed"
        return df
    for name in var_names:
        df.ix[msk,name] = df.ix[msk,name]/10.0
    return df

