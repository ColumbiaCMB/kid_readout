import numpy as np
import pandas as pd
import os
try:
    import pwd
except ImportError:
    print "couldn't import pwd, ignoring"

from kid_readout.analysis.noise_measurement import load_noise_pkl
import glob
import time

def build_simple_archive(pklnames, index_to_id = None):
    if not type(pklnames) is list:
        pklnames = glob.glob(pklnames)
    data = []
    for pklname in pklnames:
        nms = load_noise_pkl(pklname)
        for nm in nms:
            data.append(nm.to_dataframe())
    df = pd.concat(data,ignore_index=True)
    if index_to_id is None:
        indexes = list(set(df.resonator_index))
        indexes.sort()
        index_to_id = indexes
    def set_resonator_id(x):
        x['resonator_id'] = index_to_id[x.resonator_index.iloc[0]]
        return x
    df = df.groupby(['resonator_index']).apply(set_resonator_id).reset_index(drop=True)
    return df