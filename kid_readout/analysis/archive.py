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

def build_simple_archive(pklnames):
    if not type(pklnames) is list:
        pklnames = glob.glob(pklnames)
    data = []
    for pklname in pklnames:
        nms = load_noise_pkl(pklname)
        for nm in nms:
            data.append(nm.to_dataframe())
    df = pd.concat(data,ignore_index=True)
    return df