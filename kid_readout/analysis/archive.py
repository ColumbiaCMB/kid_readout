import numpy as np
import pandas as pd
import os
try:
    import pwd
except ImportError:
    print "couldn't import pwd, ignoring"
    pwd = None

from kid_readout.analysis.noise_measurement import load_noise_pkl
import glob
import time
import kid_readout.analysis.noise_fit

def build_simple_archives(pklglob,index_to_id=None):
    pklnames = glob.glob(pklglob)
    dfs = []
    for pklname in pklnames:
        dfs.append(build_simple_archive([pklname],index_to_id=index_to_id))
    return pd.concat(dfs,ignore_index=True)

def build_simple_archive(pklnames, index_to_id = None, archive_name=None):

    if not type(pklnames) is list:
        pklnames = glob.glob(pklnames)

    if archive_name is None:
        archive_name = os.path.splitext(os.path.basename(pklnames[0]))[0]
    archname = '/home/data/archive/%s.npy' % archive_name

    data = []
    for pklname in pklnames:
        pkl = load_noise_pkl(pklname)
        if type(pkl) is list:
            nms = pkl
        else:
            nms = []
            for k,v in pkl.items():
                nms.extend(v)
        for nm in nms:
            try:
                data.append(nm.to_dataframe())
            except AttributeError:
                print "skipping non noise measurement for now"
    df = pd.concat(data,ignore_index=True)
    if index_to_id is None:
        indexes = list(set(df.resonator_index))
        indexes.sort()
        index_to_id = indexes
    def set_resonator_id(x):
        x['resonator_id'] = index_to_id[x.resonator_index.iloc[0]]
        return x
    df = df.groupby(['resonator_index']).apply(set_resonator_id).reset_index(drop=True)

    save_archive(df,archname)
    return df

def add_noise_summary(df,device_band=(1,100),amplifier_band=(2e3,10e3), method=np.median):
    x = df
    device_noise = []
    amplifier_noise = []
    for k in range(len(x)):
        devmsk = (x.pca_freq.iloc[k] >= device_band[0]) & (x.pca_freq.iloc[k] <= device_band[1])
        ampmsk = (x.pca_freq.iloc[k] >= amplifier_band[0]) & (x.pca_freq.iloc[k] <= amplifier_band[1])
        device_noise.append(method(x.pca_eigvals.iloc[k][1,devmsk]))
        amplifier_noise.append(method(x.pca_eigvals.iloc[k][1,ampmsk]))
    x['device_noise'] = np.array(device_noise)
    x['amplifier_noise'] = np.array(amplifier_noise)
    return df

def add_noise_fits(df):
    def add_noise_fit_info(x):
        try:
            nf = kid_readout.analysis.noise_fit.fit_single_pole_noise(x['pca_freq'].iloc[0],
                                                                  x['pca_eigvals'].iloc[0][1,:],
                                                                  max_num_masked=8)
        except:
            return x
        x['noise_fit_fc'] = nf.fc
        x['noise_fit_fc_err'] = nf.result.params['fc'].stderr
        x['noise_fit_device_noise'] = nf.A
        x['noise_fit_device_noise_err'] = nf.result.params['A'].stderr
        x['noise_fit_amplifier_noise'] = nf.nw
        x['noise_fit_amplifier_noise_err'] = nf.result.params['nw'].stderr
        return x
    return df.groupby(df.index).apply(add_noise_fit_info)

def save_archive(df,archname):
    try:
        np.save(archname,df.to_records())
        if pwd is not None:
            os.chown(archname, os.getuid(), pwd.getpwnam('readout').pw_gid)
    except Exception,e:
        print "failed to pickle",e


def load_archive(fn):
    npa = np.load(fn)
    df = pd.DataFrame.from_records(npa)
    return df