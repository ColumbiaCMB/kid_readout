import numpy as np
import pandas as pd
import os
from kid_readout.analysis.noise_summary import load_noise_pkl
import glob

lg_5x4 = 54e-9*np.ones((20,))
cap_5x4 = np.array([27.57,
                    24.86,
                    22.58,
                    20.70,
                    18.83,
                    15.92,
                    14.67,
                    13.63,
                    12.59,
                    11.76,
                    10.40,
                    10.40,
                    10.40,
                    10.40,
                    10.40,
                    8.43,
                    8.01,
                    7.59,
                    7.17,
                    6.76])*1e-12

files = glob.glob('/home/data/noise_2014-02-1*.pkl')# + glob.glob('/home/data/noise_2014-02-14*.pkl')
files.sort()
sc3x3_0813f5_dark_info = dict(chip_name='StarCryo 3x3 0813f5 HPD Dark 1',
                   dark = True,
                   files = files,
                   index_to_resnum=range(8)
                              )

files = glob.glob('/home/data/noise_2014-02-2*.pkl')
files.sort()
sc3x3_0813f5_dark_info_2 = dict(chip_name='StarCryo 3x3 0813f5 HPD Dark 2',
                   dark = True,
                   files = files,
                   index_to_resnum=range(8)
                              )

files = glob.glob('/home/data/noise_2014-03-01_2*.pkl')
files.sort()
sc5x4_0813f10_net_info_1 = dict(chip_name='StarCryo 5x4 0813f10 LPF Horns NET 1',
                                dark = False,
                                files = files,
                                index_to_resnum=range(20))

            #index_to_resnum = [1,2,3,6,8,9,12,13,14,15,17,19])
def build_noise_archive(info, force_rebuild=False):
    chipfname = info['chip_name'].replace(' ','_')
    archname = '/home/data/archive/%s.npy' % chipfname
    df = None
    if not force_rebuild and os.path.exists(archname):
        try:
            df = load_archive(archname)
            return df
        except:
            pass
     
    nms = []
    for fname in info['files']:
        try:
            nms.extend(load_noise_pkl(fname))
        except Exception,e:
            print "couldn't get noise measurements from",fname,"error was:",e
    pnames = nms[0].params.keys()
    try:
        pnames.remove('a')
    except:
        pass
    data = {}
    for pn in pnames:
        data[pn] = [nm.params[pn].value for nm in nms]
        data[pn + '_err'] = [nm.params[pn].stderr for nm in nms]
    avals = []
    aerrs = []
    for nm in nms:
        if nm.params.has_key('a'):
            avals.append(nm.params['a'].value)
            aerrs.append(nm.params['a'].stderr)
        else:
            avals.append(np.nan)
            aerrs.append(np.nan)
    data['a'] = avals
    data['a_err'] = aerrs
    for pn in ['Q_i','swp_epoch','ts_epoch','power_dbm','atten','start_temp','end_temp',
               'pca_fr','pca_evals','pca_angles','fr','s21','chip','tsl_raw', 's0', 'ds0',
               's21m','frm']:
        data[pn] = [getattr(nm,pn) for nm in nms]
    pca_fr = data['pca_fr'][0]
    mask250 = (pca_fr > 150) & (pca_fr < 350)
    mask30k = (pca_fr > 20e3) & (pca_fr < 40e3)
    data['noise_250_hz'] = [nm.pca_evals[1,mask250].mean() for nm in nms]
    data['noise_30_khz'] = [nm.pca_evals[1,mask30k].mean() for nm in nms]
    data['dark'] = [info['dark'] for nm in nms]
    data['f_probe'] = [getattr(nm,'f0') for nm in nms]
    data['ridx'] = [info['index_to_resnum'][nm.index] for nm in nms]
    lgs = []
    cgs = []
    for nm in nms:
        try:
            lg = lg_5x4[info['index_to_resnum'][nm.index]]
            cg = cap_5x4[info['index_to_resnum'][nm.index]]
        except IndexError:
            lg = np.nan
            cg = np.nan
        lgs.append(lg)
        cgs.append(cg)
                
            
    data['Lg'] = lgs
    data['Cg'] = cgs
    data['chip_name'] = [info['chip_name'] for nm in nms]
    df = pd.DataFrame(data)
    df['round_temp'] = np.round(df['end_temp']*1000/10)*10
    np.save(archname,df.to_records())
    return df

def load_archive(fn):
    npa = np.load(fn)
    df = pd.DataFrame.from_records(npa)
    return df
    