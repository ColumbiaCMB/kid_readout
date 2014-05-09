import numpy as np
import pandas as pd
import os
from kid_readout.analysis.noise_measurement import load_noise_pkl
import glob
import socket

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

files = glob.glob('/home/data/noise_2014-04-15*.pkl')
files += glob.glob('/home/data/noise_2014-04-16*.pkl')
files += glob.glob('/home/data/noise_2014-04-17*.pkl')
files = [x for x in files if x.find('led') < 0]
files.sort()
sc3x3_0813f5_dark_info_3 = dict(chip_name='StarCryo 3x3 0813f5 HPD Dark 3',
                   dark = True,
                   files = files,
                   index_to_resnum=range(8)
                              )

files += glob.glob('/home/data/noise_2014-04-17*.pkl')
files += glob.glob('/home/data/noise_2014-04-18*.pkl')
files = [x for x in files if x.find('led') < 0]
files.sort()
sc3x3_0813f5_dark_info_4 = dict(chip_name='StarCryo 3x3 0813f5 HPD Dark 4',
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

files = glob.glob('/home/data/noise_2014-04-06*') + glob.glob('/home/data/noise_2014-04-07*')
files = [x for x in files if x.find('_net') < 0]
files.sort()
sc5x4_0813f12_dark_info = dict(chip_name = 'StarCryo_5x4_0813f12',
                               dark = True,
                               files = files,
                               index_to_resnum = range(20)
                               )

files = glob.glob('/home/data/noise_2014-04-18*') + glob.glob('/home/data/noise_2014-04-21*')
files += glob.glob('/home/data/noise_2014-04-22*')
files = [x for x in files if x.find('_net') < 0]
files = [x for x in files if x.find('compressor') < 0]
files.sort()
sc5x4_0813f12_taped_dark_info = dict(chip_name = 'StarCryo_5x4_0813f12_taped_dark',
                               dark = True,
                               files = files,
                               index_to_resnum = range(20)
                               )

            #index_to_resnum = [1,2,3,6,8,9,12,13,14,15,17,19]) 
def build_noise_archive(info, force_rebuild=False):
    nm = load_noise_pkl(info['files'][0])[0]
    chipfname = nm.chip_name.replace(' ','_').replace(',','')
    archname = '/home/data/archive/%s.npy' % chipfname
    df = None
    if not force_rebuild and os.path.exists(archname):
        try:
            df = load_archive(archname)
            print "Loaded noise archive from:",archname
            return df
        except:
            pass
     
    nms = []
    for fname in info['files']:
        try:
            nms.extend(load_noise_pkl(fname))
        except Exception,e:
            print "couldn't get noise measurements from",fname,"error was:",e
    pnames = nms[0].fit_params.keys()
    try:
        pnames.remove('a')
    except:
        pass
    data = {}
    for pn in pnames:
        data[pn] = [nm.fit_params[pn].value for nm in nms]
        data[pn + '_err'] = [nm.fit_params[pn].stderr for nm in nms]
    avals = []
    aerrs = []
    for nm in nms:
        if nm.fit_params.has_key('a'):
            avals.append(nm.fit_params['a'].value)
            aerrs.append(nm.fit_params['a'].stderr)
        else:
            avals.append(np.nan)
            aerrs.append(np.nan)
    data['a'] = avals
    data['a_err'] = aerrs
    attrs = nms[0].__dict__.keys()
    attrs.remove('fit_params')
    attrs.remove('resonator_model')
    private = [x for x in attrs if x.startswith('_')]
    for private_var in private:
        attrs.remove(private_var)
    for pn in attrs:
        data[pn] = [getattr(nm,pn) for nm in nms]
    pca_fr = data['pca_freq'][0]
    noise150 = []
    noise30k = []
    for nm in nms:
        mask150 = (nm.pca_freq > 100) & (nm.pca_freq < 200)
        mask30k = (nm.pca_freq > 20e3) & (nm.pca_freq < 40e3)
        noise150.append(nm.pca_eigvals[1,mask150].mean())
        noise30k.append(nm.pca_eigvals[1,mask30k].mean())
    data['noise_150_Hz'] = noise150
    data['noise_30_kHz'] = noise30k
    data['resonator_id'] = [info['index_to_resnum'][nm.resonator_index] for nm in nms]
    
    lgs = []
    cgs = []
    for nm in nms:
        try:
            lg = lg_5x4[info['index_to_resnum'][nm.resonator_index]]
            cg = cap_5x4[info['index_to_resnum'][nm.resonator_index]]
        except IndexError:
            lg = np.nan
            cg = np.nan
        lgs.append(lg)
        cgs.append(cg)
                
            
    data['Lg'] = lgs
    data['Cg'] = cgs
    df = pd.DataFrame(data)
    df['round_temp'] = np.round(df['end_temp']*1000/10)*10
    
    try:
        np.save(archname,df.to_records())
    except Exception,e:
        print "failed to pickle",e
    return df

def load_archive(fn):
    npa = np.load(fn)
    df = pd.DataFrame.from_records(npa)
    return df
    