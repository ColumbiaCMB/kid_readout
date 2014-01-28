import numpy as np
import pandas as pd
import os
from kid_readout.utils.plot_nc import get_all_sweeps

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
                                   
sc5x4_0813f8_info = dict(chip_name = 'StarCryo_5x4_0813f8',
                         dark = True,
                         files = ['/home/data/2013-12-03_165209.nc',
                                  '/home/data/2013-12-03_174822.nc',
                                  '/home/data/2013-12-04_094322.nc',
                                  '/home/data/2013-12-04_102845.nc',
                                  '/home/data/2013-12-04_110917.nc',
                                  '/home/data/2013-12-04_124749.nc',
                                  '/home/data/2013-12-04_142111.nc',
                                  '/home/data/2013-12-04_171233.nc',
                                  ],
                         index_to_resnum = range(18) + [19,np.nan]
                         )

jpl5x4_info = dict(chip_name='JPL_5x4_0',
                   dark = True,
                   files = ['/home/data/2013-11-11_174636.nc',
                             '/home/data/2013-11-11_133816.nc',
                             '/home/data/2013-11-11_160805.nc',
                             '/home/data/2013-11-11_204951.nc',
                             '/home/data/2013-11-12_144238.nc',
                             '/home/data/2013-11-12_112124.nc',
                             '/home/data/2013-11-12_170617.nc',
                             '/home/data/2013-11-12_185013.nc'],
                   index_to_resnum = [1,2,3,6,8,9,12,13,14,15,17,19])
def build_archive(info, use_bifurcation = False, force_rebuild=False):
    archname = '/home/data/archive/%s.npy' % info['chip_name']
    df = None
    if not force_rebuild and os.path.exists(archname):
        try:
            df = load_archive(archname)
            return df
        except:
            pass
     
    swps = []
    for fname in info['files']:
        swps.extend(get_all_sweeps(fname, bif = use_bifurcation))
    pnames = swps[0].result.params.keys()
    data = {}
    for pn in pnames:
        data[pn] = [getattr(swp,pn) for swp in swps]
        data[pn + '_err'] = [swp.result.params[pn].stderr for swp in swps]
    
    for pn in ['Q_i','Q_e','epoch','power_dbm','atten','init_temp','last_temp','sweep_name']:
        data[pn] = [getattr(swp,pn) for swp in swps]
    data['dark'] = [info['dark'] for swp in swps]
    
    data['ridx'] = [info['index_to_resnum'][swp.index] for swp in swps]
    lgs = []
    cgs = []
    for swp in swps:
        try:
            lg = lg_5x4[info['index_to_resnum'][swp.index]]
            cg = cap_5x4[info['index_to_resnum'][swp.index]]
        except IndexError:
            lg = np.nan
            cg = np.nan
        lgs.append(lg)
        cgs.append(cg)
                
            
    data['Lg'] = lgs
    data['Cg'] = cgs
    data['chip_name'] = [info['chip_name'] for swp in swps]
    df = pd.DataFrame(data)
    np.save(('/home/data/archive/%s.npy' % info['chip_name']),df.to_records())
    return df

def load_archive(fn):
    npa = np.load(fn)
    df = pd.DataFrame.from_records(npa)
    return df


sc5x4_0813f8_data = build_archive(sc5x4_0813f8_info,force_rebuild=True)
jpl5x4_data = build_archive(jpl5x4_info,force_rebuild=True)