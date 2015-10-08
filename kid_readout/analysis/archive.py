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
from kid_readout.analysis import kid_response

def build_simple_archives(pklglob,index_to_id=None):
    pklnames = glob.glob(pklglob)
    dfs = []
    for pklname in pklnames:
        dfs.append(build_simple_archive([pklname],index_to_id=index_to_id))
    return pd.concat(dfs,ignore_index=True)

def build_simple_archive(pklnames, index_to_id = None, archive_name=None, archive_path=None):

    if not type(pklnames) is list:
        pklnames = glob.glob(pklnames)
    pklnames.sort()

    if archive_name is None:
        archive_name = os.path.splitext(os.path.basename(pklnames[0]))[0]
    if archive_path is None:
        archive_path = os.path.join(os.path.split(os.path.split(pklnames[0])[0])[0],'archive')
    archname = os.path.join(archive_path,('%s.npy' % archive_name))

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
        if not np.all(np.isfinite(x.pca_freq.iloc[k])):
            device_noise.append(np.nan)
            amplifier_noise.append(np.nan)
        else:
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

def add_total_mmw_attenuator_turns(df):
    df['mmw_atten_total_turns'] = np.nan
    df.ix[~df.mmw_atten_turns.isnull(),'mmw_atten_total_turns'] = np.array([np.sum(x) for x in df[~df.mmw_atten_turns.isnull()].mmw_atten_turns])
    return df

def get_constant_package_temperature_data(df,center_temperature=0.183,peak_excursion=0.005):
    return df[abs(df.sweep_primary_package_temperature-center_temperature) < peak_excursion]


def add_calibration(x,zbd_calibration_data=None):
    #print x.resonator_id.value_counts()
    rid = x['resonator_id'].iloc[0]
    print rid
    this_res = zbd_calibration_data[zbd_calibration_data.resonator_id==rid]
    if len(this_res) > 0:
        x['f_0_max'] = this_res.f_0_max.iloc[0]
        x['frac_f0'] = 1-x.f_0/x.f_0_max
        x['response_break_point'] = this_res.response_break_point.iloc[0]
        x['response_scale'] = this_res.response_scale.iloc[0]
        x['response_break_point_err'] = this_res.response_break_point_err.iloc[0]
        x['response_scale_err'] = this_res.response_scale_err.iloc[0]
    else:
        print "warning, resonator",rid,"has no calibration data"
        x['f_0_max'] = np.nan
        x['frac_f0'] = np.nan
        x['response_break_point'] = np.nan
        x['response_scale'] = np.nan
        x['response_break_point_err'] = np.nan
        x['response_scale_err'] = np.nan

    return x


def compute_reconstructed_power(df,calibration):
    df = df.groupby(['resonator_id']).apply(add_calibration,zbd_calibration_data=calibration).reset_index(drop=True)
    df['reconstructed_power'] = kid_response.fractional_freq_to_power(x=df.frac_f0,
                                                                     break_point=df.response_break_point,
                                                                     scale=df.response_scale)
    return df



def build_response_fit_function(make_plots=False, mcmc_length=500, mcmc_burn_in=400):
    def fit_response_mcmc(x):
        x['f_0_max'] = x.f_0.max()
        x['frac_f0'] = 1-x.f_0/x.f_0_max
        mask = (x.sweep_primary_load_temperature<5) & (x.zbd_power > 0)
        fit = kid_readout.analysis.kid_response.MCMCKidResponseFitter(x[mask]['frac_f0'],x[mask]['zbd_power'],
                                                                      errors = np.where(x[mask]['zbd_power']>1e-7,
                                                                                        x[mask]['zbd_power']*1e-2,
                                                                                        1e-8))
        fit.run(burn_in=mcmc_burn_in,length=mcmc_length)
        if make_plots:
            blah = fit.triangle()
        x['response_break_point']=fit.mcmc_params['break_point'].value
        x['response_scale']=fit.mcmc_params['scale'].value
        x['response_break_point_err']=fit.mcmc_params['break_point'].stderr
        x['response_scale_err']=fit.mcmc_params['scale'].stderr
        #x['watts_to_ppm'] = (2*pp[0]*x['noise_on_frac_f0']+pp[1])
        x['reconstructed_power'] = fit.to_power(x['frac_f0'])
        return x
    return fit_response_mcmc

def normalize_f0(x):
    x['f_0_max'] = x[x.sweep_primary_package_temperature < 0.3]['f_0'].max()
    x['frac_f0'] = (x['f_0_max'] - x['f_0']) / x['f_0_max']
    return x


def save_archive(df,archname):
    try:
        np.save(archname,df.to_records())
    except Exception,e:
        print "failed to pickle",e


def load_archive(fn):
    npa = np.load(fn)
    df = pd.DataFrame.from_records(npa)
    return df