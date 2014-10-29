import numpy as np
import pandas as pd
import os
from kid_readout.analysis.noise_measurement import load_noise_pkl
import kid_readout.analysis.resonator
import glob
import socket
import collections

#sc5x4_0813f12_taped_dark_info = dict(chip_name = 'StarCryo_5x4_0813f12_taped_dark',
#                               dark = True,
#                               files = files,
#                               index_to_resnum = range(20)
#                               )

            #index_to_resnum = [1,2,3,6,8,9,12,13,14,15,17,19])


def normalize_f0(x):
    x['f_0_max'] = x[x.sweep_primary_package_temperature < 0.3]['noise_off_f_0'].max()
    x['noise_on_frac_f0'] = (x['f_0_max'] - x['noise_on_f_0']) / x['f_0_max']
    x['noise_off_frac_f0'] = (x['f_0_max'] - x['noise_off_f_0']) / x['f_0_max']
    return x

def refine_archive(df, fractional_f0_error=1e-7, fractional_Q_error=1e-2):
    df = df[df.noise_off_f_0_err != 0]
    df = df[df.noise_on_f_0_err != 0]
    df = df[df.noise_on_f_0_err/df.noise_on_f_0 < fractional_f0_error]
#    df = df[df.noise_off_f_0_err/df.noise_off_f_0 < fractional_f0_error]
    df = df[df.noise_on_Q_err/df.noise_on_Q < fractional_Q_error]
#    df = df[df.noise_off_Q_err/df.noise_off_Q < fractional_Q_error]
    df = df.groupby(['resonator_id']).apply(normalize_f0)
    return df

def build_archive(info, archive_name=None, force_rebuild=False):
    info['files'].sort()
    if archive_name is None:
        archive_name = os.path.splitext(os.path.basename(info['files'][0]))[0]
    archname = '/home/data/archive/%s.npy' % archive_name
    df = None
    if not force_rebuild and os.path.exists(archname):
        try:
            df = load_archive(archname)
            print "Loaded noise archive from:",archname
            return df
        except:
            pass
     
    data = collections.defaultdict(list)
    for fname in info['files']:
        if data:
            l0 = len(data.values()[0])
            for k,v in data.items():
                if len(v) != l0:
                    print k, l0, len(v)

        print "processing",fname
        pkl = load_noise_pkl(fname)
        noise_on = pkl['noise_on_measurements']
        noise_on_orig = noise_on
        if len(noise_on) > len(info['index_to_resnum']):
            noise_on = noise_on[::6]
        noise_off = pkl['noise_off_sweeps']
        noise_mod = pkl['noise_modulated_measurements']
        pnames = noise_on[0].fit_params.keys()
        try:
            pnames.remove('a')
        except:
            pass
        for pn in pnames:
            data['noise_on_'+ pn].extend([nm.fit_params[pn].value for nm in noise_on])
            data['noise_on_'+ pn + '_err'].extend([nm.fit_params[pn].stderr for nm in noise_on])
            data['noise_off_'+ pn].extend([params[pn].value for params in noise_off])
            data['noise_off_'+ pn + '_err'].extend([params[pn].stderr for params in noise_off])

        avals = []
        aerrs = []
        for nm in noise_on:
            if nm.fit_params.has_key('a'):
                avals.append(nm.fit_params['a'].value)
                aerrs.append(nm.fit_params['a'].stderr)
            else:
                avals.append(np.nan)
                aerrs.append(np.nan)
        data['noise_on_a'].extend(avals)
        data['noise_on_a_err'].extend(aerrs)

        avals = []
        aerrs = []
        for nm in noise_off:
            if nm.has_key('a'):
                avals.append(nm['a'].value)
                aerrs.append(nm['a'].stderr)
            else:
                avals.append(np.nan)
                aerrs.append(np.nan)
        data['noise_off_a'].extend(avals)
        data['noise_off_a_err'].extend(aerrs)


        attrs = noise_on[0].__dict__.keys()
        attrs.remove('fit_params')
        attrs.remove('zbd_voltage')

        if '_resonator_model' in attrs:
            attrs.remove('_resonator_model')
        private = [x for x in attrs if x.startswith('_')]
        for private_var in private:
            attrs.remove(private_var)
        for pn in attrs:
            data[pn].extend([getattr(nm,pn) for nm in noise_on])
        pca_fr = data['pca_freq'][0]
        noise100 = []
        noise7k = []
        for nm in noise_on:
            mask100 = (nm.pca_freq > 90) & (nm.pca_freq < 110)
            mask7k = (nm.pca_freq > 7e3)
            noise100.append(nm.pca_eigvals[1,mask100].mean())
            noise7k.append(nm.pca_eigvals[1,mask7k].mean())
        data['noise_100_Hz'].extend(noise100)
        data['noise_7_kHz'].extend(noise7k)
        data['resonator_id'].extend([info['index_to_resnum'][nm.resonator_index] for nm in noise_on])

        for nm in noise_mod:
            detuning = kid_readout.analysis.resonator.normalized_s21_to_detuning(nm.normalized_timeseries,
                                                                                 nm.resonator_model)
            folded = detuning.reshape((-1,128)).mean(0)
            response = folded.ptp()
            data['noise_fractional_response'].append(response)
            data['folded_noise_fractional_response'].append(folded)
            data['zbd_voltage'].append(nm.zbd_voltage)
#            data['zbd_power'].append(nm.zbd_power)
        for nm in noise_on_orig:
            nm._close_files()
        for nm in noise_mod:
            nm._close_files()

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


if __name__ == "__main__":

    infos = dict(
    df0924 = dict(files=glob.glob('/home/data/mmw_noise_steps_2014-09-24*.pkl'),index_to_resnum=np.arange(16)),
    df0929 = dict(files=glob.glob('/home/data/mmw_noise_steps_2014-09-29*.pkl'),index_to_resnum=np.arange(16)),
    df0930 = dict(files=glob.glob('/home/data/mmw_noise_steps_2014-09-30*.pkl'),index_to_resnum=np.arange(16)),
    df1001 = dict(files=glob.glob('/home/data/mmw_noise_steps_2014-10-01*.pkl'),index_to_resnum=np.arange(16)),
    df1014 = dict(files=glob.glob('/home/data/mmw_noise_steps_2014-10-14*.pkl'),index_to_resnum=np.arange(16)),
    df1015 = dict(files=glob.glob('/home/data/mmw_noise_steps_2014-10-15*.pkl'),index_to_resnum=np.arange(16)),
    df1017 = dict(files=glob.glob('/home/data/mmw_noise_steps_2014-10-17*.pkl'),index_to_resnum=np.arange(16)),
    df1018 = dict(files=glob.glob('/home/data/mmw_noise_steps_2014-10-18*.pkl'),index_to_resnum=np.arange(16)),
    df1022 = dict(files=glob.glob('/home/data/mmw_noise_steps_2014-10-22*.pkl'),index_to_resnum=np.arange(16)),
    df1024 = dict(files=glob.glob('/home/data/mmw_noise_steps_2014-10-24*.pkl'),index_to_resnum=np.arange(16)),
#    df1028 = dict(files=glob.glob('/home/data/mmw_noise_steps_2014-10-28*.pkl'),index_to_resnum=np.arange(16)),
    )

    for dfn,info in infos.items():
        df = build_archive(info,
                       force_rebuild=False,
                       archive_name=None)
        globals()[dfn] = df

    df = pd.concat([globals()[x] for x in infos.keys()])
    df = df.reset_index(drop=True)
    dfrall = refine_archive(df)
    dfr = dfrall[dfrall.atten == 39]