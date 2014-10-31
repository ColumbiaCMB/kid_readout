import numpy as np
import pandas as pd
import os
import pwd
from kid_readout.analysis.noise_measurement import load_noise_pkl
import glob
import collections


def normalize_f0(x):
    x['f_0_max'] = x[x.sweep_primary_package_temperature < 0.3]['f_0'].max()
    x['frac_f0'] = (x['f_0_max'] - x['f_0']) / x['f_0_max']
    return x


def refine_archive(df, fractional_f0_error=1e-7, fractional_Q_error=1e-2):
    df = df[df.f_0_err != 0]
    df = df[df.f_0_err/df.f_0 < fractional_f0_error]
    df = df[df.Q_err/df.Q < fractional_Q_error]
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
        noise_off = pkl['noise_off_sweeps']
        noise_mod = pkl['noise_modulated_measurements']

        pnames = noise_on[0].fit_params.keys()
        try:
            pnames.remove('a')
        except:
            pass
        for pn in pnames:
            data[pn].extend([nm.fit_params[pn].value for nm in noise_on])
            data[pn + '_err'].extend([nm.fit_params[pn].stderr for nm in noise_on])
            #data[pn].extend([params[pn].value for params in noise_off])
            #data[pn + '_err'].extend([params[pn].stderr for params in noise_off])

        avals = []
        aerrs = []
        for nm in noise_on:
            if 'a' in nm.fit_params:
                avals.append(nm.fit_params['a'].value)
                aerrs.append(nm.fit_params['a'].stderr)
            else:
                avals.append(np.nan)
                aerrs.append(np.nan)
        data['a'].extend(avals)
        data['a_err'].extend(aerrs)

        attrs = noise_on[0].__dict__.keys()
        attrs.remove('fit_params')
        attrs.remove('zbd_voltage')

        private = [x for x in attrs if x.startswith('_')]
        for private_var in private:
            attrs.remove(private_var)
        for pn in attrs:
            data[pn].extend([getattr(nm,pn) for nm in noise_on])

        # To do: add parameters for noise bands
        device_noise = []
        amplifier_noise = []
        for nm in noise_on:
            device_noise_mask = (nm.pca_freq > 1) & (nm.pca_freq < 110)
            amplifier_noise_mask = (nm.pca_freq > 5e3)
            device_noise.append(np.median(nm.pca_eigvals[1, device_noise_mask]))
            amplifier_noise.append(np.median(nm.pca_eigvals[1, amplifier_noise_mask]))
        data['device_noise'].extend(device_noise)
        data['amplifier_noise'].extend(amplifier_noise)
        data['resonator_id'].extend([info['index_to_resnum'][nm.resonator_index] for nm in noise_on])

        # Remove once SweepNoiseMeasurement has these attributes.
        # Add modulation frequency at which ZBD voltage was measured, from noise_mod?
        data['zbd_voltage'].extend([noise_mod[0].zbd_voltage] * len(noise_on))
        data['mmw_mod_duty_cycle'].extend([1.0] * len(noise_on))
        data['mmw_mod_phase'].extend([0.0] * len(noise_on))
        data['mmw_mod_freq'].extend([0.0] * len(noise_on))

        for nm in noise_on:
            nm._close_files()
        for nm in noise_mod:
            nm._close_files()

    df = pd.DataFrame(data)
    df['round_temp'] = np.round(df['end_temp']*1000/10)*10

    try:
        np.save(archname,df.to_records())
        os.chown(archname, os.getuid(), pwd.getpwnam('readout').pw_gid)
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
    df1028 = dict(files=glob.glob('/home/data/mmw_noise_steps_2014-10-28*.pkl'),index_to_resnum=np.arange(16)),
    df1029 = dict(files=glob.glob('/home/data/mmw_noise_steps_2014-10-29*.pkl'),index_to_resnum=np.arange(16)),
    df1030 = dict(files=glob.glob('/home/data/mmw_noise_steps_2014-10-30*.pkl'),index_to_resnum=np.arange(16)),
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
