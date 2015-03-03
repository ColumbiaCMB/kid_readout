import numpy as np
import pandas as pd
import os
import sys
try:
    import pwd
except ImportError:
    print "couldn't import pwd, ignoring"

from kid_readout.analysis.noise_measurement import load_noise_pkl
import glob
import collections
import time
import kid_readout.analysis.kid_response
kid_response = kid_readout.analysis.kid_response


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


def refine_archive(df, fractional_f0_error=1e-5, fractional_Q_error=1e-2):
    df = df[df.f_0_err != 0]
    df = df[df.f_0_err/df.f_0 < fractional_f0_error]
    df = df[df.Q_err/df.Q < fractional_Q_error]
    df = df.reset_index(drop=True).groupby(['resonator_id']).apply(normalize_f0).reset_index(drop=True)
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
    data_source_off = collections.defaultdict(list)
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
        #data['zbd_voltage'].extend([noise_mod[0].zbd_voltage] * len(noise_on))
        #data['mmw_mod_duty_cycle'].extend([1.0] * len(noise_on))
        #data['mmw_mod_phase'].extend([0.0] * len(noise_on))
        #data['mmw_mod_freq'].extend([0.0] * len(noise_on))




        # now deal with the noise off case

        # This is a horrible hack that doesn't get the noise off measurement into the archive.
        try:
            pnames = noise_off[0].keys()
        except AttributeError:
            pnames = noise_off[0].fit_params.keys()
            noise_off = [snm.fit_params for snm in noise_off]
        # End horrible hack

        try:
            pnames.remove('a')
        except:
            pass
        for pn in pnames:
            data_source_off[pn].extend([fit_params[pn].value for fit_params in noise_off])
            data_source_off[pn + '_err'].extend([fit_params[pn].stderr for fit_params in noise_off])

        avals = []
        aerrs = []
        for fit_params in noise_off:
            if 'a' in fit_params:
                avals.append(fit_params['a'].value)
                aerrs.append(fit_params['a'].stderr)
            else:
                avals.append(np.nan)
                aerrs.append(np.nan)
        data_source_off['a'].extend(avals)
        data_source_off['a_err'].extend(aerrs)

        # This is a tad sloppy; just realized the noise off here won't have a resonator id, so grab it from noise_mod
        data_source_off['resonator_id'].extend([info['index_to_resnum'][nm.resonator_index] for nm in noise_mod])

        attrs = noise_mod[0].__dict__.keys()
        attrs.remove('fit_params')
        attrs.remove('zbd_voltage')
        to_remove = [x for x in attrs if x.startswith('pca')]
        for attr in to_remove:
            attrs.remove(attr)

        private = [x for x in attrs if x.startswith('_')]
        for private_var in private:
            attrs.remove(private_var)

        for pn in attrs:
            if pn == 'timestream_modulation_duty_cycle':
                data_source_off[pn].extend([1.0 for nm in noise_mod])
            else:
                data_source_off[pn].extend([getattr(nm,pn) for nm in noise_mod])


        for nm in noise_on:
            nm._close_files()
        for nm in noise_mod:
            nm._close_files()

    df = pd.DataFrame(data)
    df['round_temp'] = np.round(df['end_temp']*1000/10)*10

    df_off = pd.DataFrame(data_source_off)
    df_off['round_temp'] = np.round(df['end_temp']*1000/10)*10

    df = pd.concat([df,df_off])
    df = df.reset_index(drop=True)
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
    """
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
    df1103 = dict(files=glob.glob('/home/data/mmw_noise_steps_2014-11-03*.pkl'),index_to_resnum=np.arange(16)),
    df1104 = dict(files=glob.glob('/home/data/mmw_noise_steps_2014-11-04*.pkl'),index_to_resnum=np.arange(16)),
    )
    """
    files = sys.argv[1:]
    infos = {
        'mmw_step': {'files': files, 'index_to_resnum': np.arange(16)}
    }

    for dfn,info in infos.items():
        df = build_archive(info,
                       force_rebuild=False,
                       archive_name=None)
        globals()[dfn] = df

    df = pd.concat([globals()[x] for x in infos.keys()])
    df = df.reset_index(drop=True)
    df = df.sort(['sweep_epoch'])
    df['zbd_volts_to_watts'] = 2200.
    df['zbd_correction'] = 1.0
    df.ix[df.sweep_epoch > time.mktime(time.strptime("2014-10-27 16:17","%Y-%m-%d %H:%M")),'zbd_correction'] = 1e4
    df['zbd_power'] = (df.zbd_voltage*df.zbd_correction)/df.zbd_volts_to_watts
    df['mmw_atten_total_turns'] = np.nan
    df.ix[~df.mmw_atten_turns.isnull(),'mmw_atten_total_turns'] = np.array([x.sum() for x in df[~df.mmw_atten_turns.isnull()].mmw_atten_turns])
    dfrall = refine_archive(df)
    dfr = dfrall[dfrall.atten == 39]
