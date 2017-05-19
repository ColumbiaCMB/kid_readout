__author__ = 'gjones'
import time

import numpy as np
from matplotlib import pyplot as plt

import kid_readout.utils.acquire
import kid_readout.roach.baseband
import kid_readout.utils.data_file
import kid_readout.analysis.resonator
import kid_readout.analysis.iqnoise


ri = kid_readout.roach.baseband.RoachBaseband()
ri.set_dac_attenuator(38)

f0s = np.load('/home/data2/resonances/2014-12-06_140825_0813f8_fit_16.npy')

coarse_sweep = kid_readout.utils.acquire.sweep(roach=ri,center_frequencies=f0s,sample_exponent=19)

fit_f0s = []
fits = []
for k in np.unique(coarse_sweep.sweep_indexes):
    fr,s21,err = coarse_sweep.select_index(k)
    res = kid_readout.analysis.resonator.fit_best_resonator(fr,s21,errors=err,delay_estimate=ri.hardware_delay_estimate*1e6)
    fits.append(res)
    fit_f0s.append(res.f_0)
fits.sort(key=lambda r: r.f_0)
fit_f0s.sort()
fit_f0s = np.array(fit_f0s)

print f0s-fit_f0s

df = kid_readout.utils.data_file.DataFile(suffix='find_noise')
df.log_hw_state(ri)
df.add_sweep(coarse_sweep)

measured_f0s=ri.set_tone_freqs(fit_f0s,nsamp=2**21)

highlight = 1
while True:
    df.log_hw_state(ri)
    start_time = time.time()
    d,addr = ri.get_data_seconds(30)
    df.add_timestream_data(d,ri,t0=start_time)
    res = fits[highlight]
    normalized = res.normalize(measured_f0s[highlight], d[:,highlight]*ri.wavenorm)
    projected = res.project_s21_to_delta_freq(measured_f0s[highlight], normalized,s21_already_normalized=True)
    model = res.normalized_model(np.linspace(res.f_0-0.05,res.f_0+0.05,1000))
    fr,S,evals,evects,angles,piq = kid_readout.analysis.iqnoise.pca_noise(projected,Fs=256e6/2**14)
    plt.subplot(211)
    plt.plot(model.real,model.imag,'r')
    plt.plot(normalized.real,normalized.imag,'k,')
    plt.subplot(212)
    plt.loglog(fr,evals[1,:])
    plt.loglog(fr,evals[0,:],'r')

    plt.show()






