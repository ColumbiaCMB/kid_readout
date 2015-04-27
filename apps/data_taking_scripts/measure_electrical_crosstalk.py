__author__ = 'gjones'

import numpy as np
import time
import sys
from kid_readout.utils import roach_interface, data_file, sweeps
from kid_readout.analysis.resonator import Resonator
from kid_readout.analysis.resonator import fit_best_resonator
from kid_readout.equipment import hittite_controller
from kid_readout.utils import acquire

ri = roach_interface.RoachBaseband()


def source_on():
    return ri.set_modulation_output(rate='low')


def source_off():
    return ri.set_modulation_output(rate='high')


def source_modulate(rate=7):
    return ri.set_modulation_output(rate=rate)


source_off()

# Wideband
mmw_source_frequency = -1.0
#suffix = "mmwnoisestep"
suffix = "electrical_crosstalk"

f0s = np.load('/home/data2/resonances/2014-12-06_140825_0813f8_working_16.npy')

coarse_exponent = 20
coarse_n_samples = 2**coarse_exponent
coarse_frequency_resolution = ri.fs / coarse_n_samples  # about 1 kHz
coarse_offset_integers = acquire.offset_integers[coarse_exponent][:-1]
coarse_offset_freqs = coarse_frequency_resolution * coarse_offset_integers

attenlist = [41, 38, 35, 32, 29, 26, 23]


ri.set_dac_attenuator(attenlist[0])
f0binned = coarse_frequency_resolution * np.round(f0s / coarse_frequency_resolution)
start = time.time()
measured_freqs = sweeps.prepare_sweep(ri, f0binned, coarse_offset_freqs, nsamp=coarse_n_samples)
print "loaded waveforms in", (time.time() - start), "seconds"
sweep_data = sweeps.do_prepared_sweep(ri, nchan_per_step=f0s.size, reads_per_step=2)

meas_cfs = []
idxs = []
delays = []
use_fmin=False
max_fit_error = 0.5
for m in range(len(f0s)):
    fr, s21, errors = sweep_data.select_by_freq(f0s[m])
    thiscf = f0s[m]
    res = fit_best_resonator(fr[1:-1], s21[1:-1], errors=errors[1:-1])  # Resonator(fr,s21,errors=errors)
    delay = res.delay
    delays.append(delay)
    s21 = s21 * np.exp(2j * np.pi * res.delay * fr)
    res = fit_best_resonator(fr, s21, errors=errors)
    fmin = fr[np.abs(s21).argmin()]
    print "s21 fmin", fmin, "original guess", thiscf, "this fit", res.f_0, "delay", delay, "resid delay", res.delay
    if use_fmin:
        meas_cfs.append(fmin)
    else:
        if abs(res.f_0 - thiscf) > max_fit_error:
            if abs(fmin - thiscf) > max_fit_error:
                print "using original guess"
                meas_cfs.append(thiscf)
            else:
                print "using fmin"
                meas_cfs.append(fmin)
        else:
            print "using this fit"
            meas_cfs.append(res.f_0)
    idx = np.unravel_index(abs(measured_freqs - meas_cfs[-1]).argmin(), measured_freqs.shape)
    idxs.append(idx)

meas_cfs = np.array(meas_cfs)
df = data_file.DataFile(suffix=suffix)
df.add_sweep(sweep_data)
for res_id in range(len(meas_cfs)):
    print "blinking resonator",res_id
    on_amps = np.ones(meas_cfs.shape,dtype='float')
    off_amps = np.ones(meas_cfs.shape,dtype='float')
    off_amps[res_id] = 0
    normfact = 0.1
    nsamp = 2**22
    ri.set_tone_freqs(meas_cfs,nsamp=nsamp,load=False,normfact=normfact,amps=on_amps)
    phases = ri.phases.copy()
    qon = ri.qwave.copy()
    ri.set_tone_freqs(meas_cfs,nsamp=nsamp,load=False,normfact=normfact,amps=off_amps,phases=phases)
    qoff = ri.qwave.copy()
    qonoff = np.concatenate((qon,qoff))
    ri.load_waveform(qonoff)
    ri.tone_nsamp = qonoff.shape[0]
    ri.tone_bins = ri.tone_bins*2
    ri.select_bank(0)
    for k, atten in enumerate(attenlist):
        ri.set_dac_attenuator(atten)
        print "measuring at attenuation", atten
        df.log_hw_state(ri)
        time.sleep(1)
        t0 = time.time()
        dmod,addr = ri.get_data_seconds(15)

        tsg = df.add_timestream_data(dmod, ri, t0, mmw_source_freq=mmw_source_frequency,)
