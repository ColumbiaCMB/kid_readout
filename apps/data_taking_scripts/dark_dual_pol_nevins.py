import time
import sys

import numpy as np

from kid_readout.roach import heterodyne
from kid_readout.utils import data_file, sweeps
from kid_readout.analysis.resonator import fit_best_resonator
from kid_readout.utils import acquire



ri = heterodyne.RoachHeterodyne(adc_valon='/dev/ttyUSB0')
print '%08x' % ri.r.read_int('fftshift')
ri.set_fft_gain(4)

def source_on():
    return ri.set_modulation_output(rate='low')


def source_off():
    return ri.set_modulation_output(rate='high')


def source_modulate(rate=7):
    return ri.set_modulation_output(rate=rate)


# Wideband
#mmw_source_frequency = -1.0
#suffix = "mmwnoise_narrow_no_output_filter"

# Narrowband
suffix = "dark"

ri.set_lo(1210.0)

f0s = np.load('/data/readout/resonances/2015-11-03-starcryo-nevins-initial-resonances-160mK.npy')

f0s.sort()
#f0s = np.load('/home/data2/resonances/2014-12-06_140825_0813f8_fit_16.npy')
#f0s = np.load('/home/flanigan/f_r_3p5_turns.npy')


# Allow for downward movement
downward_shift = 0

nsamp = 2**15
step = 1
nstep = 128
f0binned = np.round(f0s * nsamp / 512.0) * 512.0 / nsamp
offset_bins = np.arange(-(nstep + 1), (nstep + 1)) * step

offsets = offset_bins * 512.0 / nsamp

measured_freqs = sweeps.prepare_sweep(ri, f0binned, offsets, nsamp=nsamp)


print f0s
print offsets * 1e6
print len(f0s)

max_fit_error=2.0
use_fmin = False
attenlist = [12,15]
at_once = 32

delay = 31.3

while True:
    start = time.time()
    source_modulate()
    ri.set_dac_atten(15)
    print "*" * 40
    print "Enter mmw attenuator values as a tuple i.e.: 6.5,6.5 or type exit to stop collecting data"
    mmw_atten_str = raw_input("mmw attenuator values: ")
    if mmw_atten_str == 'exit':
        source_off()
        break
    else:
        mmw_atten_turns = eval(mmw_atten_str)


    for k, atten in enumerate(attenlist):
        df = data_file.DataFile(suffix=suffix)
        df.nc.mmw_atten_turns = mmw_atten_turns
        ri.set_dac_attenuator(atten)
        print "measuring at attenuation", atten

        for source_state in ['off']:
            if source_state == 'on':
                mmw_source_modulation_freq = source_on()
            else:
                mmw_source_modulation_freq = source_off()

            df.log_hw_state(ri)
            sweep_data = sweeps.do_prepared_sweep(ri, nchan_per_step=f0s.size, reads_per_step=2)
            df.add_sweep(sweep_data)

            meas_cfs = []
            idxs = []
            for m in range(len(f0s)):
                fr, s21, errors = sweep_data.select_by_freq(f0s[m])
                thiscf = f0s[m]
                s21 = s21 * np.exp(2j * np.pi * delay * fr)
                res = fit_best_resonator(fr, s21, errors=errors)  # Resonator(fr,s21,errors=errors)
                fmin = fr[np.abs(s21).argmin()]
                print "s21 fmin", fmin, "original guess", thiscf, "this fit", res.f_0
                if 'a' in res.result.params or use_fmin: #k != 0 or use_fmin:
                    print "using fmin"
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
            for mm in range(meas_cfs.shape[0]-1):
                if meas_cfs[mm+1] - meas_cfs[mm] < 0.25:
                    print "updating frequency",mm,meas_cfs[mm+1],meas_cfs[mm]
                    meas_cfs[mm+1] = meas_cfs[mm] + 0.25
            print meas_cfs

            ri.add_tone_freqs(np.array(meas_cfs))
            ri.select_bank(ri.tone_bins.shape[0] - 1)
            ri._sync()
            time.sleep(0.5)

            df.log_hw_state(ri)
            nsets = len(meas_cfs) // at_once
            tsg = None
            for iset in range(nsets):
                selection = range(len(meas_cfs))[iset::nsets]
                ri.select_fft_bins(selection)
                ri._sync()
                time.sleep(0.4)
                t0 = time.time()
                dmod, addr = ri.get_data(16*16) #ri.get_data_seconds(30)
                # Record the timestream data taken using the selected tone bank.
                tsg = df.add_timestream_data(dmod, ri, t0, tsg=tsg,
                                             mmw_source_modulation_freq=mmw_source_modulation_freq,
                                             zbd_voltage=0)
                df.sync()
            print "done with timestream with source " + source_state


        print "closing",df.filename
        df.nc.close()

    print "completed in", ((time.time() - start) / 60.0), "minutes"
