import numpy as np
import time
import sys
from kid_readout.utils import roach_interface, data_file, sweeps
from kid_readout.analysis.resonator import Resonator
from kid_readout.analysis.resonator import fit_best_resonator
from kid_readout.equipment import hittite_controller
from kid_readout.equipment import lockin_controller
from kid_readout.utils import acquire

lockin = lockin_controller.lockinController()
print lockin.get_idn()

ri = roach_interface.RoachBaseband()

def source_on():
    return ri.set_modulation_output(rate='low')


def source_off():
    return ri.set_modulation_output(rate='high')


def source_modulate(rate=7):
    return ri.set_modulation_output(rate=rate)


# Wideband
mmw_source_frequency = -1.0
suffix = "mmwnoisestep"

# Narrowband
#suffix = "mmwtonestep"
#f_mmw_source = 149e9
#hittite = hittite_controller.hittiteController()
#hittite.set_power(0)  # in dBm
#hittite.set_freq(f_mmw_source/12)  # in Hz
#hittite.on()

#f0s = np.load('/home/data2/resonances/2014-12-06_140825_0813f8_fit_16.npy')
f0s = np.load('/home/flanigan/f_r_3p5_turns.npy')

# hackalicious ... these are now the source-off resonances
original_f0s = np.load('/home/data2/resonances/2014-12-06_140825_0813f8_fit_16.npy')
# This was set to 0.999 and was causing a problem with duplicate tones.
source_on_freq_scale = 1.0  # nominally 1 if low-ish power

# Allow for downward movement
downward_shift = 54
coarse_exponent = 19
coarse_n_samples = 2**coarse_exponent
coarse_frequency_resolution = ri.fs / coarse_n_samples  # about 1 kHz
coarse_offset_integers = acquire.offset_integers[coarse_exponent][:-1] - downward_shift
coarse_offset_freqs = coarse_frequency_resolution * coarse_offset_integers

fine_exponent = 21
fine_n_samples = 2**fine_exponent
fine_frequency_resolution = ri.fs / fine_n_samples  # about 0.25 kHz
# Drop the 31st point so as to not force the added points to occupy the 32nd memory slot, which may not work.
fine_offset_integers = acquire.offset_integers[fine_exponent][:-1]
fine_offset_freqs = fine_frequency_resolution * fine_offset_integers

max_fit_error = 0.5
use_fmin = False
attenlist = [41, 38, 35, 32, 29, 26, 23]

while True:
    start = time.time()
    source_modulate()
    print "*" * 40
    print "Enter mmw attenuator values as a tuple i.e.: 6.5,6.5 or type exit to stop collecting data"
    mmw_atten_str = raw_input("mmw attenuator values: ")
    if mmw_atten_str == 'exit':
        source_off()
        break
    else:
        mmw_atten_turns = eval(mmw_atten_str)

    # Do a coarse sweep with the source on
    mmw_source_modulation_freq = source_on()
    print "setting attenuator to", attenlist[0]
    ri.set_dac_attenuator(attenlist[0])
    f0binned = coarse_frequency_resolution * np.round(f0s / coarse_frequency_resolution)
    measured_freqs = sweeps.prepare_sweep(ri, f0binned, coarse_offset_freqs, nsamp=coarse_n_samples)
    print "loaded waveforms in", (time.time() - start), "seconds"
    sweep_data = sweeps.do_prepared_sweep(ri, nchan_per_step=f0s.size, reads_per_step=2)
    orig_sweep_data = sweep_data
    meas_cfs = []
    idxs = []
    delays = []
    for m in range(len(f0s)):
        fr, s21, errors = sweep_data.select_by_freq(f0s[m])
        thiscf = f0s[m] * source_on_freq_scale
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

    delay = np.median(delays)
    print "median delay is ", delay

    meas_cfs = np.array(meas_cfs)
    f0binned_meas = fine_frequency_resolution * np.round(meas_cfs / fine_frequency_resolution)

    # Why is this here?
    f0s = f0binned_meas

    measured_freqs = sweeps.prepare_sweep(ri, f0binned_meas, fine_offset_freqs, nsamp=fine_n_samples)
    print "loaded updated waveforms in", (time.time() - start), "seconds"

    sys.stdout.flush()
    time.sleep(1)

    df = data_file.DataFile(suffix=suffix)
    df.nc.mmw_atten_turns = mmw_atten_turns

    for k, atten in enumerate(attenlist):
        ri.set_dac_attenuator(atten)
        print "measuring at attenuation", atten

        # At the first attenuation, record the coarse sweep data with the fine sweep data; at other attenuations, there
        # is only fine sweep data
        df.log_hw_state(ri)
        if k != 0:
            orig_sweep_data = None
        sweep_data = sweeps.do_prepared_sweep(ri, nchan_per_step=f0s.size, reads_per_step=2, sweep_data=orig_sweep_data)
        df.add_sweep(sweep_data)

        meas_cfs = []
        idxs = []
        for m in range(len(f0s)):
            fr, s21, errors = sweep_data.select_by_freq(f0s[m])
            thiscf = f0s[m] * source_on_freq_scale
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
        print meas_cfs

        # At the first attenuation, add a new bank containing the measured resonances;
        if k == 0:
            ri.add_tone_freqs(np.array(meas_cfs))
            ri.select_bank(ri.tone_bins.shape[0] - 1)
        # otherwise, use the tone bank for which index 0 is closest to the fit resonance.
        else:
            best_bank = (np.abs((ri.tone_bins[:, 0] * ri.fs / ri.tone_nsamp) - meas_cfs[0]).argmin())
            print "using bank", best_bank
            print "offsets:", ((ri.tone_bins[best_bank, :] * ri.fs / ri.tone_nsamp) - meas_cfs)
            ri.select_bank(best_bank)
        ri._sync()
        time.sleep(0.5)

        df.log_hw_state(ri)
        nsets = len(meas_cfs) / f0s.size
        tsg = None
        for iset in range(nsets):
            selection = range(len(meas_cfs))[iset::nsets]
            ri.select_fft_bins(selection)
            ri._sync()
            time.sleep(0.4)
            t0 = time.time()
            dmod, addr = ri.get_data_seconds(30)
            x, y, r, theta = lockin.get_data()
            # Record the timestream data taken using the selected tone bank.
            tsg = df.add_timestream_data(dmod, ri, t0, tsg=tsg, mmw_source_freq=mmw_source_frequency,
                                         mmw_source_modulation_freq=mmw_source_modulation_freq,
                                         zbd_voltage=x)
            df.sync()
            print "done with sweep"

    # Take a timestream at the first attenuation using the tone bank corresponding to this attenuation.
    ri.set_dac_attenuator(attenlist[0])
    mmw_source_modulation_freq = source_modulate()
    ri.select_bank(ri.tone_bins.shape[0] - 1)
    df.log_hw_state(ri)
    nsets = len(meas_cfs) / f0s.size
    tsg = None
    for iset in range(nsets):
        selection = range(len(meas_cfs))[iset::nsets]
        ri.select_fft_bins(selection)
        ri._sync()
        time.sleep(0.4)
        t0 = time.time()
        dmod, addr = ri.get_data_seconds(4)
        x, y, r, theta = lockin.get_data()

        tsg = df.add_timestream_data(dmod, ri, t0, tsg=tsg, mmw_source_freq=mmw_source_frequency,
                                     mmw_source_modulation_freq=mmw_source_modulation_freq,
                                     zbd_voltage=x)
        df.sync()
        print "done with sweep"

    # Source off
    mmw_source_modulation_freq = source_off()
    print "setting attenuator to", attenlist[0]
    ri.set_dac_attenuator(attenlist[0])
    # Above, the line
    # f0s = f0binned_meas
    # changes this variable to correspond to the source-on fine resolution fit frequencies, which may be much lower than
    # the originals. Using the coarse offsets in the sweep may cause the sweep to miss the resonances entirely.
    f0binned = coarse_frequency_resolution * np.round(original_f0s / coarse_frequency_resolution)
    # To measure with the source off, remove the downward shift in the offsets.
    measured_freqs = sweeps.prepare_sweep(ri, f0binned,
                                          coarse_offset_freqs + coarse_frequency_resolution * downward_shift,
                                          nsamp=coarse_n_samples)
    print "loaded waveforms in", (time.time() - start), "seconds"

    sweep_data = sweeps.do_prepared_sweep(ri, nchan_per_step=f0s.size, reads_per_step=2)
    orig_sweep_data = sweep_data
    meas_cfs = []
    idxs = []
    delays = []
    for m in range(len(original_f0s)):
        fr, s21, errors = sweep_data.select_by_freq(original_f0s[m])
        thiscf = original_f0s[m]
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

    delay = np.median(delays)
    print "median delay is ", delay
    meas_cfs = np.array(meas_cfs)
    f0binned_meas = fine_frequency_resolution * np.round(meas_cfs / fine_frequency_resolution)
    f0s = f0binned_meas
    measured_freqs = sweeps.prepare_sweep(ri, f0binned_meas, fine_offset_freqs, nsamp=fine_n_samples)
    print "loaded updated waveforms in", (time.time() - start), "seconds"

    sys.stdout.flush()
    time.sleep(1)

    df.log_hw_state(ri)
    sweep_data = sweeps.do_prepared_sweep(ri, nchan_per_step=f0s.size, reads_per_step=2, sweep_data=orig_sweep_data)
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

    print meas_cfs

    ri.add_tone_freqs(np.array(meas_cfs))
    ri.select_bank(ri.tone_bins.shape[0] - 1)

    df.log_hw_state(ri)
    nsets = len(meas_cfs) / f0s.size
    tsg = None
    for iset in range(nsets):
        selection = range(len(meas_cfs))[iset::nsets]
        ri.select_fft_bins(selection)
        ri._sync()
        time.sleep(0.4)
        t0 = time.time()
        dmod, addr = ri.get_data_seconds(30)
        x, y, r, theta = lockin.get_data()

        tsg = df.add_timestream_data(dmod, ri, t0, tsg=tsg, mmw_source_freq=mmw_source_frequency,
                                     mmw_source_modulation_freq=mmw_source_modulation_freq,
                                     zbd_voltage=x)
        df.sync()

    df.nc.close()
    print "completed in", ((time.time() - start) / 60.0), "minutes"
