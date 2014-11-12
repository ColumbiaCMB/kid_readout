import numpy as np
from matplotlib import pyplot as plt
import os

import kid_readout.analysis.fit_pulses
from kid_readout.analysis.noise_measurement import SweepNoiseMeasurement, save_noise_pkl
import kid_readout.analysis.resonator
import kid_readout.utils.readoutnc
import kid_readout.analysis.resources.skip5x4
import kid_readout.analysis.fit_pulses
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

file_id_to_res_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 17, 16, 15, 14, 13, 10, 9]


def process_file(filename):
    print filename
    try:
        rnc = kid_readout.utils.readoutnc.ReadoutNetCDF(filename)
        num_power_steps = len(rnc.timestreams)-1 # last time stream is modulated measurement
        resonator_ids = np.unique(rnc.sweeps[0].index)
        noise_on_measurements = []
        noise_modulated_measurements = []
        noise_off_sweep_params = []
        for resonator_id in resonator_ids:
            power_steps_mmw_on = []
            for idx in range(num_power_steps):
                noise_on_measurement = SweepNoiseMeasurement(sweep_filename=filename, sweep_group_index=idx,
                                                             timestream_group_index=idx,
                                                             resonator_index=resonator_id,
                                                             delay_estimate=-63.0)
                power_steps_mmw_on.append(noise_on_measurement)
                noise_on_measurement._close_files()
            noise_modulated_measurement = SweepNoiseMeasurement(sweep_filename=filename, sweep_group_index=0,
                                                                timestream_group_index=num_power_steps,
                                                                resonator_index=resonator_id,
                                                                deglitch_threshold=None,
                                                                delay_estimate=-63.0)
            #all the zbd_voltages are the same, so we can grab any of them
            noise_modulated_measurement.zbd_voltage = rnc.timestreams[-1].zbd_voltage[0]
            for noise_on_measurement in power_steps_mmw_on:
                noise_on_measurement.zbd_voltage = rnc.timestreams[-1].zbd_voltage[0]
            if noise_modulated_measurement.timestream_modulation_period_samples != 0:
                noise_modulated_measurement.folded_projected_timeseries = noise_modulated_measurement.projected_timeseries.reshape((-1, noise_modulated_measurement.timestream_modulation_period_samples))
                folded = noise_modulated_measurement.folded_projected_timeseries.mean(0)
                high, low, rising_edge = kid_readout.analysis.fit_pulses.find_high_low(folded)
                noise_modulated_measurement.folded_projected_timeseries = np.roll(noise_modulated_measurement.folded_projected_timeseries,-rising_edge, axis=1)
            else:
                noise_modulated_measurement.folded_projected_timeseries = None

            fr, s21, err = rnc.sweeps[-1].select_by_index(resonator_id)
            noise_off_sweep = kid_readout.analysis.resonator.fit_best_resonator(fr, s21, errors=err)
            noise_off_sweep_params.append(noise_off_sweep.result.params)
            noise_on_measurements.extend(power_steps_mmw_on)
            noise_modulated_measurements.append(noise_modulated_measurement)
            noise_modulated_measurement._close_files()
        rnc.close()
        data = dict(noise_on_measurements=noise_on_measurements,
                    noise_modulated_measurements=noise_modulated_measurements,
                    noise_off_sweeps=noise_off_sweep_params)
        blah, fbase = os.path.split(filename)
        fbase, ext = os.path.splitext(fbase)
        pklname = os.path.join('/home/data', 'mmw_noise_steps_' + fbase + '.pkl')
        save_noise_pkl(pklname, data)
        return data
    except KeyboardInterrupt:
        return None


if __name__ == "__main__":
    import glob
    import multiprocessing

    #fns = glob.glob('/home/data2/2014-10-01*mmwnoisestep*.nc')
    #fns = glob.glob('/home/data2/2014-10-15*mmwnoisestep*.nc')
    #fns = glob.glob('/home/data2/2014-10-17*mmwnoisestep*.nc')
    #fns = glob.glob('/home/data2/2014-10-18*mmwnoisestep*.nc')
    fns = glob.glob('/home/data2/2014-*mmw*step*.nc')
    fns.sort()
    if True:
        pool = multiprocessing.Pool(4)
        pool.map(process_file,fns)

    else:
        for fn in fns:
            blah = process_file(fn)
            if blah is None:
                break

