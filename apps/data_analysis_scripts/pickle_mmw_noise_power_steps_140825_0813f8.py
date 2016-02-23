import numpy as np
import os
import sys

import kid_readout.analysis.fit_pulses
from kid_readout.analysis.noise_measurement import SweepNoiseMeasurement, save_noise_pkl
import kid_readout.analysis.resonator
import kid_readout.measurement.io.readoutnc
import kid_readout.analysis.resources.skip5x4
import kid_readout.analysis.fit_pulses


def process_file(filename):
    print filename
    try:
        rnc = kid_readout.measurement.io.readoutnc.ReadoutNetCDF(filename)
        num_timestreams = len(rnc.timestreams)
        num_sweeps = len(rnc.sweeps)
        if num_timestreams == num_sweeps:
            has_source_off_timestream = False
            num_power_steps = num_timestreams - 1 # last time stream is modulated measurement
        elif num_timestreams == num_sweeps + 1:
            has_source_off_timestream = True
            num_power_steps = num_timestreams - 2
        else:
            raise Exception("Found unexpected number of timestreams %d and number of sweeps %d for file %s" %
                            (num_timestreams,num_sweeps,filename))
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
                                                             )
                power_steps_mmw_on.append(noise_on_measurement)
                noise_on_measurement._close_files()
            noise_modulated_measurement = SweepNoiseMeasurement(sweep_filename=filename, sweep_group_index=0,
                                                                timestream_group_index=num_power_steps,
                                                                resonator_index=resonator_id,
                                                                deglitch_threshold=None,
                                                                )
            if has_source_off_timestream:
                noise_off_measurement = SweepNoiseMeasurement(sweep_filename=filename, sweep_group_index=num_sweeps-1,
                                                              timestream_group_index=num_timestreams-1,
                                                              resonator_index=resonator_id)

            #all the zbd_voltages are the same, so we can grab any of them
            noise_modulated_measurement.zbd_voltage = rnc.timestreams[num_power_steps].zbd_voltage[0]
            for noise_on_measurement in power_steps_mmw_on:
                noise_on_measurement.zbd_voltage = rnc.timestreams[num_power_steps].zbd_voltage[0]
            if noise_modulated_measurement.timestream_modulation_period_samples != 0:
                noise_modulated_measurement.folded_projected_timeseries = noise_modulated_measurement.projected_timeseries.reshape((-1, noise_modulated_measurement.timestream_modulation_period_samples))
                folded = noise_modulated_measurement.folded_projected_timeseries.mean(0)
                high, low, rising_edge = kid_readout.analysis.fit_pulses.find_high_low(folded)
                noise_modulated_measurement.folded_projected_timeseries = np.roll(noise_modulated_measurement.folded_projected_timeseries,-rising_edge, axis=1)
                noise_modulated_measurement.folded_normalized_timeseries = np.roll(
                    noise_modulated_measurement.normalized_timeseries.reshape((-1,noise_modulated_measurement.timestream_modulation_period_samples)),
                    -rising_edge, axis=1)
            else:
                noise_modulated_measurement.folded_projected_timeseries = None

            if not has_source_off_timestream:
                fr, s21, err = rnc.sweeps[-1].select_by_index(resonator_id)
                noise_off_sweep = kid_readout.analysis.resonator.fit_best_resonator(fr, s21, errors=err)
                noise_off_sweep_params.append(noise_off_sweep.result.params)
            # This will append a SweepNoiseMeasurement to noise_off_sweep_params
            else:
                noise_off_measurement.zbd_voltage = rnc.timestreams[num_power_steps].zbd_voltage[0]
                noise_off_sweep_params.append(noise_off_measurement)


            noise_on_measurements.extend(power_steps_mmw_on)
            noise_modulated_measurements.append(noise_modulated_measurement)
            noise_modulated_measurement._close_files()
        rnc.close()
        data = dict(noise_on_measurements=noise_on_measurements,
                    noise_modulated_measurements=noise_modulated_measurements,
                    noise_off_sweeps=noise_off_sweep_params)
        blah, fbase = os.path.split(filename)
        fbase, ext = os.path.splitext(fbase)
        pklname = os.path.join('/data/readout/pkl', fbase + '.pkl')
        save_noise_pkl(pklname, data)
        return data
    except KeyboardInterrupt:
        return None


if __name__ == '__main__':
    import sys
    from glob import glob
    try:
        threads = int(sys.argv[1])
        filenames = []
        for arg in sys.argv[2:]:
            filenames.extend(glob(arg))
    except IndexError:
        print("python pickle_mmw_noise_power_steps_whatever.py <threads> <file patterns>")
        sys.exit()
    if threads == 1:
        for filename in filenames:
            process_file(filename)
    else:
        import multiprocessing
        pool = multiprocessing.Pool(threads)
        pool.map(process_file, filenames)
