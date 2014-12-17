import os
import numpy as np
from kid_readout.utils.readoutnc import ReadoutNetCDF
from kid_readout.analysis.resonator import fit_best_resonator
from kid_readout.analysis.noise_measurement import SweepNoiseMeasurement, save_noise_pkl
from kid_readout.analysis.fit_pulses import find_high_low


def extract_and_pickle(nc_filename):
    """
    The format is that the file contains equal numbers of sweeps and timestreams. The first sweep is used to locate
    the resonances and is taken with the source off at the lowest power level, i.e. the maximum attenuation. The
    first timestream is taken under the same conditions except that the source is modulated. Subsequent sweeps and
    timestreams are paired.

    :param nc_filename: the file name of the netCDF4 file with the above format.
    :return: a dictionary
    """
    try:
        all_noise_on = []
        all_noise_off = []
        all_noise_modulated = []
        all_coarse_sweep_params = []
        coarse_sweep_index = 0
        modulated_timestream_index = 0

        print("Processing {}".format(nc_filename))
        rnc = ReadoutNetCDF(nc_filename)
        resonator_indices = sorted(set(rnc.sweeps[0].index))
        n_attenuations = len(rnc.sweeps) - 1

        for resonator_index in resonator_indices:
            noise_on = []
            for on_index in range(1, n_attenuations, 2):
                noise_on.append(SweepNoiseMeasurement(nc_filename, resonator_index=resonator_index,
                                                      sweep_group_index=on_index, timestream_group_index=on_index))
            all_noise_on.extend(noise_on)

            noise_off = []
            for off_index in range(2, n_attenuations + 1, 2):
                noise_off.append(SweepNoiseMeasurement(nc_filename, resonator_index=resonator_index,
                                                       sweep_group_index=off_index, timestream_group_index=off_index))
            all_noise_off.extend(noise_off)

            # Create the modulated measurement from the modulated timestream and the noise off sweep at the same power.
            # Skip deglitching.
            attenuations = [snm.atten for snm in noise_off]
            off_max_attenuation_index = 1 + 2 * attenuations.index(max(attenuations)) + 1
            noise_modulated = SweepNoiseMeasurement(nc_filename, resonator_index=resonator_index,
                                                    sweep_group_index=off_max_attenuation_index,
                                                    timestream_group_index=modulated_timestream_index,
                                                    deglitch_threshold=None)
            noise_modulated.folded_projected_timeseries = noise_modulated.projected_timeseries.reshape(
                (-1, noise_modulated.timestream_modulation_period_samples))
            folded = noise_modulated.folded_projected_timeseries.mean(0)
            high, low, rising_edge = find_high_low(folded)
            noise_modulated.folded_projected_timeseries = np.roll(noise_modulated.folded_projected_timeseries,
                                                                  -rising_edge, axis=1)
            noise_modulated.folded_normalized_timeseries = np.roll(
                noise_modulated.normalized_timeseries.reshape((-1,
                                                               noise_modulated.timestream_modulation_period_samples)),
                -rising_edge, axis=1)
            all_noise_modulated.append(noise_modulated)

            # Save only the Parameters object from a fit to the coarse sweep.
            freq, s21, err = rnc.sweeps[coarse_sweep_index].select_by_index(resonator_index)
            coarse_resonator = fit_best_resonator(freq, s21, errors=err)
            all_coarse_sweep_params.append(coarse_resonator.result.params)

        rnc.close()
        data = {'noise_on_measurements': all_noise_on,
                'noise_off_measurements': all_noise_off,
                'noise_modulated_measurements': all_noise_modulated,
                'coarse_sweep_params': all_coarse_sweep_params}
        # We decided to keep the .pkl files in /home/data regardless of origin.
        pkl_filename = os.path.join('/home/data/pkl', os.path.splitext(os.path.split(nc_filename)[1])[0] + '.pkl')
        save_noise_pkl(pkl_filename, data)
        print("Saved {}".format(pkl_filename))
    except KeyboardInterrupt:
        print("Aborting {}".format(nc_filename))


if __name__ == '__main__':
    import sys
    from glob import glob
    try:
        threads = int(sys.argv[1])
        filenames = []
        for arg in sys.argv[2:]:
            filenames.extend(glob(arg))
    except IndexError:
        print("python pickle_load_and_mmw_source_measurements.py <threads> <file patterns>")
        sys.exit()
    if threads == 1:
        for filename in filenames:
            extract_and_pickle(filename)
    else:
        import multiprocessing
        pool = multiprocessing.Pool(threads)
        pool.map(extract_and_pickle, filenames)
