import os
from kid_readout.analysis.noise_measurement import SweepNoiseMeasurement, save_noise_pkl
from kid_readout.utils import readoutnc


def extract_and_pickle(nc_filename):
    """
    Ignore the coarse sweeps and create two SweepNoiseMeasurements that both use the fine sweep.
    """
    print("Processing {}".format(nc_filename))
    snms = []
    rnc = readoutnc.ReadoutNetCDF(nc_filename)
    if len(rnc.sweeps) != len(rnc.timestreams):
        raise ValueError("The number of sweeps does not match the number of timestreams in {}".format(nc_filename))
    for fine_index in range(1, len(rnc.sweeps), 2):
        fine_sweep = rnc.sweeps[fine_index]
        off_index = fine_index - 1
        on_index = fine_index
        for resonator_index in set(fine_sweep.index):
            off_snm = SweepNoiseMeasurement(nc_filename, sweep_group_index=fine_index,
                                            timestream_group_index=off_index, resonator_index=resonator_index)
            snms.append(off_snm)
            on_snm = SweepNoiseMeasurement(nc_filename, sweep_group_index=fine_index,
                                            timestream_group_index=on_index, resonator_index=resonator_index)
            snms.append(on_snm)
    rnc.close()
    # We decided to keep the .pkl files in /home/data regardless of origin.
    pkl_filename = os.path.join('/home/data/pkl', os.path.splitext(os.path.split(nc_filename)[1])[0] + '.pkl')
    save_noise_pkl(pkl_filename, snms)
    print("Saved {}".format(pkl_filename))


if __name__ == '__main__':
    import sys
    from glob import glob
    try:
        threads = int(sys.argv[1])
        filenames = []
        for arg in sys.argv[2:]:
            filenames.extend(glob(arg))
    except IndexError:
        print("python pickle_magnetic_pickup.py <threads> <file patterns>")
        sys.exit()
    if threads == 1:
        for filename in filenames:
            extract_and_pickle(filename)
    else:
        import multiprocessing
        pool = multiprocessing.Pool(threads)
        pool.map(extract_and_pickle, filenames)
