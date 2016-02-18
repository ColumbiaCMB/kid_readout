import os
from kid_readout.measurement.io.readoutnc import ReadoutNetCDF
from kid_readout.analysis.noise_measurement import SweepNoiseMeasurement, save_noise_pkl


def extract_and_pickle(nc_filename):
    basedir = os.path.split(nc_filename)[0] # should make this more robust, currently assumes all nc files are in top
    #  level of /data/<machine>/*.nc
    try:
        print("Processing {}".format(nc_filename))
        snms = []
        rnc = ReadoutNetCDF(nc_filename)
        if len(rnc.sweeps) != len(rnc.timestreams):
            raise ValueError("The number of sweeps does not match the number of timestreams in {}".format(nc_filename))
        for index, (sweep, timestream) in enumerate(zip(rnc.sweeps, rnc.timestreams)):
            for resonator_index in set(sweep.index):
                snm = SweepNoiseMeasurement(nc_filename, sweep_group_index=index, timestream_group_index=index,
                                            resonator_index=resonator_index)
                try:
                    snm.zbd_voltage = timestream.zbd_voltage[0]
                except AttributeError:
                    pass
                snms.append(snm)
        rnc.close()
        # We decided to keep the .pkl files in /home/data regardless of origin.
        pkl_filename = os.path.join(basedir,'pkl', os.path.splitext(os.path.split(nc_filename)[1])[0] + '.pkl')
        save_noise_pkl(pkl_filename, snms)
        print("Saved {}".format(pkl_filename))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    import sys
    from glob import glob
    try:
        threads = int(sys.argv[1])
        filenames = []
        for arg in sys.argv[2:]:
            filenames.extend(glob(arg))
    except IndexError:
        print("python pickle_sweep_noise_measurements.py <threads> <file patterns>")
        sys.exit()
    if threads == 1:
        for filename in filenames:
            extract_and_pickle(filename)
    else:
        import multiprocessing
        pool = multiprocessing.Pool(threads)
        pool.map(extract_and_pickle, filenames)
