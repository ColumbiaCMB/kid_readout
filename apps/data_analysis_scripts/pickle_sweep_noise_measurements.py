import os
from kid_readout.utils.readoutnc import ReadoutNetCDF
from kid_readout.analysis.noise_measurement import SweepNoiseMeasurement, save_noise_pkl

def process_file(nc_filename):
    try:
        print("Processing {}".format(nc_filename))
        snms = []
        rnc = ReadoutNetCDF(nc_filename)
        for index, (sweep, timestream) in enumerate(zip(rnc.sweeps, rnc.timestreams)):
            for resonator_index in set(list(sweep.index)):
                snms.append(SweepNoiseMeasurement(nc_filename, sweep_group_index=index, timestream_group_index=index,
                                                  resonator_index=resonator_index))
        rnc.close()
        pkl_filename = os.path.join('/home/data', os.path.splitext(os.path.split(nc_filename)[1])[0] + '.pkl')
        save_noise_pkl(pkl_filename, snms)
        print("Saved {}".format(pkl_filename))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    import sys
    from glob import glob
    filenames = glob(sys.argv[1])
    try:
        threads = int(sys.argv[2])
    except IndexError:
        threads = 1
    if threads == 1:
        for filename in filenames:
            process_file(filename)
    else:
        import multiprocessing
        pool = multiprocessing.Pool(threads)
        pool.map(process_file, filenames)


