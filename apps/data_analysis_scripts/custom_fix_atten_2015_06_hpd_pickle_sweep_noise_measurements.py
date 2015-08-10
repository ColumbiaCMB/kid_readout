# Automatically associate sweeps with timestreams
import os
import time
import numpy as np
from kid_readout.utils.readoutnc import ReadoutNetCDF
from kid_readout.analysis.noise_measurement import SweepNoiseMeasurement, save_noise_pkl
import cPickle
from kid_readout.roach.tools import ntone_power_correction

cryostats=dict(detectors='HPD',readout='StarCryo')

atten_map = {#"/data/detectors/2015-06-26_135847_dark.nc": [],
            #"/data/detectors/2015-06-26_142600_dark.nc": [],
            "/data/detectors/2015-06-26_151943_dark.nc": [40,30,20,10,0],
            "/data/detectors/2015-06-26_154035_dark.nc": [40,30,20,10,0],
            "/data/detectors/2015-06-26_171018_dark.nc": [40,30,20,10,0],
            "/data/detectors/2015-06-26_173844_dark.nc": [40,30,20,10,0],
            "/data/detectors/2015-06-26_180144_dark.nc": [40,30,20,10,0],
            "/data/detectors/2015-06-26_182443_dark.nc": [40,30,20,10,0],
            "/data/detectors/2015-06-26_222304_dark.nc": [40,30,20,10,0],
#            "/data/detectors/2015-06-27_015155_dark.nc": [],
#            "/data/detectors/2015-06-27_053941_dark.nc": [],
            }

def find_closest_sweep(ts,sweeps):
    start_epoch = ts.epoch.min()
#    print "ts",time.ctime(start_epoch)
    best_index = 0
    for sweep_index in range(len(sweeps)):
#        print sweep_index,time.ctime(sweeps[sweep_index].end_epoch)
        if sweeps[sweep_index].end_epoch < start_epoch:
            best_index = sweep_index
#    print "found",best_index
    return best_index

def extract_and_pickle(nc_filename):
    basedir = os.path.split(nc_filename)[0] # should make this more robust, currently assumes all nc files are in top
    #  level of /data/<machine>/*.nc
    machine = os.path.split(basedir)[1]
    cryostat = cryostats[machine]
    try:
        print("Processing {}".format(nc_filename))
        snms = []
        rnc = ReadoutNetCDF(nc_filename)
        for timestream_index,timestream in enumerate(rnc.timestreams):
            if timestream.epoch.shape[0] == 0:
                print "no timestreams in", nc_filename
                return
            start_epoch = timestream.epoch.min()
            sweep_index = find_closest_sweep(timestream,rnc.sweeps)
            sweep = rnc.sweeps[sweep_index]
            sweep_epoch = sweep.end_epoch
            resonator_indexes = np.array(list(set(sweep.index)))
            resonator_indexes.sort()
            print "%s: timestream[%d] at %s, associated sweep[%d] at %s, %d resonators" % (nc_filename,timestream_index,
                                                                                       time.ctime(start_epoch),
                                                                                       sweep_index,
                                                                                       time.ctime(sweep_epoch),
                                                                                       len(resonator_indexes))
            for resonator_index in resonator_indexes:
                snm = SweepNoiseMeasurement(rnc, sweep_group_index=sweep_index,
                                            timestream_group_index=timestream_index,
                                            resonator_index=resonator_index, cryostat=cryostat)
                if nc_filename in atten_map:
                    atten = atten_map[nc_filename][timestream_index]
                    ntone_correction = ntone_power_correction(16)
                    print "overriding attenuation",atten
                    snm.atten = atten
                    snm.total_dac_atten = atten +ntone_correction
                    snm.power_dbm = snm.dac_chain_gain - snm.total_dac_atten

                try:
                    snm.zbd_voltage = timestream.zbd_voltage[0]
                except AttributeError:
                    pass
                pkld = cPickle.dumps(snm,cPickle.HIGHEST_PROTOCOL)
                del snm
                snm = cPickle.loads(pkld)
                snms.append(snm)
        rnc.close()
        pkl_filename = os.path.join(basedir,'pkl', os.path.splitext(os.path.split(nc_filename)[1])[0] + '.pkl')
        save_noise_pkl(pkl_filename, snms)
        print("Saved {}".format(pkl_filename))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print "failed on",nc_filename,e


if __name__ == '__main__':
    import sys
    from glob import glob
    try:
        threads = int(sys.argv[1])
        filenames = atten_map.keys()
        filenames.sort()
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
