# Automatically associate sweeps with timestreams
import os
import time
import numpy as np
from kid_readout.utils.readoutnc import ReadoutNetCDF
from kid_readout.analysis.noise_measurement import SweepNoiseMeasurement, save_noise_pkl
import kid_readout.analysis.fit_pulses
import cPickle

_settings_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0],'settings')
_default_settings_file = os.path.join(_settings_dir,'default_settings.py')
cryostats=dict(detectors='HPD',readout='StarCryo')

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

def extract_and_pickle(nc_filename, deglitch_threshold=5):
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
            modulation_state,modulation_frequency = rnc.get_modulation_state_at(start_epoch)
            try:
                manual_modulation_frequency = timestream.mmw_source_modulation_frequency[0]
            except AttributeError:
                manual_modulation_frequency = 0
            if modulation_state == 2 or manual_modulation_frequency > 0:
                this_deglitch_threshold = None
            else:
                this_deglitch_threshold = deglitch_threshold
            resonator_indexes = np.array(list(set(sweep.index)))
            resonator_indexes.sort()
            print "%s: timestream[%d] at %s, associated sweep[%d] at %s, %d resonators" % (nc_filename,timestream_index,
                                                                                       time.ctime(start_epoch),
                                                                                       sweep_index,
                                                                                       time.ctime(sweep_epoch),
                                                                                       len(resonator_indexes))
            if this_deglitch_threshold is None:
                print "Found modulation, not deglitching"
            for resonator_index in resonator_indexes:
                snm = SweepNoiseMeasurement(rnc, sweep_group_index=sweep_index,
                                            timestream_group_index=timestream_index,
                                            resonator_index=resonator_index, cryostat=cryostat,
                                            deglitch_threshold=this_deglitch_threshold)
                try:
                    snm.zbd_voltage = timestream.zbd_voltage[0]
                except AttributeError:
                    pass
                
                if snm.timestream_modulation_period_samples != 0:
                    snm.folded_projected_timeseries = snm.projected_timeseries.reshape((-1, snm.timestream_modulation_period_samples))
                    folded = snm.folded_projected_timeseries.mean(0)
                    high, low, rising_edge = kid_readout.analysis.fit_pulses.find_high_low(folded)
                    snm.folded_projected_timeseries = np.roll(snm.folded_projected_timeseries,-rising_edge,
                                                              axis=1).mean(0)
                    snm.folded_normalized_timeseries = np.roll(
                        snm.normalized_timeseries.reshape((-1,snm.timestream_modulation_period_samples)),
                        -rising_edge, axis=1).mean(0)
                
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
