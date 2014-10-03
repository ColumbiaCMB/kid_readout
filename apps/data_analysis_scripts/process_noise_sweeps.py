import glob
import multiprocessing
from kid_readout.analysis import noise_measurement


def process(filename):
    try:
        errors = noise_measurement.plot_noise_nc(filename,deglitch_threshold=7,delay_estimate=-7.0)
    except KeyboardInterrupt:
        return None
    print errors
    return errors

if __name__ == "__main__":
    num_threads = 4
    
#    files = glob.glob('/home/data/2014-02-*.nc')
    files = glob.glob('/home/data2/2014-08-23_*power.nc')
    #files += glob.glob('/home/data2/2014-08-22_2*power.nc')
    files.sort()
    print files
    
    pool = multiprocessing.Pool(num_threads)
    
    print pool.map(process,files)

