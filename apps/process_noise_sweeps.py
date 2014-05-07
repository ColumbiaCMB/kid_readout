import glob
import multiprocessing
from kid_readout.analysis import noise_measurement


def process(filename):
    try:
        errors = noise_measurement.plot_noise_nc(filename)
    except KeyboardInterrupt:
        return None
    print errors
    return errors

if __name__ == "__main__":
    num_threads = 4
    
    files = glob.glob('/home/data2/2014-04-30*_power_sweep.nc')
    files.sort()
    
    pool = multiprocessing.Pool(num_threads)
    
    print pool.map(process,files)