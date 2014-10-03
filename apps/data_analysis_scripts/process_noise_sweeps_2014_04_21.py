import glob
import multiprocessing
from kid_readout.analysis import noise_measurement


def process(filename):
    try:
        errors = noise_measurement.plot_noise_nc(filename)
    except KeyboardInterrupt:
        return {}
    print errors
    return errors

if __name__ == "__main__":
    num_threads = 4
        
    files = glob.glob('/home/data2/2014-04-22_16*.nc') + glob.glob('/home/data2/2014-04-22_17*.nc')
    
    # + glob.glob('/home/data2/2014-04-22*.nc')
#    files += glob.glob('/home/data2/2014-04-22*.nc')
#    old_files = [x for x in files if x.find('_net') < 0]
    files = glob.glob('/home/data2/2014-05-17_07*.nc')
#    files = [x for x in files if x not in old_files]
    files.sort()
    print files
#    print noise_measurement.plot_noise_nc('/home/data2/2014-04-22_165921.nc')
#    7/0
    
    pool = multiprocessing.Pool(num_threads)
    
    print pool.map(process,files)