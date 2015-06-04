import sys
import getopt
import numpy as np
from kid_readout.utils import acquire


if __name__ == '__main__':
    def usage():
        print("Usage!")

    # Defaults
    f_off = np.load('/data/readout/resonances/current.npy')
    f_on = f_off.copy()
    shift_ppm = 0
    f_mmw = 0
    suffix = "mmw"
    # Add option?
    #attenuation_list = [41, 38, 35, 32, 29, 26, 23]
    attenuation_list = [41, 32]

    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:n:s:m:x:", ("off=", "on=", "shift_ppm=", "mmw_ghz=", "suffix="))
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-f", "--off"):
            f_off = np.load(arg)
        elif opt in ("-n", "--on"):
            f_on = np.load(arg)
        elif opt in ("-s", "--shift_ppm"):
            shift_ppm = float(arg)
        elif opt in ("-m", "--mmw_ghz"):
            f_mmw = 1e9 * float(arg)
        elif opt in ("-x", "--suffix"):
            suffix = arg

    f_on *= 1 - 1e-6 * shift_ppm

    acquire.mmw_source_power_step(f_off, f_on, attenuation_list, f_mmw_source=f_mmw, suffix=suffix)
