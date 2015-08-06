import sys
import getopt
import numpy as np
from kid_readout.utils import acquire


if __name__ == '__main__':
    def usage():
        print("Usage!")

    # Defaults
    f_initial = np.load('/data/readout/resonances/current.npy')
    shift_ppm = 0
    suffix = "temperature"
    # Add option?
    attenuation_list = [41, 38, 35, 32, 29, 26, 23]

    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:s:x:", ("initial=", "shift_ppm=", "suffix="))
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-f", "--initial"):
            f_off = np.load(arg)
        elif opt in ("-s", "--shift_ppm"):
            shift_ppm = float(arg)
        elif opt in ("-x", "--suffix"):
            suffix = arg

    f_initial *= 1 - 1e-6 * shift_ppm

    acquire.sweeps_and_streams(f_initial, attenuation_list, suffix=suffix)
