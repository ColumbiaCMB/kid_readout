__author__ = 'gjones'
from kid_readout.equipment import hittite_controller, lockin_controller
import numpy as np
import time
import sys

hmc = hittite_controller.hittiteController('192.168.0.200')
li = lockin_controller.lockinController()
print li.get_idn()

mmw_freqs = np.linspace(140e9,161e9,1024)

while True:
    zbd = []
    tstart = time.time()
    for mmw_freq in mmw_freqs:
        hmc.set_freq(mmw_freq/12.)
        time.sleep(0.5)
        r,_,_,_ = li.get_data()
        zbd.append(r)
        print ("\r%d/%d" % (len(zbd),len(mmw_freqs))),
        sys.stdout.flush()
    zbd = np.array(zbd)
    tend = time.time()
    fn = time.strftime("%Y-%m-%d_%H%M%S")
    np.savez(('/data/readout/mmw_sweeps/%s_cryo_waveguide_short.npz' % fn),
             start_epoch=tstart, end_epoch = tend,
             mmw_freqs=mmw_freqs, zbd = zbd)
    print fn
