import time

from kid_readout.interactive import *

#setup = hardware.Hardware()

ri = Roach2Baseband()

ri.set_modulation_output('high')

ncycle = 0

while True:
    print "cycle",ncycle
    print " "
    for dac_atten in [0]:
        ri.set_dac_atten(dac_atten)

        df = acquire.new_nc_file(suffix='vna_dac_atten_%.1f_dB' % dac_atten)
        swa = acquire.run_sweep(ri,np.linspace(100,180,64)[None,:]+np.arange(650,dtype='int')[:,None]*512./2.**18,
                                2**18,
                                verbose=True,length_seconds=.1,
                                )
        df.write(swa)
        df.close()
        print "waiting 20 minutes"
        time.sleep(1200)

    ncycle += 1

#for example
#170-230 MHz band, steps are (230-170)/128
#then sampling 480 times between each of these steps by stepping an additional 2**18