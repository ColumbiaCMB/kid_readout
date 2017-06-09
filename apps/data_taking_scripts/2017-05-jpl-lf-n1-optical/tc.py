from kid_readout.interactive import *
import time

#setup = hardware.Hardware()

ri = Roach2Baseband()

ri.set_modulation_output('high')

initial_f0s = np.linspace(20, 240, 16)

nf = len(initial_f0s)
atonce = 16
if nf % atonce > 0:
    print "extending list of resonators to make a multiple of ", atonce
    initial_f0s = np.concatenate((initial_f0s, np.arange(1, 1 + atonce - (nf % atonce)) + initial_f0s.max()))

nsamp = 2**19 #going above 2**18 with 128 simultaneous tones doesn't quite work yet
offsets = np.arange(-16,16)*512./nsamp

ri.set_dac_atten(20)

ncf = new_nc_file(suffix='Tc')
ri.set_tone_baseband_freqs(initial_f0s, nsamp = nsamp)

try:
    while True:
        tic = time.time()

        measurement = ri.get_measurement(num_seconds=.1)

        ncf.write(measurement)

        print "sleeping for 30 seconds"
        time.sleep(30)

except KeyboardInterrupt:
    ncf.close()
